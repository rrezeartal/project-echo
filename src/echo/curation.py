"""LaBSE (contextual) sentence embeddings for transcript curation and semantic scoring."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from . import DATA_DIR

LABSE_MODEL_ID = "sentence-transformers/LaBSE"
CURATION_DIR = DATA_DIR / "curation"


@dataclass
class CurationResult:
    lang: str
    params: dict
    stats: dict
    kept_uids: list[str]
    near_duplicate_uids: list[str]
    outlier_uids: list[str]


def load_labse(device: str | None = None) -> SentenceTransformer:
    return SentenceTransformer(LABSE_MODEL_ID, device=device)


def embed_texts(model: SentenceTransformer, texts, batch_size: int = 256, show_progress: bool = True) -> np.ndarray:
    return model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )


def pairwise_similarity(model: SentenceTransformer, a, b, batch_size: int = 256) -> np.ndarray:
    ea = embed_texts(model, a, batch_size, show_progress=False)
    eb = embed_texts(model, b, batch_size, show_progress=False)
    return np.sum(ea * eb, axis=1)


def group_near_duplicates(
    embeddings: np.ndarray, threshold: float = 0.90, chunk_size: int = 1024, device: str | None = None
) -> np.ndarray:
    """Greedy leader clustering: each unassigned row claims every unassigned row with cosine >= threshold."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    emb = torch.from_numpy(embeddings).to(device)
    n = emb.shape[0]
    groups = torch.full((n,), -1, dtype=torch.long, device=device)
    for start in range(0, n, chunk_size):
        sims = emb[start : start + chunk_size] @ emb.T
        for offset in range(sims.shape[0]):
            i = start + offset
            if groups[i] >= 0:
                continue
            groups[(sims[offset] >= threshold) & (groups < 0)] = i
            groups[i] = i
    return groups.cpu().numpy()


def curate(
    lang: str,
    uids: list[str],
    embeddings: np.ndarray,
    dup_threshold: float = 0.90,
    max_per_group: int = 3,
    outlier_quantile: float = 0.02,
    seed: int = 42,
) -> CurationResult:
    n = len(uids)
    centroid = embeddings.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    centroid_sim = embeddings @ centroid
    outlier_cutoff = float(np.quantile(centroid_sim, outlier_quantile))
    is_outlier = centroid_sim < outlier_cutoff

    groups = group_near_duplicates(embeddings, dup_threshold)
    candidates = np.flatnonzero(~is_outlier)
    order = candidates[np.argsort(groups[candidates], kind="stable")]
    _, starts = np.unique(groups[order], return_index=True)

    rng = np.random.default_rng(seed)
    is_duplicate = np.zeros(n, dtype=bool)
    groups_over_cap = 0
    for members in np.split(order, starts[1:]):
        if len(members) > max_per_group:
            groups_over_cap += 1
            is_duplicate[rng.permutation(members)[max_per_group:]] = True

    kept = ~is_outlier & ~is_duplicate
    uid_array = np.asarray(uids)
    return CurationResult(
        lang=lang,
        params={
            "model": LABSE_MODEL_ID,
            "dup_threshold": dup_threshold,
            "max_per_group": max_per_group,
            "outlier_quantile": outlier_quantile,
            "seed": seed,
        },
        stats={
            "n_input": n,
            "n_kept": int(kept.sum()),
            "n_outliers": int(is_outlier.sum()),
            "n_near_duplicates": int(is_duplicate.sum()),
            "n_groups": len(starts),
            "n_groups_over_cap": groups_over_cap,
            "outlier_cutoff": outlier_cutoff,
        },
        kept_uids=uid_array[kept].tolist(),
        near_duplicate_uids=uid_array[is_duplicate].tolist(),
        outlier_uids=uid_array[is_outlier].tolist(),
    )


def curation_path(lang: str) -> Path:
    return CURATION_DIR / f"{lang}_labse.json"


def save_curation(result: CurationResult) -> Path:
    path = curation_path(result.lang)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_curation(lang: str) -> CurationResult:
    return CurationResult(**json.loads(curation_path(lang).read_text(encoding="utf-8")))


def run_curation(lang: str, train_ds, **params) -> CurationResult:
    """Embed the training transcripts with LaBSE, curate, and save the result to data/curation."""
    model = load_labse()
    embeddings = embed_texts(model, train_ds["text"])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result = curate(lang, list(train_ds["uid"]), embeddings, **params)
    save_curation(result)
    return result
