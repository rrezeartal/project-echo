"""Load the Albanian and Macedonian ASR datasets into one shared schema."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from datasets import Audio, Dataset, DatasetDict, Value, concatenate_datasets, load_dataset
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16_000
LANGUAGES = {"sq": "albanian", "mk": "macedonian"}
TEXT_COLUMN_CANDIDATES = ("sentence", "text", "transcript", "transcription", "raw_transcription")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    repo_id: str
    train_split: str
    test_split: str | None = None
    config: str | None = None
    text_column: str | None = None
    optional: bool = False


SOURCES: dict[str, list[SourceSpec]] = {
    "sq": [
        SourceSpec("cv24", "Kushtrim/common_voice_24_sq", "train", "test", text_column="sentence"),
        # Gated and schema unverified: skipped unless it exposes raw audio plus a text column.
        SourceSpec("akadriu10k", "akadriu/albanian-asr-10k-processed", "train", optional=True),
    ],
    "mk": [
        SourceSpec(
            "shunya", "shunyalabs/macedonian-speech-dataset", "train+validation", "test", text_column="transcript"
        ),
        SourceSpec(
            "fleurs", "google/fleurs", "train+validation", "test", config="mk_mk", text_column="raw_transcription"
        ),
    ],
}

_normalizer = BasicTextNormalizer()


def normalize_text(text: str) -> str:
    return " ".join(_normalizer(text).split())


def audio_to_array(audio) -> np.ndarray:
    """Return a mono float32 array from either a datasets<4 dict or a datasets>=4 AudioDecoder."""
    if isinstance(audio, dict):
        array = audio["array"]
    else:
        array = audio.get_all_samples().data.numpy()
    array = np.asarray(array, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=0)
    return array


def _load_split_strict(spec: SourceSpec, split: str, lang: str) -> Dataset:
    ds = load_dataset(spec.repo_id, spec.config, split=split)
    text_column = spec.text_column or next((c for c in TEXT_COLUMN_CANDIDATES if c in ds.column_names), None)
    if "audio" not in ds.column_names or text_column is None:
        raise ValueError(f"{spec.repo_id} has columns {ds.column_names}; expected 'audio' and a text column")

    ds = ds.select_columns(["audio", text_column])
    if text_column != "text":
        ds = ds.rename_column(text_column, "text")
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.cast_column("text", Value("string"))

    n = len(ds)
    ds = ds.add_column("lang", [lang] * n)
    ds = ds.add_column("source", [spec.name] * n)
    # Stable IDs assigned before filtering so curation results survive re-loading.
    ds = ds.add_column("uid", [f"{spec.name}:{split}:{i}" for i in range(n)])
    return ds.filter(lambda text: bool(text and text.strip()), input_columns=["text"])


def _load_split(spec: SourceSpec, split: str, lang: str) -> Dataset | None:
    try:
        return _load_split_strict(spec, split, lang)
    except Exception as exc:
        if not spec.optional:
            raise
        logger.warning("Skipping optional source %s (%s): %s", spec.repo_id, split, exc)
        return None


def _load_parts(lang: str, part: str) -> Dataset:
    if lang not in SOURCES:
        raise ValueError(f"Unsupported language '{lang}'. Choose from {list(SOURCES)}")
    datasets = []
    for spec in SOURCES[lang]:
        split = spec.train_split if part == "train" else spec.test_split
        if split is None:
            continue
        ds = _load_split(spec, split, lang)
        if ds is not None:
            logger.info("Loaded %s %s: %d rows", spec.name, split, len(ds))
            datasets.append(ds)
    return concatenate_datasets(datasets)


def load_test_split(lang: str) -> Dataset:
    return _load_parts(lang, "test")


def remove_test_overlap(train: Dataset, test: Dataset) -> Dataset:
    test_texts = {normalize_text(t) for t in test["text"]}
    before = len(train)
    train = train.filter(lambda text: normalize_text(text) not in test_texts, input_columns=["text"])
    logger.info("Removed %d training rows whose transcript also appears in the test set", before - len(train))
    return train


def load_language_splits(lang: str) -> DatasetDict:
    """Train/test splits with columns audio (16 kHz), text, lang, source, uid."""
    test = load_test_split(lang)
    train = remove_test_overlap(_load_parts(lang, "train"), test)
    return DatasetDict(train=train, test=test)


def split_validation(train: Dataset, n_val: int = 500, seed: int = 42) -> tuple[Dataset, Dataset]:
    n_val = min(n_val, max(1, len(train) // 20))
    order = np.random.default_rng(seed).permutation(len(train))
    return train.select(np.sort(order[n_val:])), train.select(np.sort(order[:n_val]))


def filter_by_uids(ds: Dataset, uids) -> Dataset:
    keep = set(uids)
    return ds.filter(lambda uid: uid in keep, input_columns=["uid"])
