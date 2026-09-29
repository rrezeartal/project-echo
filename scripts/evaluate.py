"""Evaluate base whisper-medium or a LoRA adapter on the fixed test set of one language.

Example:
    python scripts/evaluate.py --lang sq --model base
    python scripts/evaluate.py --lang sq --model labse
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jiwer  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from tqdm import tqdm  # noqa: E402

from echo import OUTPUTS_DIR, RESULTS_DIR  # noqa: E402
from echo.curation import load_labse, pairwise_similarity  # noqa: E402
from echo.data import LANGUAGES, audio_to_array, load_test_split, normalize_text  # noqa: E402
from echo.modeling import generate_batch, load_for_inference, load_processor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", choices=sorted(LANGUAGES), required=True)
    parser.add_argument("--model", choices=["base", "raw", "labse"], required=True)
    parser.add_argument("--adapter_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=225)
    parser.add_argument("--max_samples", type=int, default=None, help="Evaluate a random subset (smoke tests).")
    return parser.parse_args()


def score(refs: list[str], preds: list[str], sims: np.ndarray) -> dict:
    # jiwer rejects empty references, so those pairs are skipped.
    ortho = [(r, p) for r, p in zip(refs, preds) if r.strip()]
    norm = [(nr, normalize_text(p)) for r, p in zip(refs, preds) if (nr := normalize_text(r))]
    return {
        "n": len(refs),
        "wer": 100 * jiwer.wer([r for r, _ in ortho], [p for _, p in ortho]),
        "cer": 100 * jiwer.cer([r for r, _ in ortho], [p for _, p in ortho]),
        "wer_norm": 100 * jiwer.wer([r for r, _ in norm], [p for _, p in norm]),
        "cer_norm": 100 * jiwer.cer([r for r, _ in norm], [p for _, p in norm]),
        "semantic_similarity": float(np.mean(sims)),
    }


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    language = LANGUAGES[args.lang]

    test = load_test_split(args.lang)
    if args.max_samples:
        test = test.shuffle(seed=42).select(range(min(args.max_samples, len(test))))

    adapter_dir = None
    adapters = None
    if args.model != "base":
        adapter_dir = args.adapter_dir or OUTPUTS_DIR / f"whisper-medium-lora-{args.lang}-{args.model}"
        if not (Path(adapter_dir) / "adapter_config.json").exists():
            raise FileNotFoundError(f"No LoRA adapter found in {adapter_dir}. Train it with scripts/train_lora.py first.")
        adapters = {f"{args.lang}_{args.model}": adapter_dir}

    processor = load_processor(language)
    model = load_for_inference(adapters)

    preds, refs, sources, uids = [], [], [], []
    for batch in tqdm(test.iter(batch_size=args.batch_size), total=math.ceil(len(test) / args.batch_size)):
        audios = [audio_to_array(a) for a in batch["audio"]]
        preds += generate_batch(model, processor, audios, language, args.max_new_tokens)
        refs += batch["text"]
        sources += batch["source"]
        uids += batch["uid"]

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sims = pairwise_similarity(load_labse(), preds, refs)

    sources_arr = np.asarray(sources)
    by_source = {}
    for source in sorted(set(sources)):
        idx = np.flatnonzero(sources_arr == source)
        by_source[source] = score([refs[i] for i in idx], [preds[i] for i in idx], sims[idx])

    results = {
        "lang": args.lang,
        "model": args.model,
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "overall": score(refs, preds, sims),
        "by_source": by_source,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.lang}_{args.model}"
    (RESULTS_DIR / f"{stem}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with open(RESULTS_DIR / f"{stem}_predictions.jsonl", "w", encoding="utf-8") as f:
        for uid, source, ref, pred, sim in zip(uids, sources, refs, preds, sims):
            row = {"uid": uid, "source": source, "reference": ref, "prediction": pred, "similarity": float(sim)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(results["overall"], indent=2))


if __name__ == "__main__":
    main()
