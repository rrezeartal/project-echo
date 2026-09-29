"""Fine-tune openai/whisper-medium with LoRA for one language and one data variant.

Example:
    python scripts/train_lora.py --lang sq --arm raw
    python scripts/train_lora.py --lang mk --arm labse --load_in_8bit
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed  # noqa: E402

from echo import OUTPUTS_DIR  # noqa: E402
from echo.curation import curation_path, load_curation, run_curation  # noqa: E402
from echo.data import LANGUAGES, filter_by_uids, load_language_splits, split_validation  # noqa: E402
from echo.modeling import (  # noqa: E402
    MAX_LABEL_LENGTH,
    DataCollatorSpeechSeq2SeqWithPadding,
    build_lora_model,
    load_processor,
)

logger = logging.getLogger("train_lora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", choices=sorted(LANGUAGES), required=True)
    parser.add_argument("--arm", choices=["raw", "labse"], required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default=None)
    return parser.parse_args()


def labse_kept_uids(lang: str, train_full) -> list[str]:
    if curation_path(lang).exists():
        result = load_curation(lang)
        if result.stats["n_input"] == len(train_full):
            return result.kept_uids
        logger.warning("Curation file for '%s' does not match the current training set; recomputing", lang)
    return run_curation(lang, train_full).kept_uids


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    set_seed(args.seed)

    language = LANGUAGES[args.lang]
    run_name = f"whisper-medium-lora-{args.lang}-{args.arm}"
    output_dir = args.output_dir or OUTPUTS_DIR / run_name

    processor = load_processor(language)
    train_full = load_language_splits(args.lang)["train"]
    # Validation rows are drawn before curation so both arms are validated on identical data.
    train, val = split_validation(train_full, n_val=args.n_val, seed=args.seed)
    if args.arm == "labse":
        train = filter_by_uids(train, labse_kept_uids(args.lang, train_full))

    tokenizer = processor.tokenizer

    def fits(text: str) -> bool:
        return len(tokenizer(text).input_ids) <= MAX_LABEL_LENGTH

    train = train.filter(fits, input_columns=["text"])
    val = val.filter(fits, input_columns=["text"])
    logger.info("Train rows: %d | validation rows: %d", len(train), len(val))

    model = build_lora_model(language, load_in_8bit=args.load_in_8bit)
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=model.config.decoder_start_token_id
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        logging_steps=25,
        predict_with_generate=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=0,
        report_to=["tensorboard"],
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id or (run_name if args.push_to_hub else None),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=collator,
        processing_class=processor.feature_extractor,
    )
    trainer.train()

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    run_info = {
        "lang": args.lang,
        "arm": args.arm,
        "n_train": len(train),
        "n_val": len(val),
        "best_eval_loss": trainer.state.best_metric,
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (Path(output_dir) / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    logger.info("Saved adapter to %s", output_dir)

    if args.push_to_hub:
        trainer.push_to_hub(
            finetuned_from="openai/whisper-medium",
            tasks="automatic-speech-recognition",
            language=args.lang,
            model_name=f"Whisper Medium LoRA {args.lang.upper()} ({args.arm})",
        )


if __name__ == "__main__":
    main()
