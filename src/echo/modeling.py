"""Whisper-medium + LoRA: model construction, data collator and inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, WhisperForConditionalGeneration, WhisperProcessor

from .data import SAMPLING_RATE, audio_to_array

BASE_MODEL_ID = "openai/whisper-medium"
MAX_LABEL_LENGTH = 448
CHUNK_SAMPLES = 30 * SAMPLING_RATE
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]


def load_processor(language: str | None = None) -> WhisperProcessor:
    if language is None:
        return WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    return WhisperProcessor.from_pretrained(BASE_MODEL_ID, language=language, task="transcribe")


def _set_generation_language(model: WhisperForConditionalGeneration, language: str) -> None:
    model.generation_config.language = language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None


def build_lora_model(
    language: str,
    load_in_8bit: bool = False,
    r: int = 32,
    alpha: int = 64,
    dropout: float = 0.05,
) -> PeftModel:
    if load_in_8bit:
        model = WhisperForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID, dtype=dtype)

    model.config.use_cache = False
    _set_generation_language(model, language)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Computes log-Mel features on the fly so no per-clip spectrograms are cached on disk."""

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audio = [audio_to_array(f["audio"]) for f in features]
        batch = self.processor.feature_extractor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt")

        labels_batch = self.processor.tokenizer([f["text"] for f in features], padding=True, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

        # The model prepends the start token itself during training.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def load_for_inference(adapters: dict[str, str | Path] | None = None):
    """Load whisper-medium once and attach each LoRA adapter under its own name."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID, dtype=dtype).to(device)
    model.generation_config.forced_decoder_ids = None

    if adapters:
        (first_name, first_path), *rest = adapters.items()
        model = PeftModel.from_pretrained(model, str(first_path), adapter_name=first_name)
        for name, path in rest:
            model.load_adapter(str(path), adapter_name=name)

    model.eval()
    return model


def _device_and_dtype(model) -> tuple[torch.device, torch.dtype]:
    base = model.get_base_model() if isinstance(model, PeftModel) else model
    weight = base.model.encoder.conv1.weight
    return weight.device, weight.dtype


@torch.inference_mode()
def generate_batch(model, processor, audios: list[np.ndarray], language: str, max_new_tokens: int = 225) -> list[str]:
    """Short-form (<=30 s) batched transcription, used for evaluation."""
    device, dtype = _device_and_dtype(model)
    inputs = processor.feature_extractor(audios, sampling_rate=SAMPLING_RATE, return_tensors="pt")
    ids = model.generate(
        input_features=inputs.input_features.to(device, dtype),
        language=language,
        task="transcribe",
        max_new_tokens=max_new_tokens,
    )
    return [text.strip() for text in processor.batch_decode(ids, skip_special_tokens=True)]


@torch.inference_mode()
def _transcribe(model, processor, audio: np.ndarray, language: str) -> str:
    device, dtype = _device_and_dtype(model)
    long_form = len(audio) > CHUNK_SAMPLES
    extra = {"truncation": False, "padding": "longest"} if long_form else {}
    inputs = processor.feature_extractor(
        audio, sampling_rate=SAMPLING_RATE, return_tensors="pt", return_attention_mask=True, **extra
    )
    output = model.generate(
        input_features=inputs.input_features.to(device, dtype),
        attention_mask=inputs.attention_mask.to(device),
        language=language,
        task="transcribe",
        return_timestamps=long_form,
    )
    ids = output if isinstance(output, torch.Tensor) else output["sequences"]
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def transcribe(model, processor, audio: np.ndarray, language: str, adapter: str | None = None) -> str:
    """Transcribe audio of any length; Whisper handles >30 s input as sequential 30 s windows."""
    if isinstance(model, PeftModel):
        if adapter is None:
            with model.disable_adapter():
                return _transcribe(model, processor, audio, language)
        model.set_adapter(adapter)
    elif adapter is not None:
        raise ValueError(f"Adapter '{adapter}' requested but no adapters are loaded")
    return _transcribe(model, processor, audio, language)
