"""Chainlit ASR demo for base and LoRA-tuned whisper-medium (Albanian / Macedonian).

Run from the repo root:
    chainlit run app/app.py
"""

from __future__ import annotations

import io
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import chainlit as cl  # noqa: E402
import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from chainlit.input_widget import Select  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from echo import OUTPUTS_DIR  # noqa: E402
from echo.data import LANGUAGES, SAMPLING_RATE  # noqa: E402
from echo.modeling import load_for_inference, load_processor, transcribe  # noqa: E402

load_dotenv(ROOT / ".env")

LANGUAGE_OPTIONS = {"Albanian": "sq", "Macedonian": "mk"}
MODEL_OPTIONS = {"Base (zero-shot)": None, "LoRA": "raw", "LoRA + LaBSE": "labse"}


def _discover_adapters() -> dict[str, Path]:
    adapters = {}
    for lang in LANGUAGES:
        for arm in ("raw", "labse"):
            path = OUTPUTS_DIR / f"whisper-medium-lora-{lang}-{arm}"
            if (path / "adapter_config.json").exists():
                adapters[f"{lang}_{arm}"] = path
    return adapters


ADAPTERS = _discover_adapters()
MODEL = load_for_inference(ADAPTERS)
PROCESSOR = load_processor()
# One shared model: adapter switching and generation must not interleave across sessions.
MODEL_LOCK = threading.Lock()


def _run_transcription(audio: np.ndarray, lang: str, adapter: str | None) -> str:
    with MODEL_LOCK:
        return transcribe(MODEL, PROCESSOR, audio.astype(np.float32), LANGUAGES[lang], adapter)


def _selection() -> tuple[str, str, str, str | None]:
    settings = cl.user_session.get("settings") or {}
    language_label = settings.get("language", next(iter(LANGUAGE_OPTIONS)))
    model_label = settings.get("model", next(iter(MODEL_OPTIONS)))
    lang = LANGUAGE_OPTIONS[language_label]
    arm = MODEL_OPTIONS[model_label]
    adapter = None if arm is None else f"{lang}_{arm}"
    return language_label, model_label, lang, adapter


async def _transcribe_and_reply(audio: np.ndarray) -> None:
    language_label, model_label, lang, adapter = _selection()
    if adapter is not None and adapter not in ADAPTERS:
        await cl.Message(
            content=f"No trained adapter for **{language_label} / {model_label}**. "
            f"Train it with `scripts/train_lora.py` or pick another model in the settings."
        ).send()
        return
    if audio.size == 0:
        await cl.Message(content="The recording was empty.").send()
        return

    reply = cl.Message(content=f"Transcribing with **{model_label}** ({language_label})...")
    await reply.send()
    text = await cl.make_async(_run_transcription)(audio, lang, adapter)
    reply.content = f"**{language_label} · {model_label}**\n\n{text or '_(no speech detected)_'}"
    await reply.update()


@cl.on_chat_start
async def on_chat_start() -> None:
    settings = await cl.ChatSettings(
        [
            Select(id="language", label="Language", values=list(LANGUAGE_OPTIONS), initial_index=0),
            Select(id="model", label="Model", values=list(MODEL_OPTIONS), initial_index=0),
        ]
    ).send()
    cl.user_session.set("settings", settings)

    available = ", ".join(sorted(ADAPTERS)) or "none (only the base model is available)"
    await cl.Message(
        content="Record with the microphone or upload an audio file to transcribe it. "
        "Choose the language and model in the settings panel.\n\n"
        f"Loaded LoRA adapters: {available}"
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    cl.user_session.set("settings", settings)


@cl.on_audio_start
async def on_audio_start() -> bool:
    cl.user_session.set("audio_chunks", [])
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk) -> None:
    cl.user_session.get("audio_chunks").append(chunk.data)


@cl.on_audio_end
async def on_audio_end() -> None:
    chunks = cl.user_session.get("audio_chunks") or []
    cl.user_session.set("audio_chunks", [])
    # Chainlit streams 16-bit PCM at the sample rate set in .chainlit/config.toml.
    audio = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32) / 32768.0

    wav = io.BytesIO()
    sf.write(wav, audio, SAMPLING_RATE, format="WAV")
    await cl.Message(
        author="You",
        type="user_message",
        content="",
        elements=[cl.Audio(name="recording.wav", content=wav.getvalue(), mime="audio/wav")],
    ).send()

    await _transcribe_and_reply(audio)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    files = [el for el in message.elements if (el.mime or "").startswith("audio/") and el.path]
    if not files:
        await cl.Message(content="Please upload an audio file or use the microphone.").send()
        return

    for element in files:
        try:
            audio, _ = await cl.make_async(librosa.load)(element.path, sr=SAMPLING_RATE, mono=True)
        except Exception as exc:
            await cl.Message(content=f"Could not decode `{element.name}`: {exc}").send()
            continue
        await _transcribe_and_reply(audio)
