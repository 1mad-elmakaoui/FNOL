"""Local speech-to-text with faster-whisper. Audio never leaves the machine.

Heavy imports (faster_whisper / ctranslate2) are done lazily so the API, the
tests and the eval harness can import this package without the ML stack.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import get_settings

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


@dataclass
class Word:
    text: str
    start_ms: int
    end_ms: int
    prob: float = 1.0


@dataclass
class AsrResult:
    language: str
    language_probability: float
    duration_s: float
    words: list[Word] = field(default_factory=list)


def detect_device() -> tuple[str, str]:
    """Return (device, compute_type). GPU is used automatically when CUDA is visible."""
    settings = get_settings()
    device = settings.whisper_device
    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:  # pragma: no cover - depends on the host
            device = "cpu"
    compute_type = settings.whisper_compute_type
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


@lru_cache(maxsize=1)
def _load_model():
    from faster_whisper import WhisperModel

    settings = get_settings()
    device, compute_type = detect_device()
    log.info("Loading faster-whisper model=%s device=%s compute_type=%s", settings.whisper_model, device, compute_type)
    try:
        return WhisperModel(settings.whisper_model, device=device, compute_type=compute_type)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Whisper model {settings.whisper_model!r} on {device}. The first run downloads it from "
            f"Hugging Face into the models volume, so network access is needed once. ({type(exc).__name__}: {exc})"
        ) from exc


def load_audio(path: str | Path, split_stereo: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Decode any container/codec to 16 kHz float32 using PyAV (bundled with faster-whisper)."""
    from faster_whisper.audio import decode_audio

    return decode_audio(str(path), sampling_rate=SAMPLE_RATE, split_stereo=split_stereo)


def transcribe(path: str | Path) -> AsrResult:
    settings = get_settings()
    model = _load_model()
    segments, info = model.transcribe(
        str(path),
        language=settings.whisper_language,
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=False,
    )
    words: list[Word] = []
    for seg in segments:  # generator: decoding happens here
        for w in seg.words or []:
            text = w.word
            if not text.strip():
                continue
            words.append(
                Word(
                    text=text,
                    start_ms=int(round(w.start * 1000)),
                    end_ms=int(round(w.end * 1000)),
                    prob=float(w.probability),
                )
            )
    log.info(
        "Transcribed %s: language=%s (p=%.2f) duration=%.1fs words=%d",
        Path(path).name,
        info.language,
        info.language_probability,
        info.duration,
        len(words),
    )
    return AsrResult(
        language=info.language,
        language_probability=float(info.language_probability),
        duration_s=float(info.duration),
        words=words,
    )
