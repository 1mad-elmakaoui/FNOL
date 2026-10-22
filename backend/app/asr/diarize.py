"""Speaker diarization.

* With ``HF_TOKEN`` set: pyannote.audio (``pyannote/speaker-diarization-3.1`` by
  default), run locally — the token is only used to download the gated weights.
* Without it: a simple energy-based two-speaker split, which is logged loudly.

Fallback algorithm
------------------
Speech spans come from the ASR word timestamps (words separated by < 400 ms are
one utterance). Then:

1. **Stereo recordings** (typical of call-centre systems: agent on one channel,
   caller on the other) — each word goes to the channel with the higher RMS
   energy, and consecutive same-channel words form a turn. This is reliable when
   the channels actually differ, even with very short turn-taking gaps.
2. **Mono recordings** — per-utterance features (log energy, spectral centroid,
   zero-crossing rate) are standardised and split with 2-means. This is a
   heuristic: it works when the two voices/lines differ in level or timbre (e.g.
   a headset agent vs. a mobile caller) and degrades gracefully otherwise.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.config import get_settings

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
UTTERANCE_GAP_MS = 400


@dataclass
class Turn:
    start_ms: int
    end_ms: int
    speaker: str


# --------------------------------------------------------------------------- spans
def speech_spans(word_times: list[tuple[int, int]], gap_ms: int = UTTERANCE_GAP_MS) -> list[tuple[int, int]]:
    """Merge word (start, end) pairs into utterance spans split on pauses."""
    spans: list[list[int]] = []
    for start, end in sorted(word_times):
        if spans and start - spans[-1][1] <= gap_ms:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    return [(s, e) for s, e in spans]


def _slice(audio: np.ndarray, start_ms: int, end_ms: int) -> np.ndarray:
    a = int(start_ms * SAMPLE_RATE / 1000)
    b = max(a + 1, int(end_ms * SAMPLE_RATE / 1000))
    return audio[a:b]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def _features(x: np.ndarray) -> np.ndarray:
    if x.size < 32:
        return np.zeros(3)
    log_energy = 20 * np.log10(_rms(x) + 1e-8)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    freqs = np.fft.rfftfreq(x.size, 1 / SAMPLE_RATE)
    centroid = float((spectrum * freqs).sum() / (spectrum.sum() + 1e-8))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x).astype(np.int8)))))
    return np.array([log_energy, centroid, zcr])


def _two_means(features: np.ndarray, iters: int = 50) -> np.ndarray:
    """Deterministic 2-means on standardised features. Returns labels 0/1."""
    n = features.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    z = (features - features.mean(axis=0)) / std
    # Initialise on the extremes of the dominant (first principal) direction.
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    proj = z @ vt[0]
    centers = np.stack([z[proj.argmin()], z[proj.argmax()]])
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for k in (0, 1):
            if np.any(labels == k):
                centers[k] = z[labels == k].mean(0)
    return labels


def _stereo_is_distinct(left: np.ndarray, right: np.ndarray) -> bool:
    if left.size == 0 or right.size == 0:
        return False
    n = min(left.size, right.size)
    l, r = left[:n], right[:n]
    if _rms(l) < 1e-6 or _rms(r) < 1e-6:
        return True
    corr = float(np.corrcoef(l, r)[0, 1])
    return corr < 0.95


def _merge_labelled(units: list[tuple[int, int]], labels: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Merge consecutive units that share a label into one turn."""
    merged: list[list[int]] = []
    merged_labels: list[int] = []
    for (s, e), lab in zip(units, labels):
        if merged and merged_labels[-1] == lab:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
            merged_labels.append(int(lab))
    return [(s, e) for s, e in merged], np.array(merged_labels)


def energy_diarize(
    spans: list[tuple[int, int]],
    mono: np.ndarray,
    stereo: tuple[np.ndarray, np.ndarray] | None = None,
    word_spans: list[tuple[int, int]] | None = None,
) -> tuple[list[Turn], str]:
    """Assign speech to SPEAKER_00 / SPEAKER_01. SPEAKER_00 is whoever speaks first.

    Stereo decisions are made per word when word timings are given, because
    turn-taking gaps on a phone line are often shorter than the utterance gap.
    """
    if not spans:
        return [], "energy-empty"
    if stereo is not None and _stereo_is_distinct(*stereo):
        left, right = stereo
        units = sorted(word_spans) if word_spans else spans
        labels = np.array(
            [0 if _rms(_slice(left, s, e)) >= _rms(_slice(right, s, e)) else 1 for s, e in units]
        )
        spans, labels = _merge_labelled(units, labels)
        method = "energy-stereo"
    else:
        feats = np.stack([_features(_slice(mono, s, e)) for s, e in spans])
        labels = _two_means(feats)
        method = "energy-mono"
    # Relabel so the first speaker is SPEAKER_00 (stable, human-friendly ordering).
    first = labels[0]
    turns = [
        Turn(start_ms=s, end_ms=e, speaker="SPEAKER_00" if lab == first else "SPEAKER_01")
        for (s, e), lab in zip(spans, labels)
    ]
    return turns, method


# ------------------------------------------------------------------------ pyannote
def _pyannote_diarize(mono: np.ndarray) -> list[Turn]:
    import torch
    from pyannote.audio import Pipeline

    settings = get_settings()
    pipeline = Pipeline.from_pretrained(settings.pyannote_model, use_auth_token=settings.hf_token)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    waveform = torch.from_numpy(np.ascontiguousarray(mono)).unsqueeze(0)
    annotation = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE}, min_speakers=1, max_speakers=3)
    turns = [
        Turn(start_ms=int(seg.start * 1000), end_ms=int(seg.end * 1000), speaker=str(label))
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t.start_ms)
    return turns


def diarize(
    spans: list[tuple[int, int]],
    mono: np.ndarray,
    stereo: tuple[np.ndarray, np.ndarray] | None = None,
    word_spans: list[tuple[int, int]] | None = None,
) -> tuple[list[Turn], str]:
    settings = get_settings()
    if settings.hf_token:
        try:
            turns = _pyannote_diarize(mono)
            if turns:
                return turns, "pyannote"
            log.warning("pyannote returned no speaker turns; falling back to energy-based split")
        except Exception as exc:  # model download / auth / runtime failure
            log.warning("pyannote diarization failed (%s); falling back to energy-based split", exc)
    else:
        log.warning(
            "HF_TOKEN is not set: pyannote diarization disabled. Using the simple energy-based "
            "two-speaker split instead (less accurate, see README)."
        )
    return energy_diarize(spans, mono, stereo, word_spans)
