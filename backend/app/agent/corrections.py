"""Deterministic safety net for caller self-corrections.

The model is instructed that the later statement wins. This module double-checks
its citations: if the same speaker says a correction cue ("no sorry", "je veux
dire", "لا عفوا", ...) shortly *after* the end of the cited evidence, the value
may be the superseded one. The field is then flagged with a warning and its
confidence capped, so a reviewer looks at it. The value itself is never changed
here — that stays the model's (or the human's) call.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Segment
from app.rag.search import segments_between

CORRECTION_WINDOW_MS = 8_000

_CUES_RAW = {
    "en": [
        "no sorry", "sorry no", "sorry i mean", "i mean", "i meant", "actually", "correction", "scratch that",
        "no wait", "let me correct", "my mistake", "i got that wrong",
    ],
    "fr": [
        "non pardon", "pardon", "je veux dire", "enfin non", "non non", "en fait", "plutot",
        "je me suis trompe", "je me suis trompee", "je rectifie", "je corrige", "excusez moi",
    ],
    "ar": ["لا عفوا", "عفوا", "اقصد", "اعني", "في الحقيقة", "بالاحرى", "اصحح", "غلطت", "لا لا"],
}

_WORD = re.compile(r"\w+", re.UNICODE)


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


def _tokens(text: str) -> list[str]:
    return _WORD.findall(_fold(text))


CUES: list[tuple[str, ...]] = sorted(
    {tuple(_tokens(c)) for cues in _CUES_RAW.values() for c in cues}, key=len, reverse=True
)


def find_correction_cues(segment: Segment | Any) -> list[dict[str, Any]]:
    """Correction cues in a segment, with the time (ms) at which each is spoken."""
    timed: list[tuple[str, int]] = []
    words = getattr(segment, "words", None) or []
    if words:
        for w in words:
            for tok in _tokens(w.get("w", "")):
                timed.append((tok, int(w.get("s", segment.start_ms))))
    else:
        timed = [(tok, segment.start_ms) for tok in _tokens(segment.text)]
    found, i = [], 0
    toks = [t for t, _ in timed]
    while i < len(toks):
        for cue in CUES:
            if tuple(toks[i : i + len(cue)]) == cue:
                found.append({"time_ms": timed[i][1], "cue": " ".join(cue), "speaker": segment.speaker})
                i += len(cue) - 1
                break
        i += 1
    return found


def superseding_correction(
    session: Session, call_id: int, cited: list[Segment], evidence_end_ms: int, window_ms: int = CORRECTION_WINDOW_MS
) -> dict[str, Any] | None:
    """First correction cue by the cited speaker(s) within ``window_ms`` after the evidence ends.

    Scanning stops at the first turn by another speaker that starts after the
    evidence: once the agent has moved the conversation on, a later "actually"
    is most likely about something else.
    """
    speakers = {s.speaker for s in cited}
    for seg in segments_between(session, call_id, evidence_end_ms, evidence_end_ms + window_ms):
        if seg.speaker not in speakers:
            if seg.start_ms >= evidence_end_ms:
                break
            continue
        for cue in find_correction_cues(seg):
            if evidence_end_ms <= cue["time_ms"] <= evidence_end_ms + window_ms:
                return cue
    return None
