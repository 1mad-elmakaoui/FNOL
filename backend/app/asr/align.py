"""Merge ASR words with diarization turns into speaker segments."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.asr.diarize import Turn
from app.asr.transcribe import Word

SEGMENT_PAUSE_MS = 1200
SEGMENT_SOFT_MAX_WORDS = 30
_SENTENCE_END = (".", "?", "!", "…", "؟")


@dataclass
class SegmentDraft:
    speaker: str
    start_ms: int
    end_ms: int
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(w.text for w in self.words).strip()

    def words_json(self) -> list[dict]:
        return [
            {"w": w.text.strip(), "s": w.start_ms, "e": w.end_ms, "p": round(w.prob, 3)} for w in self.words
        ]


def _speaker_for(word: Word, turns: list[Turn]) -> str:
    """Speaker whose turn overlaps the word most; nearest turn if none overlaps."""
    overlap = lambda t: min(word.end_ms, t.end_ms) - max(word.start_ms, t.start_ms)  # noqa: E731
    best = max(turns, key=overlap)
    if overlap(best) > 0:
        return best.speaker
    mid = (word.start_ms + word.end_ms) / 2
    return min(turns, key=lambda t: min(abs(mid - t.start_ms), abs(mid - t.end_ms))).speaker


def assign_roles(speakers_in_order: list[str]) -> dict[str, str]:
    """Heuristic: the insurer answers the phone, so the first voice is the agent.

    The next distinct voice is the caller; any further voices keep their label.
    """
    mapping: dict[str, str] = {}
    for spk in speakers_in_order:
        if spk in mapping:
            continue
        if not mapping:
            mapping[spk] = "agent"
        elif "caller" not in mapping.values():
            mapping[spk] = "caller"
        else:
            mapping[spk] = spk.lower()
    return mapping


def build_segments(words: list[Word], turns: list[Turn]) -> list[SegmentDraft]:
    segments: list[SegmentDraft] = []
    for w in words:
        spk = _speaker_for(w, turns) if turns else "SPEAKER_00"
        cur = segments[-1] if segments else None
        new_segment = (
            cur is None
            or cur.speaker != spk
            or w.start_ms - cur.end_ms > SEGMENT_PAUSE_MS
            or (len(cur.words) >= SEGMENT_SOFT_MAX_WORDS and cur.words[-1].text.strip().endswith(_SENTENCE_END))
        )
        if new_segment:
            segments.append(SegmentDraft(speaker=spk, start_ms=w.start_ms, end_ms=w.end_ms, words=[w]))
        else:
            cur.words.append(w)
            cur.end_ms = max(cur.end_ms, w.end_ms)
    roles = assign_roles([s.speaker for s in segments])
    for s in segments:
        s.speaker = roles.get(s.speaker, s.speaker)
    return segments
