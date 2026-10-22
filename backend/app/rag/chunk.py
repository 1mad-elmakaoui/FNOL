"""Split a diarized transcript into overlapping retrieval chunks.

Chunks are windows of consecutive segments (never splitting a segment), so each
chunk has exact start/end timestamps and keeps a speaker's correction together
with the statement it corrects in most cases. Consecutive windows overlap by
``overlap_segments`` segments so a fact spanning a window boundary is still
retrievable in one piece.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


class SegmentLike(Protocol):
    speaker: str
    start_ms: int
    end_ms: int
    text: str


@dataclass
class ChunkDraft:
    text: str
    start_ms: int
    end_ms: int
    first_segment: int
    last_segment: int


def format_segment(seg: SegmentLike) -> str:
    return f"{seg.speaker}: {seg.text.strip()}"


def chunk_segments(
    segments: Sequence[SegmentLike], max_words: int = 80, overlap_segments: int = 1
) -> list[ChunkDraft]:
    if max_words < 1:
        raise ValueError("max_words must be >= 1")
    overlap_segments = max(0, overlap_segments)
    chunks: list[ChunkDraft] = []
    n = len(segments)
    i = 0
    while i < n:
        j, words = i, 0
        while j < n:
            seg_words = len(segments[j].text.split())
            if j > i and words + seg_words > max_words:
                break
            words += seg_words
            j += 1
        window = segments[i:j]
        chunks.append(
            ChunkDraft(
                text="\n".join(format_segment(s) for s in window),
                start_ms=min(s.start_ms for s in window),
                end_ms=max(s.end_ms for s in window),
                first_segment=i,
                last_segment=j - 1,
            )
        )
        if j >= n:
            break
        # Step forward, keeping `overlap_segments` of context, but always advance.
        i = max(i + 1, j - overlap_segments)
    return chunks
