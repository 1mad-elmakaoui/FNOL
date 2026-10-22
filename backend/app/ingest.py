"""Persist a diarized transcript: segments, retrieval chunks and their embeddings.

Shared by the ASR worker, the seed script and the eval harness (which injects
synthetic transcripts directly, bypassing ASR).
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Call, Chunk, Segment
from app.rag.chunk import chunk_segments
from app.rag.embed import embed_texts

log = logging.getLogger(__name__)


def synthesize_words(text: str, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    """Evenly spread word timestamps over a segment (for fixtures without word timings)."""
    tokens = text.split()
    if not tokens:
        return []
    total_chars = sum(len(t) for t in tokens) or 1
    span = max(end_ms - start_ms, len(tokens))
    words, cursor = [], float(start_ms)
    for tok in tokens:
        dur = span * len(tok) / total_chars
        words.append({"w": tok, "s": int(cursor), "e": int(cursor + dur), "p": 1.0})
        cursor += dur
    words[-1]["e"] = end_ms
    return words


def store_transcript(session: Session, call: Call, segments: Iterable[Mapping[str, Any]]) -> int:
    """Replace the call's segments/chunks with ``segments``. Returns the number of chunks."""
    settings = get_settings()
    session.execute(delete(Chunk).where(Chunk.call_id == call.id))
    session.execute(delete(Segment).where(Segment.call_id == call.id))
    session.flush()

    rows: list[Segment] = []
    for raw in sorted(segments, key=lambda s: (int(s["start_ms"]), int(s["end_ms"]))):
        text = str(raw["text"]).strip()
        if not text:
            continue
        start, end = int(raw["start_ms"]), int(raw["end_ms"])
        rows.append(
            Segment(
                call_id=call.id,
                speaker=str(raw.get("speaker") or "unknown"),
                start_ms=start,
                end_ms=end,
                text=text,
                words=list(raw.get("words") or synthesize_words(text, start, end)),
            )
        )
    session.add_all(rows)
    session.flush()

    drafts = chunk_segments(rows, settings.chunk_max_words, settings.chunk_overlap_segments)
    vectors = embed_texts([d.text for d in drafts])
    session.add_all(
        Chunk(call_id=call.id, text=d.text, start_ms=d.start_ms, end_ms=d.end_ms, embedding=vec.tolist())
        for d, vec in zip(drafts, vectors)
    )
    if rows and not call.duration:
        call.duration = max(r.end_ms for r in rows) / 1000.0
    session.flush()
    session.expire(call, ["segments"])
    log.info("Stored call %s: %d segments, %d chunks", call.id, len(rows), len(drafts))
    return len(drafts)
