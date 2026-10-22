"""pgvector search over transcript chunks, and verbatim transcript slicing."""
from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Segment
from app.rag.embed import embed_texts


def segments_between(session: Session, call_id: int, start_ms: int, end_ms: int) -> list[Segment]:
    """Segments overlapping [start_ms, end_ms] (inclusive), in time order."""
    if end_ms < start_ms:
        start_ms, end_ms = end_ms, start_ms
    stmt = (
        select(Segment)
        .where(Segment.call_id == call_id, Segment.end_ms >= start_ms, Segment.start_ms <= end_ms)
        .order_by(Segment.start_ms, Segment.id)
    )
    return list(session.scalars(stmt))


def verbatim_quote(segments: list[Segment], start_ms: int, end_ms: int) -> str:
    """Exact words spoken in [start_ms, end_ms], using word timestamps when available."""
    parts: list[str] = []
    for seg in segments:
        words = seg.words or []
        inside = [w["w"] for w in words if w.get("e", 0) > start_ms and w.get("s", 0) < end_ms]
        if inside:
            text = " ".join(inside)
            if len(inside) == len(words):
                text = seg.text.strip()  # keep original punctuation/spacing for whole segments
        else:
            text = seg.text.strip()
        if text:
            parts.append(text)
    return " … ".join(parts)


def segment_dict(seg: Segment) -> dict[str, Any]:
    return {"speaker": seg.speaker, "start_ms": seg.start_ms, "end_ms": seg.end_ms, "text": seg.text}


def search_chunks(session: Session, call_id: int, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Cosine-similarity search restricted to one call.

    Each hit carries the chunk text plus the individual segments it spans, so
    the caller gets precise per-utterance timestamps for citations.
    """
    top_k = max(1, min(int(top_k), 20))
    qvec = embed_texts([query])[0]
    distance = Chunk.embedding.cosine_distance(qvec)
    stmt = (
        select(Chunk, distance.label("distance"))
        .where(Chunk.call_id == call_id)
        .order_by(distance, Chunk.start_ms)
        .limit(top_k)
    )
    hits: list[dict[str, Any]] = []
    for chunk, dist in session.execute(stmt):
        segs = segments_between(session, call_id, chunk.start_ms, chunk.end_ms)
        hits.append(
            {
                "chunk_id": chunk.id,
                "score": round(1.0 - float(dist), 4),
                "start_ms": chunk.start_ms,
                "end_ms": chunk.end_ms,
                "text": chunk.text,
                "segments": [segment_dict(s) for s in segs],
            }
        )
    return hits
