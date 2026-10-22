from types import SimpleNamespace

import numpy as np
import pytest

from app.db.models import Call, Chunk, Segment
from app.ingest import store_transcript, synthesize_words
from app.rag.chunk import chunk_segments
from app.rag.embed import HashingEmbedder
from app.rag.search import search_chunks, segments_between, verbatim_quote


def seg(i, words, speaker="caller"):
    return SimpleNamespace(speaker=speaker, start_ms=i * 1000, end_ms=i * 1000 + 900, text=" ".join(["w"] * words))


def test_chunks_cover_every_segment_with_overlap():
    segments = [seg(i, 10) for i in range(20)]
    chunks = chunk_segments(segments, max_words=35, overlap_segments=1)
    covered = set()
    for c in chunks:
        covered.update(range(c.first_segment, c.last_segment + 1))
        assert sum(len(s.text.split()) for s in segments[c.first_segment : c.last_segment + 1]) <= 35
        assert c.start_ms == segments[c.first_segment].start_ms
        assert c.end_ms == segments[c.last_segment].end_ms
    assert covered == set(range(20))
    for a, b in zip(chunks, chunks[1:]):
        assert b.first_segment == a.last_segment  # one segment of overlap
        assert b.first_segment > a.first_segment  # always advances


def test_oversized_segment_is_its_own_chunk():
    segments = [seg(0, 5), seg(1, 200), seg(2, 5)]
    chunks = chunk_segments(segments, max_words=50, overlap_segments=0)
    assert [(c.first_segment, c.last_segment) for c in chunks] == [(0, 0), (1, 1), (2, 2)]


def test_chunk_text_keeps_speaker_labels():
    segments = [SimpleNamespace(speaker="agent", start_ms=0, end_ms=1000, text="Your plate?"),
                SimpleNamespace(speaker="caller", start_ms=1200, end_ms=2500, text="GE-418-TR.")]
    (chunk,) = chunk_segments(segments, max_words=80)
    assert chunk.text == "agent: Your plate?\ncaller: GE-418-TR."


def test_chunk_rejects_bad_config():
    with pytest.raises(ValueError):
        chunk_segments([seg(0, 1)], max_words=0)


def test_hashing_embedder_is_normalized_and_lexical():
    emb = HashingEmbedder()
    vecs = emb.embed(["the licence plate is GE-418-TR", "plate number of the car", "the weather is nice"])
    assert vecs.shape == (3, 1024)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    assert vecs[0] @ vecs[1] > vecs[0] @ vecs[2]
    assert np.array_equal(emb.embed(["same"]), emb.embed(["same"]))  # deterministic


def test_synthesized_word_timings_span_segment():
    words = synthesize_words("the plate is GE-418-TR", 1000, 3000)
    assert words[0]["s"] == 1000 and words[-1]["e"] == 3000
    assert all(a["e"] <= b["s"] + 1 for a, b in zip(words, words[1:]))


def test_store_and_search_real_pgvector(db, seeded):
    call_id = seeded["clean_call"]
    assert db.query(Chunk).filter_by(call_id=call_id).count() > 3

    hits = search_chunks(db, call_id, "what is the registration plate of the vehicle", top_k=3)
    assert len(hits) == 3
    assert hits[0]["score"] >= hits[1]["score"] >= hits[2]["score"]
    top_text = " ".join(h["text"] for h in hits)
    assert "GE-418-TR" in top_text
    for h in hits:
        assert h["segments"], "each hit carries its segments with timestamps"
        assert all(h["start_ms"] <= s["end_ms"] and s["start_ms"] <= h["end_ms"] for s in h["segments"])

    # Search is scoped to one call.
    other = seeded["french_call"]
    assert all("GE-418-TR" not in h["text"] for h in search_chunks(db, other, "GE-418-TR plate", top_k=5))

    # top_k is clamped.
    assert len(search_chunks(db, call_id, "plate", top_k=500)) <= 20


def test_segments_between_and_verbatim_quote(db):
    call = Call(filename="t", status="transcribed")
    db.add(call)
    db.flush()
    store_transcript(db, call, [
        {"speaker": "agent", "start_ms": 0, "end_ms": 2000, "text": "What is your plate?"},
        {"speaker": "caller", "start_ms": 2500, "end_ms": 6500, "text": "The plate is 1234-A-56, no sorry, 1234-B-56."},
        {"speaker": "agent", "start_ms": 7000, "end_ms": 8000, "text": "Noted."},
    ])
    segs = segments_between(db, call.id, 2600, 2700)
    assert [s.speaker for s in segs] == ["caller"]
    assert verbatim_quote(segs, 0, 10_000) == "The plate is 1234-A-56, no sorry, 1234-B-56."
    # Word timestamps let the quote be trimmed to the cited sub-range.
    words = segs[0].words
    late = [w for w in words if w["w"].startswith("1234-B")][0]
    assert verbatim_quote(segs, late["s"], late["e"]).strip() == "1234-B-56."
    assert len(segments_between(db, call.id, 0, 10_000)) == 3
    assert isinstance(db.get(Segment, segs[0].id).words, list)
