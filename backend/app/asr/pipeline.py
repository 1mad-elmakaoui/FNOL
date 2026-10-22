"""Audio file -> diarized, word-timestamped transcript stored in Postgres."""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.asr.align import build_segments
from app.asr.diarize import diarize, speech_spans
from app.asr.transcribe import load_audio, transcribe
from app.db.models import Call
from app.ingest import store_transcript

log = logging.getLogger(__name__)


def transcribe_call(session: Session, call: Call) -> None:
    if not call.audio_path:
        raise ValueError(f"Call {call.id} has no audio file")
    asr = transcribe(call.audio_path)
    if not asr.words:
        raise ValueError("No speech detected in the recording")

    mono = load_audio(call.audio_path)
    left, right = load_audio(call.audio_path, split_stereo=True)
    word_spans = [(w.start_ms, w.end_ms) for w in asr.words]
    spans = speech_spans(word_spans)
    turns, method = diarize(spans, mono, (left, right), word_spans)
    segments = build_segments(asr.words, turns)
    log.info("Call %s: %d speech spans, %d segments, diarization=%s", call.id, len(spans), len(segments), method)

    call.language = asr.language
    call.duration = asr.duration_s
    call.diarization_method = method
    store_transcript(
        session,
        call,
        [
            {"speaker": s.speaker, "start_ms": s.start_ms, "end_ms": s.end_ms, "text": s.text, "words": s.words_json()}
            for s in segments
        ],
    )
