"""Background worker: transcribes queued calls, then (optionally) runs the extraction agent.

Jobs are rows in ``calls`` with status ``queued``; claiming uses
``SELECT ... FOR UPDATE SKIP LOCKED`` so several workers can run side by side.
"""
from __future__ import annotations

import logging
import signal
import time

from sqlalchemy import text

from app.agent.loop import AgentAPIError, AgentConfigError, run_extraction
from app.config import get_settings
from app.db.models import Call, CallStatus
from app.db.session import session_scope
from app.logging_setup import configure_logging

log = logging.getLogger("worker")
_stop = False

CLAIM_SQL = text(
    """
    UPDATE calls SET status = :transcribing, error = NULL
    WHERE id = (
        SELECT id FROM calls WHERE status = :queued ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1
    )
    RETURNING id
    """
)


def claim_next() -> int | None:
    with session_scope() as s:
        row = s.execute(CLAIM_SQL, {"transcribing": CallStatus.TRANSCRIBING, "queued": CallStatus.QUEUED}).first()
        return int(row[0]) if row else None


def requeue_stale() -> None:
    """Calls left 'transcribing'/'extracting' by a crashed worker go back to a sane state."""
    with session_scope() as s:
        n = s.execute(
            text("UPDATE calls SET status = :q WHERE status = :t"), {"q": CallStatus.QUEUED, "t": CallStatus.TRANSCRIBING}
        ).rowcount
        m = s.execute(
            text("UPDATE calls SET status = :d WHERE status = :e"), {"d": CallStatus.TRANSCRIBED, "e": CallStatus.EXTRACTING}
        ).rowcount
        if n or m:
            log.info("Recovered %d stale transcription job(s) and %d stale extraction(s)", n, m)


def process(call_id: int) -> None:
    from app.asr.pipeline import transcribe_call  # heavy imports only in the worker

    started = time.monotonic()
    try:
        with session_scope() as s:
            call = s.get(Call, call_id)
            transcribe_call(s, call)
            call.status = CallStatus.TRANSCRIBED
        log.info("Call %s transcribed in %.1fs", call_id, time.monotonic() - started)
    except Exception as exc:
        log.exception("Transcription of call %s failed", call_id)
        with session_scope() as s:
            call = s.get(Call, call_id)
            call.status, call.error = CallStatus.FAILED, f"transcription failed: {exc}"
        return

    settings = get_settings()
    if not settings.auto_extract:
        return
    if not settings.anthropic_api_key:
        log.warning("Call %s: skipping auto-extraction, ANTHROPIC_API_KEY is not set", call_id)
        return
    try:
        with session_scope() as s:
            result = run_extraction(s, call_id, settings.default_schema)
        log.info("Call %s auto-extracted: extraction %s (%s)", call_id, result.extraction_id, result.termination_reason)
    except Exception as exc:
        if isinstance(exc, (AgentConfigError, AgentAPIError)):
            log.error("Call %s: auto-extraction failed: %s", call_id, exc)
        else:
            log.exception("Call %s: auto-extraction crashed", call_id)
        with session_scope() as s:
            call = s.get(Call, call_id)
            call.status, call.error = CallStatus.TRANSCRIBED, f"auto-extraction failed: {exc}"


def _handle_stop(*_: object) -> None:
    global _stop
    _stop = True
    log.info("Stopping after the current job…")


def main() -> None:
    configure_logging()
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    settings = get_settings()
    log.info("Worker started (whisper=%s, embeddings=%s)", settings.whisper_model, settings.embedding_backend)
    if not settings.hf_token:
        log.warning("HF_TOKEN not set: diarization will use the energy-based two-speaker fallback")
    requeue_stale()
    while not _stop:
        call_id = claim_next()
        if call_id is None:
            time.sleep(settings.worker_poll_seconds)
            continue
        log.info("Processing call %s", call_id)
        process(call_id)


if __name__ == "__main__":
    main()
