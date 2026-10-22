from __future__ import annotations

import logging
import mimetypes
import re
import shutil
from collections.abc import Iterator
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.loop import run_extraction
from app.api.serializers import call_detail, call_summary, extraction_detail
from app.config import get_settings
from app.db.models import Call, CallStatus, Extraction
from app.db.session import get_db

log = logging.getLogger(__name__)
router = APIRouter(prefix="/calls", tags=["calls"])

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".oga", ".opus", ".flac", ".webm", ".aac", ".mp4"}
_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")
_CHUNK = 64 * 1024


def _get_call(db: Session, call_id: int) -> Call:
    call = db.get(Call, call_id)
    if call is None:
        raise HTTPException(404, f"Call {call_id} not found")
    return call


@router.post("", status_code=201)
def upload_call(file: UploadFile = File(...), db: Session = Depends(get_db)) -> dict:
    settings = get_settings()
    name = Path(file.filename or "audio").name
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported audio type {ext or '(none)'}; allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    call = Call(filename=name, status=CallStatus.QUEUED, source="upload")
    db.add(call)
    db.flush()
    target_dir = settings.audio_dir / str(call.id)
    target_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    target = target_dir / safe
    limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with target.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"File larger than MAX_UPLOAD_MB={settings.max_upload_mb}")
                out.write(chunk)
        if written == 0:
            raise HTTPException(400, "Empty file")
    except HTTPException:
        db.rollback()
        shutil.rmtree(target_dir, ignore_errors=True)
        raise
    call.audio_path = str(target)
    db.commit()
    log.info("Queued call %s (%s, %d bytes) for transcription", call.id, name, written)
    return {"call_id": call.id, "status": call.status}


@router.get("")
def list_calls(db: Session = Depends(get_db)) -> list[dict]:
    calls = db.scalars(select(Call).order_by(Call.id.desc()).limit(200))
    return [call_summary(c) for c in calls]


@router.get("/{call_id}")
def get_call(call_id: int, db: Session = Depends(get_db)) -> dict:
    return call_detail(_get_call(db, call_id))


@router.delete("/{call_id}", status_code=204)
def delete_call(call_id: int, db: Session = Depends(get_db)) -> Response:
    """Right to erasure: removes the audio file, transcript, embeddings and extractions."""
    call = _get_call(db, call_id)
    if call.audio_path:
        shutil.rmtree(Path(call.audio_path).parent, ignore_errors=True)
    db.delete(call)
    db.commit()
    return Response(status_code=204)


def _iter_file(path: Path, start: int, length: int) -> Iterator[bytes]:
    with path.open("rb") as fh:
        fh.seek(start)
        remaining = length
        while remaining > 0:
            data = fh.read(min(_CHUNK, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


@router.get("/{call_id}/audio")
def get_audio(call_id: int, request: Request, db: Session = Depends(get_db)) -> Response:
    """Serve the original recording with HTTP Range support (needed for seeking in the browser)."""
    call = _get_call(db, call_id)
    if not call.audio_path or not Path(call.audio_path).is_file():
        raise HTTPException(404, "No audio for this call")
    path = Path(call.audio_path)
    size = path.stat().st_size
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-store"}

    range_header = request.headers.get("range")
    if not range_header:
        return StreamingResponse(_iter_file(path, 0, size), media_type=media_type,
                                 headers={**headers, "Content-Length": str(size)})

    m = _RANGE.match(range_header.strip().split(",")[0].strip())
    if not m or (m.group(1) == "" and m.group(2) == ""):
        return Response(status_code=416, headers={**headers, "Content-Range": f"bytes */{size}"})
    first, last = m.group(1), m.group(2)
    if first == "":  # suffix range: last N bytes
        length = min(int(last), size)
        start, end = size - length, size - 1
    else:
        start = int(first)
        end = min(int(last), size - 1) if last else size - 1
    if start >= size or start > end:
        return Response(status_code=416, headers={**headers, "Content-Range": f"bytes */{size}"})
    length = end - start + 1
    return StreamingResponse(
        _iter_file(path, start, length),
        status_code=206,
        media_type=media_type,
        headers={**headers, "Content-Range": f"bytes {start}-{end}/{size}", "Content-Length": str(length)},
    )


class ExtractRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    schema_name: str = Field(default="", alias="schema")


@router.post("/{call_id}/extract")
def extract(call_id: int, body: ExtractRequest | None = None, db: Session = Depends(get_db)) -> dict:
    call = _get_call(db, call_id)
    if call.status in (CallStatus.QUEUED, CallStatus.TRANSCRIBING):
        raise HTTPException(409, f"Call {call_id} is still being transcribed ({call.status})")
    if call.status == CallStatus.EXTRACTING:
        raise HTTPException(409, f"An extraction is already running for call {call_id}")
    if not call.segments:
        raise HTTPException(409, f"Call {call_id} has no transcript")
    schema_name = (body.schema_name if body else "") or get_settings().default_schema
    result = run_extraction(db, call_id, schema_name)
    extraction = db.get(Extraction, result.extraction_id)
    return extraction_detail(extraction)
