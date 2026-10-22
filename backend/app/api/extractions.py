from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.agent.prompts import resolve_language
from app.agent.schema import load_schema
from app.agent.tools import ToolContext, build_followup_script
from app.agent.validators import ValidationContext
from app.api.serializers import extraction_detail
from app.config import get_settings
from app.db.models import Extraction, FieldCorrection, FieldEvidence, MissingField
from app.db.session import get_db
from app.privacy.redact import Redactor

router = APIRouter(prefix="/extractions", tags=["extractions"])


def _get(db: Session, extraction_id: int) -> Extraction:
    extraction = db.get(Extraction, extraction_id)
    if extraction is None:
        raise HTTPException(404, f"Extraction {extraction_id} not found")
    return extraction


@router.get("/{extraction_id}")
def get_extraction(extraction_id: int, trace: bool = Query(False), db: Session = Depends(get_db)) -> dict:
    return extraction_detail(_get(db, extraction_id), include_trace=trace)


class FieldCorrectionBody(BaseModel):
    value: Any = None


@router.patch("/{extraction_id}/fields/{field_path:path}")
def correct_field(
    extraction_id: int, field_path: str, body: FieldCorrectionBody, db: Session = Depends(get_db)
) -> dict:
    """Human correction. Keeps the agent's original value alongside the corrected one.

    ``{"value": null}`` clears a field (it is then listed as missing again if required).
    """
    extraction = _get(db, extraction_id)
    schema = load_schema(extraction.form_schema_name)
    if field_path not in schema.by_path:
        raise HTTPException(404, f"Unknown field {field_path!r} for schema {schema.name}")
    call = extraction.call
    settings = get_settings()

    new_value = body.value
    if isinstance(new_value, str) and not new_value.strip():
        new_value = None
    if new_value is not None:
        check = schema.validate_field(
            field_path,
            new_value,
            ValidationContext(reference_time=call.recorded_at or call.created_at, tz=settings.call_timezone),
        )
        if not check.valid:
            raise HTTPException(422, {"message": "Validation failed", "errors": check.errors})
        new_value = check.value

    payload = copy.deepcopy(extraction.payload or {})
    original = schema.get_value(payload, field_path)
    evidence = next((e for e in extraction.evidence if e.field_path == field_path), None)
    now = datetime.now(timezone.utc)

    if evidence is None:
        evidence = FieldEvidence(field_path=field_path, value=None, confidence=1.0, source="human", tool_trace=[], warnings=[])
        extraction.evidence.append(evidence)
    if evidence.corrected_at is None:
        evidence.original_value = evidence.value  # first correction: remember what the agent said
    evidence.value = new_value
    evidence.corrected_at = now
    evidence.confidence = 1.0 if new_value is not None else 0.0

    if new_value is None:
        schema.delete_value(payload, field_path)
    else:
        schema.set_value(payload, field_path, new_value)
    extraction.payload = payload
    extraction.corrections.append(
        FieldCorrection(field_path=field_path, original_value=original, corrected_value=new_value)
    )
    flagged = [m for m in extraction.missing if m.field_path == field_path]
    for m in flagged:
        m.resolved = new_value is not None
    if new_value is None and not flagged and field_path in schema.missing_required(payload):
        extraction.missing.append(
            MissingField(
                field_path=field_path,
                reason="Cleared by a reviewer.",
                suggested_question=schema.field(field_path).question(resolve_language(extraction.language)),
            )
        )

    ctx = ToolContext(
        session=db, call=call, extraction=extraction, schema=schema,
        redactor=Redactor(False), validation=ValidationContext(tz=settings.call_timezone),
    )
    extraction.followup_script = build_followup_script(ctx)
    db.commit()
    db.refresh(extraction)
    return extraction_detail(extraction)
