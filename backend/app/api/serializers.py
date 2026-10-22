"""Response shaping (plain dicts; the API surface is small enough not to need Pydantic models)."""
from __future__ import annotations

from typing import Any

from app.agent.prompts import resolve_language
from app.agent.schema import FormSchema, SchemaNotFound, load_schema
from app.db.models import Call, Extraction, FieldEvidence


def call_summary(call: Call) -> dict[str, Any]:
    return {
        "id": call.id,
        "filename": call.filename,
        "duration": call.duration,
        "language": call.language,
        "status": call.status,
        "error": call.error,
        "source": call.source,
        "has_audio": bool(call.audio_path),
        "diarization_method": call.diarization_method,
        "recorded_at": call.recorded_at.isoformat() if call.recorded_at else None,
        "created_at": call.created_at.isoformat() if call.created_at else None,
        "latest_extraction_id": call.extractions[-1].id if call.extractions else None,
    }


def call_detail(call: Call) -> dict[str, Any]:
    segments = [
        {
            "id": s.id,
            "speaker": s.speaker,
            "start_ms": s.start_ms,
            "end_ms": s.end_ms,
            "text": s.text,
            "words": s.words,
        }
        for s in call.segments
    ]
    return {
        **call_summary(call),
        "transcript": "\n".join(f"[{s.start_ms / 1000:7.2f}s] {s.speaker}: {s.text}" for s in call.segments),
        "segments": segments,
        "extractions": [
            {
                "id": e.id,
                "schema": e.form_schema_name,
                "status": e.status,
                "termination_reason": e.termination_reason,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in call.extractions
        ],
    }


def _evidence(ev: FieldEvidence) -> dict[str, Any]:
    return {
        "field_path": ev.field_path,
        "value": ev.value,
        "confidence": ev.confidence,
        "start_ms": ev.start_ms,
        "end_ms": ev.end_ms,
        "quote": ev.quote,
        "source": ev.source,
        "warnings": ev.warnings or [],
        "original_value": ev.original_value,
        "corrected": ev.corrected_at is not None,
        "tool_trace": ev.tool_trace or [],
    }


def schema_fields(schema: FormSchema, lang: str) -> list[dict[str, Any]]:
    out = []
    for f in schema.fields:
        item = {
            "path": f.path,
            "label": f.label(lang),
            "type": f.type,
            "required": f.required,
            "format": f.schema.get("format"),
            "enum": f.schema.get("enum"),
            "description": f.schema.get("description"),
        }
        if f.type == "array" and isinstance(f.schema.get("items"), dict):
            item["items"] = {k: v for k, v in f.schema["items"].items() if not k.startswith("x-")}
        out.append(item)
    return out


def extraction_detail(extraction: Extraction, include_trace: bool = False) -> dict[str, Any]:
    lang = resolve_language(extraction.language)
    try:
        fields = schema_fields(load_schema(extraction.form_schema_name), lang)
    except SchemaNotFound:
        fields = []
    out = {
        "id": extraction.id,
        "call_id": extraction.call_id,
        "schema": extraction.form_schema_name,
        "status": extraction.status,
        "termination_reason": extraction.termination_reason,
        "language": extraction.language,
        "model": extraction.model,
        "iterations": extraction.iterations,
        "tool_calls": extraction.tool_calls,
        "error": extraction.error,
        "created_at": extraction.created_at.isoformat() if extraction.created_at else None,
        "completed_at": extraction.completed_at.isoformat() if extraction.completed_at else None,
        "payload": extraction.payload,
        "fields": [_evidence(e) for e in extraction.evidence],
        "missing": [
            {
                "field_path": m.field_path,
                "reason": m.reason,
                "suggested_question": m.suggested_question,
                "resolved": m.resolved,
            }
            for m in extraction.missing
        ],
        "corrections": [
            {
                "field_path": c.field_path,
                "original_value": c.original_value,
                "corrected_value": c.corrected_value,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in extraction.corrections
        ],
        "followup_script": extraction.followup_script,
        "schema_fields": fields,
    }
    if include_trace:
        out["trace"] = extraction.trace
    return out
