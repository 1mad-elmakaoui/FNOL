"""Agent tools. Each one is a plain Python function over the real database.

``TOOL_DEFINITIONS`` is what is sent to the Anthropic API; ``ToolContext``
dispatches ``tool_use`` blocks to the functions, validates their input, and
records a trace entry per call (later copied into ``field_evidence.tool_trace``).
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from jsonschema import Draft202012Validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agent.corrections import superseding_correction
from app.agent.prompts import FOLLOWUP, resolve_language
from app.agent.schema import FormSchema
from app.agent.validators import ValidationContext
from app.db.models import Call, Extraction, FieldEvidence, MissingField, Policy
from app.privacy.redact import Redactor
from app.rag.search import search_chunks, segment_dict, segments_between, verbatim_quote

log = logging.getLogger(__name__)

MAX_SEGMENT_WINDOW_MS = 120_000
TRACE_OUTPUT_CHARS = 1500

# ----------------------------------------------------------------------------- definitions
_EVIDENCE_ITEM = {
    "type": "object",
    "properties": {
        "field_path": {"type": "string"},
        "start_ms": {"type": "integer", "minimum": 0},
        "end_ms": {"type": "integer", "minimum": 0},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "quote": {"type": "string", "description": "Short verbatim quote of what was said."},
    },
    "required": ["field_path", "start_ms", "end_ms", "confidence"],
}

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "search_transcript",
        "description": (
            "Semantic search (pgvector cosine similarity) over the call transcript. Returns the best-matching "
            "transcript chunks, each with its segments (speaker, start_ms, end_ms, verbatim text). Results are "
            "ordered by relevance, not time: compare start_ms to spot later corrections."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, e.g. 'licence plate of the car'."},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                "field_path": {"type": "string", "description": "Form field this search is for (used for the audit trace)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_segment",
        "description": (
            "Verbatim transcript between start_ms and end_ms (max 120 s window): every segment overlapping the "
            "range with speaker and timestamps, plus the exact quoted words. Use it to read context around a "
            "search hit (e.g. what was said right after) before settling a value."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_ms": {"type": "integer", "minimum": 0},
                "end_ms": {"type": "integer", "minimum": 0},
                "field_path": {"type": "string", "description": "Form field this read is for (used for the audit trace)."},
            },
            "required": ["start_ms", "end_ms"],
        },
    },
    {
        "name": "lookup_policy",
        "description": (
            "Look up a policy number in the insurer's policy database. Returns the policy (holder_name, plate, "
            "vehicle, coverage) or null when no policy matches. For verification only: never copy values from "
            "it into the form that the caller did not say."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"policy_number": {"type": "string"}},
            "required": ["policy_number"],
        },
    },
    {
        "name": "validate_field",
        "description": (
            "Validate one value against the form's JSON Schema rule for that field plus custom validators "
            "(ISO date-time not in the future, FR/MA plate format, phone format...). Returns valid, errors and, "
            "when the value can be normalized, normalized_value — use the normalized value."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "field_path": {"type": "string"},
                "value": {"description": "Candidate value (string, number, boolean, array or object)."},
            },
            "required": ["field_path", "value"],
        },
    },
    {
        "name": "flag_missing",
        "description": (
            "Record that a field could not be filled reliably from the call (not discussed, unclear, contradictory "
            "or failed validation), with a follow-up question for the next call. Prefer this over guessing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "field_path": {"type": "string"},
                "reason": {"type": "string"},
                "suggested_question": {"type": "string", "description": "Question to ask the caller, in the call's language."},
            },
            "required": ["field_path", "reason", "suggested_question"],
        },
    },
    {
        "name": "submit_form",
        "description": (
            "Submit the completed form and end the task. payload holds only fields with a value (dot paths map "
            "to nested objects); evidence has one item per filled field citing start_ms/end_ms of the statement "
            "the value comes from, and a confidence between 0 and 1. Required fields not in payload and not "
            "flagged are automatically flagged as missing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "payload": {"type": "object"},
                "evidence": {"type": "array", "items": _EVIDENCE_ITEM},
            },
            "required": ["payload", "evidence"],
        },
    },
]

_TOOL_VALIDATORS = {t["name"]: Draft202012Validator(t["input_schema"]) for t in TOOL_DEFINITIONS}


class ToolError(Exception):
    """An error reported back to the model as an is_error tool_result."""


# ----------------------------------------------------------------------------- context
@dataclass
class ToolContext:
    session: Session
    call: Call
    extraction: Extraction
    schema: FormSchema
    redactor: Redactor
    validation: ValidationContext
    trace: list[dict[str, Any]] = field(default_factory=list)
    candidates: dict[str, Any] = field(default_factory=dict)
    tool_calls: int = 0
    iteration: int = 0
    submitted: bool = False
    submission: dict[str, Any] | None = None

    @property
    def language(self) -> str:
        return resolve_language(self.call.language)

    # ------------------------------------------------------------------ dispatch
    def execute(self, name: str, tool_input: dict[str, Any], tool_use_id: str | None = None) -> dict[str, Any]:
        """Run one tool call. Returns an Anthropic ``tool_result`` content block."""
        self.tool_calls += 1
        started = time.perf_counter()
        is_error = False
        hit_ranges: list[list[int]] = []
        try:
            if name not in TOOL_FUNCTIONS:
                raise ToolError(f"Unknown tool {name!r}")
            errors = sorted(_TOOL_VALIDATORS[name].iter_errors(tool_input), key=str)
            if errors:
                raise ToolError("Invalid input: " + "; ".join(e.message for e in errors))
            output = TOOL_FUNCTIONS[name](self, **tool_input)
            hit_ranges = output.pop("_hit_ranges", []) if isinstance(output, dict) else []
        except ToolError as exc:
            is_error, output = True, {"error": str(exc)}
        except Exception as exc:  # unexpected: report to the model, keep the loop alive
            log.exception("Tool %s failed", name)
            self.session.rollback()
            is_error, output = True, {"error": f"{type(exc).__name__}: {exc}"}
        text = json.dumps(output, ensure_ascii=False, default=str)
        self.trace.append(
            {
                "n": self.tool_calls,
                "iteration": self.iteration,
                "tool": name,
                "input": tool_input,
                "output": text if len(text) <= TRACE_OUTPUT_CHARS else text[:TRACE_OUTPUT_CHARS] + "…",
                "is_error": is_error,
                "hit_ranges": hit_ranges,
                "ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
        block: dict[str, Any] = {"type": "tool_result", "content": text}
        if tool_use_id is not None:
            block["tool_use_id"] = tool_use_id
        if is_error:
            block["is_error"] = True
        return block

    def _field(self, path: str):
        try:
            return self.schema.field(path)
        except KeyError as exc:
            raise ToolError(str(exc.args[0])) from None


# ----------------------------------------------------------------------------- tools
def search_transcript(ctx: ToolContext, query: str, top_k: int = 5, field_path: str | None = None) -> dict[str, Any]:
    hits = search_chunks(ctx.session, ctx.call.id, query, top_k)
    ranges = []
    for h in hits:
        h["text"] = ctx.redactor(h["text"])
        for s in h["segments"]:
            s["text"] = ctx.redactor(s["text"])
            ranges.append([s["start_ms"], s["end_ms"]])
    return {"query": query, "results": hits, "_hit_ranges": ranges}


def get_segment(ctx: ToolContext, start_ms: int, end_ms: int, field_path: str | None = None) -> dict[str, Any]:
    if end_ms < start_ms:
        start_ms, end_ms = end_ms, start_ms
    if end_ms - start_ms > MAX_SEGMENT_WINDOW_MS:
        raise ToolError(f"Window too large ({end_ms - start_ms} ms); max is {MAX_SEGMENT_WINDOW_MS} ms")
    segs = segments_between(ctx.session, ctx.call.id, start_ms, end_ms)
    return {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "segments": [{**segment_dict(s), "text": ctx.redactor(s.text)} for s in segs],
        "quote": ctx.redactor(verbatim_quote(segs, start_ms, end_ms)),
        "_hit_ranges": [[s.start_ms, s.end_ms] for s in segs],
    }


def _policy_dict(p: Policy) -> dict[str, Any]:
    return {
        "policy_number": p.policy_number,
        "holder_name": p.holder_name,
        "plate": p.plate,
        "vehicle": p.vehicle,
        "coverage": p.coverage,
    }


def lookup_policy(ctx: ToolContext, policy_number: str) -> dict[str, Any]:
    wanted = re.sub(r"[\s\-_.]", "", policy_number).upper()
    if not wanted:
        raise ToolError("policy_number is empty")
    stmt = select(Policy).where(func.upper(func.regexp_replace(Policy.policy_number, r"[\s\-_.]", "", "g")) == wanted)
    policy = ctx.session.scalars(stmt).first()
    return {"policy_number": policy_number, "found": policy is not None, "policy": _policy_dict(policy) if policy else None}


def validate_field(ctx: ToolContext, field_path: str, value: Any) -> dict[str, Any]:
    ctx._field(field_path)
    result = ctx.schema.validate_field(field_path, value, ctx.validation)
    if result.valid:
        ctx.candidates[field_path] = result.value
    return {"field_path": field_path, **result.as_dict(value)}


def _upsert_missing(ctx: ToolContext, field_path: str, reason: str, question: str) -> MissingField:
    existing = next((m for m in ctx.extraction.missing if m.field_path == field_path), None)
    if existing:
        existing.reason, existing.suggested_question, existing.resolved = reason, question, False
        return existing
    row = MissingField(extraction_id=ctx.extraction.id, field_path=field_path, reason=reason, suggested_question=question)
    ctx.extraction.missing.append(row)
    return row


def flag_missing(ctx: ToolContext, field_path: str, reason: str, suggested_question: str) -> dict[str, Any]:
    ctx._field(field_path)
    _upsert_missing(ctx, field_path, reason.strip(), suggested_question.strip() or ctx.schema.field(field_path).question(ctx.language))
    ctx.candidates.pop(field_path, None)
    ctx.session.flush()
    return {"recorded": True, "field_path": field_path, "missing_count": len(ctx.extraction.missing)}


def submit_form(ctx: ToolContext, payload: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    if ctx.submitted:
        raise ToolError("The form was already submitted")
    return finalize(ctx, payload, evidence, source="agent")


TOOL_FUNCTIONS = {
    "search_transcript": search_transcript,
    "get_segment": get_segment,
    "lookup_policy": lookup_policy,
    "validate_field": validate_field,
    "flag_missing": flag_missing,
    "submit_form": submit_form,
}


# ----------------------------------------------------------------------------- persistence
def trace_for_field(trace: list[dict[str, Any]], path: str, start_ms: int | None, end_ms: int | None) -> list[dict[str, Any]]:
    """Tool calls relevant to one field: tagged with its field_path, or whose results cover its evidence."""
    out = []
    for entry in trace:
        if entry["tool"] == "submit_form":
            continue
        if entry["input"].get("field_path") == path:
            out.append(entry)
        elif start_ms is not None and end_ms is not None and any(
            s <= end_ms and e >= start_ms for s, e in entry.get("hit_ranges", [])
        ):
            out.append(entry)
    return out


def finalize(ctx: ToolContext, payload: dict[str, Any], evidence: list[dict[str, Any]], source: str) -> dict[str, Any]:
    """Validate, attach verbatim evidence, persist field_evidence / missing_fields, build follow-up script."""
    schema, lang = ctx.schema, ctx.language
    unknown = [p for p in _leaf_paths(payload) if p not in schema.by_path]
    clean = schema.clean_payload(payload)
    ev_by_path = {e["field_path"]: e for e in evidence if isinstance(e, dict) and "field_path" in e}

    ctx.extraction.evidence.clear()
    ctx.session.flush()
    filled: list[str] = []
    warnings_by_field: dict[str, list[str]] = {}
    for spec in schema.fields:
        value = schema.get_value(clean, spec.path)
        if value is None:
            continue
        filled.append(spec.path)
        ev = ev_by_path.get(spec.path, {})
        warnings: list[str] = []
        confidence = float(ev.get("confidence", 0.5 if source == "forced" else 0.3))
        confidence = min(max(confidence, 0.0), 1.0)
        start, end = ev.get("start_ms"), ev.get("end_ms")
        quote = None
        if start is None or end is None:
            warnings.append("no evidence timestamps cited")
            confidence = min(confidence, 0.3 if source != "forced" else 0.5)
        else:
            start, end = int(min(start, end)), int(max(start, end))
            segs = segments_between(ctx.session, ctx.call.id, start, end)
            if not segs:
                warnings.append(f"cited range {start}-{end} ms contains no transcript")
                confidence = min(confidence, 0.3)
            else:
                quote = verbatim_quote(segs, start, end)
                correction = superseding_correction(ctx.session, ctx.call.id, segs, end)
                if correction:
                    warnings.append(
                        f"possible later self-correction at {correction['time_ms']} ms "
                        f"(\"{correction['cue']}\"): check the value"
                    )
                    confidence = min(confidence, 0.6)
        check = schema.validate_field(spec.path, value, ctx.validation)
        if not check.valid:
            warnings += check.errors
            confidence = min(confidence, 0.4)
        if source == "forced":
            warnings.append("force-submitted after the agent hit its budget")
        warnings_by_field[spec.path] = warnings
        ctx.extraction.evidence.append(
            FieldEvidence(
                field_path=spec.path,
                value=value,
                confidence=round(confidence, 3),
                start_ms=start,
                end_ms=end,
                quote=quote if quote is not None else (ev.get("quote") or None),
                tool_trace=trace_for_field(ctx.trace, spec.path, start, end)
                + [{"tool": "submit_form", "iteration": ctx.iteration, "input": {"value": value, **ev}}],
                source=source,
                warnings=warnings,
            )
        )

    # Filled fields are no longer missing; required gaps get auto-flagged.
    for m in list(ctx.extraction.missing):
        if m.field_path in filled:
            ctx.extraction.missing.remove(m)
    flagged = {m.field_path for m in ctx.extraction.missing}
    for path in schema.missing_required(clean):
        if path not in flagged:
            _upsert_missing(ctx, path, "Required field not established in the call (auto-flagged at submission).", schema.field(path).question(lang))

    ctx.extraction.payload = clean
    ctx.extraction.followup_script = build_followup_script(ctx)
    ctx.extraction.status = "completed"
    ctx.extraction.completed_at = datetime.now(timezone.utc)
    ctx.session.flush()
    ctx.submitted = True
    ctx.submission = {
        "status": "submitted",
        "filled": filled,
        "missing": [m.field_path for m in ctx.extraction.missing],
        "warnings": {k: v for k, v in warnings_by_field.items() if v},
        "ignored_unknown_fields": unknown,
    }
    return ctx.submission


def _leaf_paths(payload: Any, prefix: str = "") -> list[str]:
    if not isinstance(payload, dict):
        return []
    out = []
    for k, v in payload.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out += _leaf_paths(v, f"{path}.")
        else:
            out.append(path)
    return out


def build_followup_script(ctx: ToolContext) -> str:
    lang = ctx.language
    t = FOLLOWUP[lang]
    order = {f.path: i for i, f in enumerate(ctx.schema.fields)}
    missing = sorted((m for m in ctx.extraction.missing if not m.resolved), key=lambda m: order.get(m.field_path, 999))
    name_field = ctx.schema.raw.get("x-display-name-field")
    name = ctx.schema.get_value(ctx.extraction.payload or {}, name_field) if name_field else None
    lines = [
        t["title"].format(form_title=ctx.schema.title),
        t["meta"].format(call_id=ctx.call.id, n=len(missing)),
        "",
    ]
    if not missing:
        lines.append(t["none"])
        return "\n".join(lines)
    lines += [t["intro"].format(name=f" {name}" if name else ""), ""]
    for i, m in enumerate(missing, 1):
        label = ctx.schema.field(m.field_path).label(lang)
        lines.append(f"{i}. {m.suggested_question}")
        lines.append(f"   [{label} — {t['note']}: {m.reason}]")
    lines += ["", t["outro"]]
    return "\n".join(lines)
