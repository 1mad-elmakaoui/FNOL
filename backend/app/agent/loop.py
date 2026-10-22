"""The extraction agent: a manual tool-use loop over the Anthropic Messages API.

A manual loop (rather than the SDK tool runner) because we need hard caps with
a forced final submission, per-call tracing, and custom retry behaviour.

Termination reasons (stored in ``extractions.termination_reason``):

* ``submitted``       — the model called submit_form.
* ``max_tool_calls``  — tool-call cap hit; the model was forced to submit.
* ``max_iterations``  — iteration cap hit; the model was forced to submit.
* ``no_submit``       — the model stopped without submitting even after a nudge.
* ``*+fallback``      — the forced submission failed too, so whatever had been
                        validated (validate_field successes) was persisted.
* ``api_error``       — the API kept failing after retries; validated values persisted.
"""
from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

import anthropic
from sqlalchemy.orm import Session

from app.agent.prompts import FORCE_SUBMIT, NUDGE_SUBMIT, build_initial_message, build_system_prompt, resolve_language
from app.agent.schema import load_schema
from app.agent.tools import TOOL_DEFINITIONS, ToolContext, finalize
from app.agent.validators import ValidationContext
from app.config import Settings, get_settings
from app.db.models import Call, CallStatus, Extraction
from app.privacy.redact import Redactor

log = logging.getLogger(__name__)


class AgentConfigError(RuntimeError):
    """Configuration problem the user must fix (e.g. missing API key). Rendered as a clean HTTP error."""


class AgentAPIError(RuntimeError):
    """The Anthropic API kept failing after retries."""


MISSING_KEY_MESSAGE = (
    "ANTHROPIC_API_KEY is not set. Add it to your .env file (see .env.example) and restart the "
    "stack with `docker compose up -d`. Audio is never sent to the API — only transcript text."
)


def make_client(settings: Settings | None = None) -> anthropic.Anthropic:
    settings = settings or get_settings()
    if not settings.anthropic_api_key:
        raise AgentConfigError(MISSING_KEY_MESSAGE)
    # Retries are handled by call_with_backoff so they are visible in our logs.
    return anthropic.Anthropic(api_key=settings.anthropic_api_key, max_retries=0)


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        value = headers.get("retry-after")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def call_with_backoff(
    client: Any,
    params: dict[str, Any],
    retries: int,
    sleep: Callable[[float], None] = time.sleep,
    base_delay: float = 1.0,
) -> Any:
    """messages.create with exponential backoff on 429 / 5xx / connection errors."""
    for attempt in range(retries + 1):
        try:
            return client.messages.create(**params)
        except anthropic.AuthenticationError as exc:
            raise AgentConfigError(
                "ANTHROPIC_API_KEY was rejected by the API (401). Check the key in your .env file."
            ) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code != 429 and exc.status_code < 500:
                raise
            error: Exception = exc
        except anthropic.APIConnectionError as exc:
            error = exc
        if attempt == retries:
            raise AgentAPIError(f"Anthropic API failed after {retries + 1} attempts: {error}") from error
        delay = _retry_after(error) or base_delay * (2**attempt)
        delay = min(delay, 60.0) * (1 + random.uniform(0, 0.25))
        log.warning("Anthropic API error (%s); retry %d/%d in %.1fs", error, attempt + 1, retries, delay)
        sleep(delay)
    raise AssertionError("unreachable")


@dataclass
class LoopResult:
    extraction_id: int
    termination_reason: str
    iterations: int
    tool_calls: int


def _call_info(call: Call, tz: str) -> dict[str, Any]:
    recorded = call.recorded_at or call.created_at
    local = recorded.astimezone(ZoneInfo(tz)) if recorded and recorded.tzinfo else recorded
    speakers = sorted({s.speaker for s in call.segments})
    return {
        "id": call.id,
        "recorded_at": local.strftime("%Y-%m-%dT%H:%M (%A)") if local else None,
        "timezone": tz,
        "duration": call.duration,
        "language": call.language,
        "speakers": speakers,
    }


def _block_attr(block: Any, name: str, default: Any = None) -> Any:
    return block.get(name, default) if isinstance(block, dict) else getattr(block, name, default)


def run_extraction(
    session: Session,
    call_id: int,
    schema_name: str,
    client: Any | None = None,
    settings: Settings | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> LoopResult:
    settings = settings or get_settings()
    schema = load_schema(schema_name)
    call = session.get(Call, call_id)
    if call is None:
        raise LookupError(f"Call {call_id} not found")
    if not call.segments:
        raise ValueError(f"Call {call_id} has no transcript yet (status: {call.status})")
    client = client or make_client(settings)

    extraction = Extraction(
        call_id=call.id,
        form_schema_name=schema.name,
        language=call.language,
        model=settings.anthropic_model,
        status="running",
        payload={},
    )
    session.add(extraction)
    call.status = CallStatus.EXTRACTING
    session.commit()

    try:
        return _run(session, call, extraction, schema, client, settings, sleep)
    except Exception as exc:
        session.rollback()
        extraction.status = "failed"
        extraction.error = str(exc) if isinstance(exc, AgentConfigError) else f"{type(exc).__name__}: {exc}"
        call.status = CallStatus.TRANSCRIBED
        session.commit()
        raise


def _run(
    session: Session,
    call: Call,
    extraction: Extraction,
    schema: Any,
    client: Any,
    settings: Settings,
    sleep: Callable[[float], None],
) -> LoopResult:
    lang = resolve_language(call.language)
    ctx = ToolContext(
        session=session,
        call=call,
        extraction=extraction,
        schema=schema,
        redactor=Redactor(settings.redaction_enabled, allow=schema.pii_allow),
        validation=ValidationContext(reference_time=call.recorded_at or call.created_at, tz=settings.call_timezone),
    )
    system = build_system_prompt(schema, call.language, settings.agent_max_tool_calls)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": build_initial_message(schema, _call_info(call, settings.call_timezone))}
    ]
    base_params: dict[str, Any] = {
        "model": settings.anthropic_model,
        "max_tokens": settings.agent_max_tokens,
        "system": system,
        "tools": TOOL_DEFINITIONS,
        # Automatic prompt caching: the growing transcript of tool calls is re-read every turn.
        "cache_control": {"type": "ephemeral"},
    }

    def api(msgs: list[dict[str, Any]], **extra: Any) -> Any:
        return call_with_backoff(client, {**base_params, **extra, "messages": msgs}, settings.agent_api_retries, sleep)

    reason: str | None = None
    nudged = False
    try:
        while True:
            if ctx.iteration >= settings.agent_max_iterations:
                reason = "max_iterations"
                break
            ctx.iteration += 1
            response = api(messages)
            content = list(response.content)
            messages.append({"role": "assistant", "content": content})
            tool_uses = [b for b in content if _block_attr(b, "type") == "tool_use"]

            if not tool_uses:
                if nudged:
                    reason = "no_submit"
                    break
                nudged = True
                messages.append({"role": "user", "content": NUDGE_SUBMIT[lang]})
                continue

            results: list[dict[str, Any]] = []
            # Run submit_form last so the other calls of the same turn are traced first.
            ordered = sorted(tool_uses, key=lambda b: _block_attr(b, "name") == "submit_form")
            budget_hit = False
            for block in ordered:
                name, tid, tinput = _block_attr(block, "name"), _block_attr(block, "id"), _block_attr(block, "input") or {}
                if name == "submit_form":
                    result = ctx.execute(name, tinput, tid)  # always allowed: it ends the loop
                    if ctx.submitted:
                        break
                    results.append(result)
                    continue
                if ctx.tool_calls >= settings.agent_max_tool_calls:
                    budget_hit = True
                    results.append(
                        {"type": "tool_result", "tool_use_id": tid, "is_error": True,
                         "content": "Not executed: tool-call budget exhausted."}
                    )
                    continue
                results.append(ctx.execute(name, tinput, tid))
            session.commit()

            if ctx.submitted:
                reason = "submitted"
                break
            messages.append({"role": "user", "content": results})
            if budget_hit or ctx.tool_calls >= settings.agent_max_tool_calls:
                reason = "max_tool_calls"
                break
    except AgentAPIError as exc:
        log.error("Extraction %s: %s", extraction.id, exc)
        extraction.error = str(exc)
        reason = "api_error"

    if not ctx.submitted and reason != "api_error":
        reason = _force_submit(ctx, messages, reason or "no_submit", lang, api)
    if not ctx.submitted:
        # Last resort: persist the values that passed validate_field during the run.
        log.warning("Extraction %s: persisting %d validated candidate(s)", extraction.id, len(ctx.candidates))
        payload: dict[str, Any] = {}
        for path, value in ctx.candidates.items():
            schema.set_value(payload, path, value)
        finalize(ctx, payload, [], source="forced")

    extraction.termination_reason = reason
    extraction.iterations = ctx.iteration
    extraction.tool_calls = ctx.tool_calls
    extraction.trace = ctx.trace
    call.status = CallStatus.EXTRACTED
    session.commit()
    log.info(
        "Extraction %s (call %s, schema %s) finished: reason=%s iterations=%d tool_calls=%d filled=%d missing=%d",
        extraction.id, call.id, schema.name, reason, ctx.iteration, ctx.tool_calls,
        len(extraction.evidence), len(extraction.missing),
    )
    return LoopResult(extraction.id, reason, ctx.iteration, ctx.tool_calls)


def _force_submit(ctx: ToolContext, messages: list[dict[str, Any]], reason: str, lang: str, api: Callable[..., Any]) -> str:
    """One extra API call with tool_choice pinned to submit_form."""
    log.warning("Extraction %s: %s — forcing submit_form", ctx.extraction.id, reason)
    note = {"type": "text", "text": FORCE_SUBMIT[lang].format(reason=reason)}
    last = messages[-1]
    if last["role"] == "user" and isinstance(last["content"], list):
        last["content"] = [*last["content"], note]
    elif last["role"] == "user":
        last["content"] = [{"type": "text", "text": last["content"]}, note]
    else:
        messages.append({"role": "user", "content": [note]})
    try:
        response = api(messages, tool_choice={"type": "tool", "name": "submit_form"})
    except AgentAPIError as exc:
        ctx.extraction.error = str(exc)
        return f"{reason}+fallback"
    for block in response.content:
        if _block_attr(block, "type") == "tool_use" and _block_attr(block, "name") == "submit_form":
            ctx.execute("submit_form", _block_attr(block, "input") or {}, _block_attr(block, "id"))
            break
    return reason if ctx.submitted else f"{reason}+fallback"
