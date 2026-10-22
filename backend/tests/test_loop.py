"""Agent loop behaviour with a scripted model and the real tools/DB."""
import anthropic
import httpx2 as httpx
import pytest

from app.agent.loop import (
    MISSING_KEY_MESSAGE,
    AgentAPIError,
    AgentConfigError,
    call_with_backoff,
    make_client,
    run_extraction,
)
from app.config import Settings, get_settings
from app.db.models import Call, CallStatus, Extraction
from tests.fakes import ScriptedClient, response, text_block, tool_use


def _api_status_error(cls, status):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(status, request=req, headers={"retry-after": "0"})
    return cls(message=f"HTTP {status}", response=resp, body=None)


def always_searching(i, params):
    """A model that never stops calling tools (3 per turn) until it is forced to submit."""
    if params.get("tool_choice", {}).get("name") == "submit_form":
        return response(tool_use("submit_form", payload={"vehicle_plate": "GE-418-TR"}, evidence=[]))
    return response(*[tool_use("search_transcript", query=f"query {i}-{k}") for k in range(3)])


def test_tool_call_cap_forces_submission(db, seeded):
    client = ScriptedClient(always_searching)
    result = run_extraction(db, seeded["clean_call"], "fnol", client=client)
    assert result.termination_reason == "max_tool_calls"
    extraction = db.get(Extraction, result.extraction_id)
    research = [t for t in extraction.trace if t["tool"] != "submit_form"]
    assert len(research) == 40  # 13 turns x 3 + 1; the 41st and 42nd calls are refused
    assert result.tool_calls == 41  # + the forced, terminal submit_form
    assert extraction.status == "completed" and extraction.termination_reason == "max_tool_calls"
    assert extraction.payload == {"vehicle_plate": "GE-418-TR"}
    assert len(extraction.trace) == 40 + 1  # + the forced submit_form
    # The forced request pins tool_choice and includes a result for every tool_use.
    forced = client.requests[-1]
    assert forced["tool_choice"] == {"type": "tool", "name": "submit_form"}
    last_user = forced["messages"][-1]["content"]
    assert sum(1 for b in last_user if b.get("type") == "tool_result") == 3
    assert "budget exhausted" in str(last_user)
    assert db.get(Call, seeded["clean_call"]).status == CallStatus.EXTRACTED


def test_iteration_cap_forces_submission(db, seeded):
    def one_call_per_turn(i, params):
        if params.get("tool_choice"):
            return response(tool_use("submit_form", payload={}, evidence=[]))
        return response(tool_use("lookup_policy", policy_number="FNX-204518"))

    client = ScriptedClient(one_call_per_turn)
    result = run_extraction(db, seeded["clean_call"], "fnol", client=client)
    assert result.termination_reason == "max_iterations"
    assert result.iterations == 25 and result.tool_calls == 25 + 1  # + forced submit_form
    assert len(client.requests) == 26  # 25 iterations + 1 forced submission


def test_custom_caps_are_respected(db, seeded):
    settings = Settings(agent_max_tool_calls=5, agent_max_iterations=3)
    client = ScriptedClient(always_searching)
    result = run_extraction(db, seeded["clean_call"], "fnol", client=client, settings=settings)
    assert result.termination_reason == "max_tool_calls" and result.tool_calls == 5 + 1


def test_fallback_persists_validated_values_when_forced_submit_fails(db, seeded):
    def script(i, params):
        if params.get("tool_choice"):
            return response(text_block("I refuse"))
        if i == 0:
            return response(tool_use("validate_field", field_path="vehicle_plate", value="ge-418-tr"),
                            tool_use("validate_field", field_path="callback_number", value="not a phone"))
        return response(tool_use("search_transcript", query="x"))

    settings = Settings(agent_max_tool_calls=4)
    result = run_extraction(db, seeded["clean_call"], "fnol", client=ScriptedClient(script), settings=settings)
    assert result.termination_reason == "max_tool_calls+fallback"
    extraction = db.get(Extraction, result.extraction_id)
    assert extraction.payload == {"vehicle_plate": "GE-418-TR"}  # only the value that validated
    ev = extraction.evidence[0]
    assert ev.source == "forced" and ev.confidence <= 0.5


def test_no_submit_after_nudge(db, seeded):
    client = ScriptedClient(lambda i, p: response(tool_use("submit_form", payload={}, evidence=[]))
                            if p.get("tool_choice") else response(text_block("All done!")))
    result = run_extraction(db, seeded["clean_call"], "fnol", client=client)
    assert result.termination_reason == "no_submit"
    assert "submit_form" in client.requests[1]["messages"][-1]["content"]  # the nudge
    assert len(client.requests) == 3


def test_submit_in_same_turn_as_other_tools(db, seeded):
    client = ScriptedClient([
        response(tool_use("flag_missing", field_path="witnesses", reason="not asked", suggested_question="Witnesses?"),
                 tool_use("submit_form", payload={"injuries_reported": False}, evidence=[])),
    ])
    result = run_extraction(db, seeded["clean_call"], "fnol", client=client)
    assert result.termination_reason == "submitted" and result.tool_calls == 2
    extraction = db.get(Extraction, result.extraction_id)
    assert [t["tool"] for t in extraction.trace] == ["flag_missing", "submit_form"]


def test_request_shape(db, seeded):
    client = ScriptedClient([response(tool_use("submit_form", payload={}, evidence=[]))])
    run_extraction(db, seeded["french_call"], "fnol", client=client)
    req = client.requests[0]
    assert req["model"] == "claude-sonnet-4-6"
    assert {t["name"] for t in req["tools"]} >= {"search_transcript", "submit_form"}
    assert "LA PLUS TARDIVE" in req["system"]  # French call -> French system prompt
    first = req["messages"][0]["content"]
    assert "policy_number" in first and "2026-09-23" in first
    # Only text is ever sent: no audio/media blocks anywhere in the request.
    assert "base64" not in str(req) and "audio" not in str(req["messages"]).lower()


def test_retry_on_429_and_500_then_success():
    ok = response(text_block("ok"))
    client = ScriptedClient([
        _api_status_error(anthropic.RateLimitError, 429),
        _api_status_error(anthropic.InternalServerError, 500),
        ok,
    ])
    sleeps = []
    assert call_with_backoff(client, {"messages": []}, retries=5, sleep=sleeps.append, base_delay=1.0) is ok
    assert len(sleeps) == 2


def test_backoff_is_exponential_without_retry_after():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.InternalServerError(message="boom", response=httpx.Response(500, request=req), body=None)
    client = ScriptedClient([err] * 4 + [response(text_block("ok"))])
    sleeps = []
    call_with_backoff(client, {"messages": []}, retries=5, sleep=sleeps.append, base_delay=1.0)
    for n, slept in enumerate(sleeps):  # 1, 2, 4, 8 s, each with up to +25 % jitter
        assert 2**n <= slept <= 2**n * 1.25


def test_gives_up_after_retries():
    client = ScriptedClient([_api_status_error(anthropic.InternalServerError, 500)] * 3)
    with pytest.raises(AgentAPIError):
        call_with_backoff(client, {"messages": []}, retries=2, sleep=lambda s: None)


def test_client_errors_are_not_retried():
    client = ScriptedClient([_api_status_error(anthropic.BadRequestError, 400)])
    with pytest.raises(anthropic.BadRequestError):
        call_with_backoff(client, {"messages": []}, retries=5, sleep=lambda s: None)
    assert len(client.requests) == 1


def test_api_outage_persists_what_was_validated(db, seeded):
    def script(i, params):
        if i == 0:
            return response(tool_use("validate_field", field_path="policy_number", value="FNX-204518"))
        raise _api_status_error(anthropic.InternalServerError, 503)

    result = run_extraction(db, seeded["clean_call"], "fnol", client=ScriptedClient(script), sleep=lambda s: None)
    assert result.termination_reason == "api_error"
    extraction = db.get(Extraction, result.extraction_id)
    assert extraction.payload == {"policy_number": "FNX-204518"}
    assert "failed after" in extraction.error


def test_missing_api_key_is_a_clean_config_error(db, seeded):
    assert get_settings().anthropic_api_key is None
    with pytest.raises(AgentConfigError) as exc:
        make_client()
    assert "ANTHROPIC_API_KEY is not set" in str(exc.value)
    with pytest.raises(AgentConfigError):
        run_extraction(db, seeded["clean_call"], "fnol")
    assert db.query(Extraction).count() == 0  # nothing half-created
    assert MISSING_KEY_MESSAGE.startswith("ANTHROPIC_API_KEY")


def test_invalid_api_key_is_a_clean_config_error(db, seeded):
    client = ScriptedClient([_api_status_error(anthropic.AuthenticationError, 401)])
    with pytest.raises(AgentConfigError, match="rejected"):
        run_extraction(db, seeded["clean_call"], "fnol", client=client)
    extraction = db.query(Extraction).one()
    assert extraction.status == "failed"
    assert db.get(Call, seeded["clean_call"]).status == CallStatus.TRANSCRIBED
