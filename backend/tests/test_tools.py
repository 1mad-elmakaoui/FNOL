import json

import pytest

from app.agent.schema import load_schema
from app.agent.tools import TOOL_DEFINITIONS, ToolContext
from app.agent.validators import ValidationContext
from app.db.models import Call, Extraction, MissingField
from app.privacy.redact import Redactor


def make_ctx(db, call_id, schema="fnol"):
    call = db.get(Call, call_id)
    extraction = Extraction(call_id=call_id, form_schema_name=schema, payload={}, language=call.language)
    db.add(extraction)
    db.flush()
    form = load_schema(schema)
    return ToolContext(
        session=db,
        call=call,
        extraction=extraction,
        schema=form,
        redactor=Redactor(True, allow=form.pii_allow),
        validation=ValidationContext(reference_time=call.recorded_at),
    )


def run(ctx, name, **kw):
    block = ctx.execute(name, kw, "toolu_x")
    return json.loads(block["content"]), block.get("is_error", False)


def test_tool_definitions_are_well_formed():
    names = {t["name"] for t in TOOL_DEFINITIONS}
    assert names == {"search_transcript", "get_segment", "lookup_policy", "validate_field", "flag_missing", "submit_form"}
    for t in TOOL_DEFINITIONS:
        assert t["input_schema"]["type"] == "object" and t["description"]


def test_search_transcript_hits_pgvector(db, seeded):
    ctx = make_ctx(db, seeded["clean_call"])
    out, err = run(ctx, "search_transcript", query="other driver insurer", top_k=3, field_path="third_party_insurer")
    assert not err and len(out["results"]) == 3
    assert any("MAIF" in s["text"] for r in out["results"] for s in r["segments"])
    assert ctx.trace[-1]["tool"] == "search_transcript" and ctx.trace[-1]["hit_ranges"]


def test_get_segment_returns_verbatim_slice(db, seeded):
    ctx = make_ctx(db, seeded["clean_call"])
    hits, _ = run(ctx, "search_transcript", query="registration of your vehicle plate")
    seg = next(s for r in hits["results"] for s in r["segments"] if "GE-418-TR" in s["text"])
    out, err = run(ctx, "get_segment", start_ms=seg["start_ms"], end_ms=seg["end_ms"])
    assert not err
    assert "GE-418-TR" in out["quote"]
    assert out["segments"][0]["speaker"] == "caller"
    _, err = run(ctx, "get_segment", start_ms=0, end_ms=10_000_000)
    assert err  # window too large


def test_lookup_policy_real_db(db, seeded):
    ctx = make_ctx(db, seeded["clean_call"])
    out, _ = run(ctx, "lookup_policy", policy_number="FNX-204518")
    assert out["found"] and out["policy"]["holder_name"] == "Daniel Morgan"
    out, _ = run(ctx, "lookup_policy", policy_number="fnx 204518")  # spoken/spaced variants match
    assert out["found"] and out["policy"]["plate"] == "GE-418-TR"
    out, _ = run(ctx, "lookup_policy", policy_number="FNX-000000")
    assert out == {"policy_number": "FNX-000000", "found": False, "policy": None}


def test_validate_field_tool(db, seeded):
    ctx = make_ctx(db, seeded["clean_call"])
    out, err = run(ctx, "validate_field", field_path="vehicle_plate", value="ge 418 tr")
    assert not err and out["valid"] and out["normalized_value"] == "GE-418-TR"
    assert ctx.candidates["vehicle_plate"] == "GE-418-TR"
    out, _ = run(ctx, "validate_field", field_path="incident_datetime", value="2030-01-01T10:00:00")
    assert not out["valid"] and "future" in out["errors"][0]
    out, err = run(ctx, "validate_field", field_path="nope", value="x")
    assert err and "Unknown field_path" in out["error"]


def test_flag_missing_records_gap(db, seeded):
    ctx = make_ctx(db, seeded["missing_policy"])
    out, err = run(ctx, "flag_missing", field_path="policy_number", reason="Caller does not have it",
                   suggested_question="Could you read me the policy number from your certificate?")
    assert not err and out["recorded"]
    rows = db.query(MissingField).filter_by(extraction_id=ctx.extraction.id).all()
    assert [(r.field_path, r.resolved) for r in rows] == [("policy_number", False)]
    # Flagging again updates rather than duplicates.
    run(ctx, "flag_missing", field_path="policy_number", reason="still missing", suggested_question="Policy number?")
    assert db.query(MissingField).filter_by(extraction_id=ctx.extraction.id).count() == 1


def test_invalid_tool_input_is_reported_not_raised(db, seeded):
    ctx = make_ctx(db, seeded["clean_call"])
    out, err = run(ctx, "flag_missing", field_path="policy_number")
    assert err and "required property" in out["error"]
    out, err = run(ctx, "no_such_tool")
    assert err


def test_submit_form_persists_evidence_and_followup(db, seeded):
    ctx = make_ctx(db, seeded["clean_call"])
    hits, _ = run(ctx, "search_transcript", query="policy number", field_path="policy_number")
    seg = next(s for r in hits["results"] for s in r["segments"] if "FNX-204518" in s["text"])
    run(ctx, "flag_missing", field_path="witnesses", reason="not sure", suggested_question="Any witnesses?")
    out, err = run(
        ctx,
        "submit_form",
        payload={"policy_number": "fnx-204518", "vehicle_plate": "GE-418-TR", "injuries_reported": False,
                 "bogus": 1},
        evidence=[
            {"field_path": "policy_number", "start_ms": seg["start_ms"], "end_ms": seg["end_ms"], "confidence": 0.95},
            {"field_path": "injuries_reported", "start_ms": 999_000, "end_ms": 999_500, "confidence": 0.9},
        ],
    )
    assert not err and ctx.submitted
    assert out["ignored_unknown_fields"] == ["bogus"]
    ev = {e.field_path: e for e in ctx.extraction.evidence}
    assert ev["policy_number"].value == "FNX-204518"  # normalized
    assert ev["policy_number"].confidence == 0.95
    assert "FNX-204518" in ev["policy_number"].quote
    assert any(t["tool"] == "search_transcript" for t in ev["policy_number"].tool_trace)
    assert ev["vehicle_plate"].confidence <= 0.3  # no evidence cited
    assert ev["injuries_reported"].confidence <= 0.3  # cited range has no transcript
    missing = {m.field_path for m in ctx.extraction.missing}
    assert "witnesses" in missing and "policyholder_name" in missing  # flagged + auto-flagged required
    assert "policy_number" not in missing
    script = ctx.extraction.followup_script
    assert "Follow-up call script" in script and "Any witnesses?" in script
    _, err = run(ctx, "submit_form", payload={}, evidence=[])
    assert err  # only once


def test_redaction_applies_to_tool_output(db):
    from app.ingest import store_transcript

    call = Call(filename="pii", status="transcribed", language="en")
    db.add(call)
    db.flush()
    store_transcript(db, call, [{"speaker": "caller", "start_ms": 0, "end_ms": 5000,
                                 "text": "My card is 4111 1111 1111 1111 and my email is jo@example.com, plate GE-418-TR"}])
    ctx = make_ctx(db, call.id)
    out, _ = run(ctx, "get_segment", start_ms=0, end_ms=5000)
    assert "4111" not in out["quote"] and "[REDACTED:CARD]" in out["quote"]
    assert "[REDACTED:EMAIL]" in out["quote"]
    assert "GE-418-TR" in out["quote"]  # form data is never redacted
    ctx_kyc = make_ctx(db, call.id, schema="kyc")  # KYC opts e-mail back in
    out, _ = run(ctx_kyc, "get_segment", start_ms=0, end_ms=5000)
    assert "jo@example.com" in out["quote"]


@pytest.mark.parametrize("fixture", ["clean_call", "french_call"])
def test_followup_script_localized(db, seeded, fixture):
    ctx = make_ctx(db, seeded[fixture])
    run(ctx, "submit_form", payload={}, evidence=[])
    script = ctx.extraction.followup_script
    if fixture == "french_call":
        assert script.startswith("Script d'appel de relance")
        assert "Pourriez-vous me donner votre numéro de contrat" in script
    else:
        assert script.startswith("Follow-up call script")
