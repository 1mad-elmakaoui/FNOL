"""Self-correction: "plate is 1234-A-56, no sorry, 1234-B-56" -> the later value wins.

Covered at three levels:
1. the rule is in the system prompt, in every supported language;
2. the deterministic safety net flags evidence that cites a superseded statement;
3. an end-to-end loop run where the model cites the corrected statement.
"""
import json
from types import SimpleNamespace

import pytest

from app.agent.corrections import find_correction_cues
from app.agent.loop import run_extraction
from app.agent.prompts import SUPPORTED_LANGUAGES, build_system_prompt
from app.agent.schema import load_schema
from app.db.models import Extraction, Segment
from tests.fakes import ScriptedClient, response, tool_use
from tests.test_tools import make_ctx, run


@pytest.mark.parametrize("lang", SUPPORTED_LANGUAGES)
def test_later_statement_wins_rule_in_every_prompt(lang):
    prompt = build_system_prompt(load_schema("fnol"), lang, 40)
    marker = {"en": "The LATER statement always wins", "fr": "LA PLUS TARDIVE l'emporte toujours",
              "ar": "التصريح اللاحق هو الذي يُعتمد دائمًا"}[lang]
    assert marker in prompt
    assert "1234-A-56" in prompt and "1234-B-56" in prompt
    assert "flag_missing" in prompt and "start_ms" in prompt


def test_prompt_falls_back_to_english_for_unsupported_language():
    prompt = build_system_prompt(load_schema("fnol"), "de", 40)
    assert "The LATER statement always wins" in prompt


@pytest.mark.parametrize(
    "text,cue",
    [
        ("The plate is 1234-A-56, no sorry, 1234-B-56.", "no sorry"),
        ("around three pm... actually no, it was closer to half past four", "actually"),
        ("c'est le 06 23 45 67 98, non pardon, 89 à la fin", "non pardon"),
        ("رقم اللوحة 1234-A-56، لا عفوًا، 1234-B-56", "لا عفوا"),
    ],
)
def test_correction_cues_detected(text, cue):
    s = SimpleNamespace(speaker="caller", start_ms=0, end_ms=5000, text=text, words=[])
    assert cue in [c["cue"] for c in find_correction_cues(s)]


def _plate_words(db, call_id):
    seg = db.query(Segment).filter(Segment.call_id == call_id, Segment.text.contains("no sorry")).one()
    first = next(w for w in seg.words if w["w"].startswith("1234-A"))
    last = next(w for w in seg.words if w["w"].startswith("1234-B"))
    return seg, first, last


def test_citing_superseded_value_is_flagged(db, seeded):
    call_id = seeded["self_correction"]
    seg, first, _ = _plate_words(db, call_id)
    ctx = make_ctx(db, call_id)
    run(ctx, "submit_form", payload={"vehicle_plate": "1234-A-56"},
        evidence=[{"field_path": "vehicle_plate", "start_ms": seg.start_ms, "end_ms": first["e"], "confidence": 0.95}])
    ev = ctx.extraction.evidence[0]
    assert any("self-correction" in w for w in ev.warnings)
    assert ev.confidence <= 0.6  # amber at best: a reviewer must look


def test_citing_corrected_value_is_clean(db, seeded):
    call_id = seeded["self_correction"]
    seg, _, last = _plate_words(db, call_id)
    ctx = make_ctx(db, call_id)
    run(ctx, "submit_form", payload={"vehicle_plate": "1234-B-56"},
        evidence=[{"field_path": "vehicle_plate", "start_ms": seg.start_ms, "end_ms": seg.end_ms, "confidence": 0.95}])
    ev = ctx.extraction.evidence[0]
    assert ev.warnings == [] and ev.confidence == 0.95
    assert "1234-B-56" in ev.quote


def test_loop_end_to_end_keeps_corrected_value(db, seeded):
    call_id = seeded["self_correction"]

    def script(i, params):
        if i == 0:
            return response(tool_use("search_transcript", query="vehicle registration plate", field_path="vehicle_plate"))
        if i == 1:
            # The model reads the search result that the real tool returned from Postgres.
            result = json.loads(params["messages"][-1]["content"][0]["content"])
            seg = next(s for r in result["results"] for s in r["segments"] if "no sorry" in s["text"])
            script.seg = seg
            return response(tool_use("validate_field", field_path="vehicle_plate", value="1234 b 56"))
        return response(tool_use(
            "submit_form",
            payload={"vehicle_plate": "1234-B-56"},
            evidence=[{"field_path": "vehicle_plate", "start_ms": script.seg["start_ms"],
                       "end_ms": script.seg["end_ms"], "confidence": 0.93}],
        ))

    result = run_extraction(db, call_id, "fnol", client=ScriptedClient(script))
    assert result.termination_reason == "submitted"
    extraction = db.get(Extraction, result.extraction_id)
    assert extraction.payload["vehicle_plate"] == "1234-B-56"
    ev = next(e for e in extraction.evidence if e.field_path == "vehicle_plate")
    assert "no sorry, 1234-B-56" in ev.quote and ev.warnings == []
    assert [t["tool"] for t in ev.tool_trace] == ["search_transcript", "validate_field", "submit_form"]
