"""The eval harness's matching/metrics and its regex baseline."""
import importlib.util
import json
from pathlib import Path

import pytest

from app.config import get_settings

EVAL = Path(get_settings().eval_dir)
spec = importlib.util.spec_from_file_location("run_eval", EVAL / "run_eval.py")
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)


def test_every_fixture_has_labels_for_every_field():
    labels = json.loads((EVAL / "labels.json").read_text())
    fixtures = sorted(p.stem for p in (EVAL / "fixtures").glob("*.json"))
    assert fixtures == sorted(k for k in labels if not k.startswith("_"))
    assert len(fixtures) == 6
    for fid in fixtures:
        assert set(labels[fid]) == set(run_eval.FIELD_KIND), fid
        doc = json.loads((EVAL / "fixtures" / f"{fid}.json").read_text())
        assert doc["language"] in ("en", "fr", "ar") and doc["segments"]
    assert any(json.loads((EVAL / "fixtures" / f"{f}.json").read_text())["language"] == "fr" for f in fixtures)


@pytest.mark.parametrize(
    "field,pred,gold,ok",
    [
        ("vehicle_plate", "ge 418 tr", "GE-418-TR", True),
        ("callback_number", "+33612457890", "0612457890", True),
        ("callback_number", "0612457891", "0612457890", False),
        ("incident_datetime", "2026-09-13T17:40:00+02:00", "2026-09-13T17:40:00", True),
        ("incident_datetime", "2026-09-13T17:45:00", "2026-09-13T17:40:00", False),
        ("policyholder_name", "Mr Daniel Morgan", "Daniel Morgan", True),
        ("damage_description", "rear bumper crushed and boot dented, tail light broken",
         "Rear bumper crushed, boot lid dented, right tail light broken", True),
        ("witnesses", [{"name": "Julie Martin", "phone": "06 87 45 12 36"}],
         [{"name": "Julie Martin", "phone": "0687451236"}], True),
        ("witnesses", [], [{"name": "Julie Martin", "phone": None}], False),
        ("police_report_number", "pv-2026-091344", "PV-2026-091344", True),
    ],
)
def test_matching(field, pred, gold, ok):
    assert run_eval.matches(field, pred, gold) is ok


def test_scoring_and_aggregation():
    gold = {f: None for f in run_eval.FIELD_KIND} | {"vehicle_plate": "GE-418-TR", "policy_number": "FNX-204518"}
    pred = {"vehicle_plate": "GE-418-TR", "policy_number": "FNX-000000", "third_party_name": "X"}
    out = run_eval.score(pred, gold)
    assert out["vehicle_plate"] == "tp" and out["policy_number"] == "fp+fn"
    assert out["third_party_name"] == "fp" and out["witnesses"] == "tn"
    m = run_eval.aggregate({"a": out})
    assert m["vehicle_plate"]["precision"] == 1.0
    assert m["policy_number"]["recall"] == 0.0
    assert m["_micro"]["tp"] == 1 and m["_micro"]["fp"] == 2 and m["_micro"]["fn"] == 1
    assert "| **all fields (micro)** |" in run_eval.markdown_table(m)


def test_baseline_keeps_latest_self_correction():
    doc = json.loads((EVAL / "fixtures" / "self_correction.json").read_text())
    pred = run_eval.baseline_extract(doc["segments"])
    assert pred["vehicle_plate"] == "1234-B-56"
