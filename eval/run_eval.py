#!/usr/bin/env python
"""Extraction eval over synthetic transcripts (no audio, ASR bypassed).

Each fixture in eval/fixtures/ is ingested as a call (segments -> chunks ->
embeddings in Postgres), then extracted with:

* ``--agent claude``   — the real agent loop (needs ANTHROPIC_API_KEY);
* ``--agent baseline`` — a deterministic keyword/regex extractor, a floor to
  compare the agent against (no API key needed);
* ``--agent auto``     — claude when a key is set, otherwise baseline (default).

Predictions are compared with eval/labels.json per field:
TP = predicted and correct, FP = predicted but wrong (or should be empty),
FN = gold value exists but was missed or wrong. Exact-match counts both-empty
as correct. Free-text fields use token-F1 >= 0.5; names use token containment.

    python eval/run_eval.py [--agent auto|claude|baseline] [--fixtures a,b] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
BACKEND = Path(os.getenv("BACKEND_DIR", REPO / "backend"))
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("SCHEMAS_DIR", str(REPO / "schemas"))
os.environ.setdefault("SEED_DIR", str(REPO / "seed"))
os.environ.setdefault("EVAL_DIR", str(EVAL_DIR))

SCHEMA = "fnol"

# ------------------------------------------------------------------ normalization / matching
FIELD_KIND = {
    "policy_number": "exact",
    "policyholder_name": "name",
    "incident_datetime": "datetime",
    "incident_location": "fuzzy",
    "vehicle_plate": "plate",
    "damage_description": "fuzzy",
    "injuries_reported": "exact",
    "injury_details": "fuzzy",
    "third_party_involved": "exact",
    "third_party_name": "name",
    "third_party_plate": "plate",
    "third_party_insurer": "name",
    "police_report_filed": "exact",
    "police_report_number": "code",
    "witnesses": "people",
    "callback_number": "phone",
}
_STOP = set(
    "a an and the of in on at to is was his her my its it for with de du des la le les et au aux un une "
    "sur à en dans".split()
)


def fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", fold(text)) if t not in _STOP]


def token_f1(pred: str, gold: str) -> float:
    p, g = tokens(pred), tokens(gold)
    if not p or not g:
        return 0.0
    common = sum(min(p.count(t), g.count(t)) for t in set(g))
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def phone_key(value: str) -> str:
    digits = re.sub(r"\D", "", str(value))
    for cc in ("33", "212"):
        if str(value).strip().startswith(("+" + cc, "00" + cc)):
            digits = "0" + digits[len(cc) + (2 if str(value).strip().startswith("00") else 0):]
    return digits


def name_match(pred: str, gold: str) -> bool:
    g, p = set(tokens(gold)), set(tokens(pred))
    return bool(g) and g <= p


def dt_key(value: str) -> str | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).strftime("%Y-%m-%dT%H:%M")
    except ValueError:
        return None


def is_empty(value: Any) -> bool:
    return value is None or value == ""


def matches(field: str, pred: Any, gold: Any) -> bool:
    kind = FIELD_KIND.get(field, "exact")
    if kind == "exact":
        return pred == gold or (isinstance(pred, str) and isinstance(gold, str) and pred.strip().upper() == gold.upper())
    if kind == "plate":
        return re.sub(r"[\W_]", "", str(pred)).upper() == re.sub(r"[\W_]", "", str(gold)).upper()
    if kind == "code":
        return re.sub(r"\s", "", str(pred)).upper() == re.sub(r"\s", "", str(gold)).upper()
    if kind == "phone":
        return phone_key(pred) == phone_key(gold)
    if kind == "datetime":
        return dt_key(pred) is not None and dt_key(pred) == dt_key(gold)
    if kind == "name":
        return name_match(str(pred), str(gold))
    if kind == "fuzzy":
        return token_f1(str(pred), str(gold)) >= 0.5
    if kind == "people":
        if not isinstance(pred, list) or len(pred) != len(gold):
            return False
        remaining = list(pred)
        for g in gold:
            hit = next(
                (p for p in remaining if isinstance(p, dict) and name_match(str(p.get("name", "")), g["name"])
                 and (g.get("phone") is None or phone_key(p.get("phone") or "") == phone_key(g["phone"]))),
                None,
            )
            if hit is None:
                return False
            remaining.remove(hit)
        return True
    raise ValueError(kind)


def score(pred: dict[str, Any], gold: dict[str, Any]) -> dict[str, str]:
    """Per-field outcome: tp | fp | fn | fp+fn | tn."""
    out = {}
    for field in FIELD_KIND:
        p, g = pred.get(field), gold.get(field)
        if is_empty(g) and is_empty(p):
            out[field] = "tn"
        elif is_empty(g):
            out[field] = "fp"
        elif is_empty(p):
            out[field] = "fn"
        else:
            out[field] = "tp" if matches(field, p, g) else "fp+fn"
    return out


def aggregate(outcomes: dict[str, dict[str, str]]) -> dict[str, dict[str, float | int | None]]:
    per_field: dict[str, dict[str, Any]] = {}
    for field in FIELD_KIND:
        c = defaultdict(int)
        for fixture_outcomes in outcomes.values():
            o = fixture_outcomes[field]
            c["n"] += 1
            c["gold"] += o in ("tp", "fn", "fp+fn")
            c["tp"] += o == "tp"
            c["fp"] += o in ("fp", "fp+fn")
            c["fn"] += o in ("fn", "fp+fn")
            c["em"] += o in ("tp", "tn")
        per_field[field] = {
            **c,
            "precision": c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None,
            "recall": c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None,
            "exact_match": c["em"] / c["n"] if c["n"] else None,
        }
    tot = defaultdict(int)
    for f in per_field.values():
        for k in ("n", "tp", "fp", "fn", "em", "gold"):
            tot[k] += f[k]
    per_field["_micro"] = {
        **tot,
        "precision": tot["tp"] / (tot["tp"] + tot["fp"]) if tot["tp"] + tot["fp"] else None,
        "recall": tot["tp"] / (tot["tp"] + tot["fn"]) if tot["tp"] + tot["fn"] else None,
        "exact_match": tot["em"] / tot["n"] if tot["n"] else None,
    }
    return per_field


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def markdown_table(metrics: dict[str, dict[str, Any]]) -> str:
    lines = ["| field | precision | recall | exact match | gold values |", "|---|---:|---:|---:|---:|"]
    for field in FIELD_KIND:
        m = metrics[field]
        lines.append(f"| {field} | {pct(m['precision'])} | {pct(m['recall'])} | {pct(m['exact_match'])} | {m['gold']} |")
    m = metrics["_micro"]
    lines.append(f"| **all fields (micro)** | **{pct(m['precision'])}** | **{pct(m['recall'])}** | "
                 f"**{pct(m['exact_match'])}** | {m['gold']} |")
    return "\n".join(lines)


# ------------------------------------------------------------------ baseline extractor
_PLATE = re.compile(r"\b(?:[A-Z]{2}-\d{3}-[A-Z]{2}|\d{1,5}-[A-Z]-\d{1,2})\b")
_POLICY = re.compile(r"\b[A-Z]{3}-\d{6}\b")
_PHONE = re.compile(r"(\+?\d[\d .]{7,}\d)")
_CODE = re.compile(r"\b(?=[A-Z0-9/-]*\d)(?=[A-Z0-9/-]*[/-])[A-Z0-9]+(?:[/-][A-Z0-9]+){1,3}\b")
_NAME = re.compile(r"\b([A-ZÉ][\w'’-]+(?:\s+(?:El\s+|el\s+|de\s+|Le\s+)?[A-ZÉ][\w'’-]+)+)")
_INSURERS = ["Wafa Assurance", "Groupama", "MAIF", "MACIF", "AXA", "Allianz", "MAAF", "Generali", "RMA", "Saham"]
_NEG = re.compile(r"^\W*(no|non|nobody|nothing|personne|aucun|not)\b", re.I)

QUESTIONS = {
    "policy_number": ["policy", "contrat"],
    "policyholder_name": ["your name", "full name", "policyholder", "à quel nom", "c'est à quel nom", "can i start with your name"],
    "incident_location": ["where", "où"],
    "vehicle_plate": ["registration", "immatriculation", "your car", "votre véhicule"],
    "damage_description": ["damage", "dégâts", "dommages", "what's damaged"],
    "injuries_reported": ["injur", "hurt", "blessé"],
    "third_party_involved": ["another vehicle", "other vehicle", "autre véhicule", "vehicle involved"],
    "third_party_details": ["other driver", "driver, do you", "details"],
    "police_report_filed": ["police"],
    "callback_number": ["number to", "call you back", "reach you", "rappeler", "number should", "number can"],
}


def _qa_pairs(segments: list[dict]) -> list[tuple[str, str]]:
    pairs, question = [], None
    for s in segments:
        if s["speaker"] == "agent":
            question = s["text"]
        elif question is not None:
            if pairs and pairs[-1][0] is question:
                pairs[-1] = (question, pairs[-1][1] + " " + s["text"])
            else:
                pairs.append((question, s["text"]))
    return pairs


def baseline_extract(segments: list[dict]) -> dict[str, Any]:
    """Keyword question/answer matching + regexes; the latest matching answer wins."""
    out: dict[str, Any] = {}
    for q, a in _qa_pairs(segments):
        ql = q.lower()
        asked = {f for f, kws in QUESTIONS.items() if any(k in ql for k in kws)}
        if "policy_number" in asked and (m := _POLICY.findall(a)):
            out["policy_number"] = m[-1]
        if "policyholder_name" in asked and (m := _NAME.findall(a)):
            out["policyholder_name"] = m[-1]
        if "incident_location" in asked:
            out["incident_location"] = a
        if "vehicle_plate" in asked and (m := _PLATE.findall(a)):
            out["vehicle_plate"] = m[-1]
        if "damage_description" in asked:
            out["damage_description"] = a
        if "injuries_reported" in asked:
            out["injuries_reported"] = not _NEG.search(a)
            if out["injuries_reported"]:
                out["injury_details"] = a
        if "third_party_involved" in asked:
            out["third_party_involved"] = not _NEG.search(a)
        if asked & {"third_party_involved", "third_party_details"}:
            if m := _PLATE.findall(a):
                out["third_party_plate"] = m[-1]
            if m := re.findall(r"(?:driver is|name is|monsieur|madame|Mr|Mrs)\s+(" + _NAME.pattern[3:-1] + ")", a):
                out["third_party_name"] = m[-1]
            for ins in _INSURERS:
                if ins.lower() in a.lower():
                    out["third_party_insurer"] = ins
        if "police_report_filed" in asked:
            out["police_report_filed"] = not _NEG.search(a)
            codes = [c for c in _CODE.findall(a) if not _PLATE.fullmatch(c) and not _POLICY.fullmatch(c)]
            if codes:
                out["police_report_number"] = codes[-1]
        if "callback_number" in asked and (m := _PHONE.findall(a)):
            from app.agent.validators import normalize_phone

            out["callback_number"] = normalize_phone(m[-1])
    return out


# ------------------------------------------------------------------ runner
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=["auto", "claude", "baseline"], default="auto")
    parser.add_argument("--fixtures", help="comma-separated fixture ids (default: all)")
    parser.add_argument("--json", type=Path, help="write detailed results to this file")
    args = parser.parse_args()

    from app.config import get_settings
    from app.db.session import session_scope
    from app.logging_setup import configure_logging

    configure_logging()
    settings = get_settings()
    agent = args.agent
    if agent == "auto":
        agent = "claude" if settings.anthropic_api_key else "baseline"
        if agent == "baseline":
            print("!! ANTHROPIC_API_KEY is not set: running the regex BASELINE, not the Claude agent.\n"
                  "!! Set the key in .env and re-run `make eval` to evaluate the agent.\n", file=sys.stderr)
    elif agent == "claude" and not settings.anthropic_api_key:
        print("error: --agent claude requires ANTHROPIC_API_KEY (see .env.example)", file=sys.stderr)
        return 2

    from alembic import command
    from alembic.config import Config

    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "alembic"))
    command.upgrade(cfg, "head")

    from app.agent.loop import run_extraction
    from app.db.models import Extraction
    from app.seed import ingest_fixture, load_policies

    labels = json.loads((EVAL_DIR / "labels.json").read_text(encoding="utf-8"))
    wanted = set(args.fixtures.split(",")) if args.fixtures else None
    paths = [p for p in sorted((EVAL_DIR / "fixtures").glob("*.json")) if wanted is None or p.stem in wanted]
    if not paths:
        print("no fixtures selected", file=sys.stderr)
        return 2

    with session_scope() as s:
        load_policies(s, settings.seed_dir / "policies.csv")

    outcomes: dict[str, dict[str, str]] = {}
    details: dict[str, Any] = {}
    for path in paths:
        fixture = json.loads(path.read_text(encoding="utf-8"))
        gold = labels[fixture["id"]]
        started = time.monotonic()
        with session_scope() as s:
            call = ingest_fixture(s, fixture, source="eval")
            call_id = call.id
        run_info: dict[str, Any] = {}
        if agent == "claude":
            with session_scope() as s:
                result = run_extraction(s, call_id, SCHEMA)
                extraction = s.get(Extraction, result.extraction_id)
                pred = dict(extraction.payload)
                run_info = {
                    "extraction_id": result.extraction_id,
                    "termination": result.termination_reason,
                    "iterations": result.iterations,
                    "tool_calls": result.tool_calls,
                    "missing": [m.field_path for m in extraction.missing],
                }
        else:
            pred = baseline_extract(fixture["segments"])
            run_info = {"termination": "baseline", "iterations": 0, "tool_calls": 0}
        outcomes[fixture["id"]] = score(pred, gold)
        details[fixture["id"]] = {"prediction": pred, "outcomes": outcomes[fixture["id"]], **run_info,
                                  "seconds": round(time.monotonic() - started, 1)}

    metrics = aggregate(outcomes)
    model = settings.anthropic_model if agent == "claude" else "regex baseline"
    print(f"\n### FNOL extraction eval — {model} — {len(paths)} fixtures\n")
    print(markdown_table(metrics))
    print("\n| fixture | exact-match fields | termination | iterations | tool calls | time |")
    print("|---|---:|---|---:|---:|---:|")
    for fid, d in details.items():
        em = sum(o in ("tp", "tn") for o in d["outcomes"].values())
        print(f"| {fid} | {em}/{len(FIELD_KIND)} | {d['termination']} | {d['iterations']} | {d['tool_calls']} | {d['seconds']}s |")
    wrong = {fid: [f for f, o in d["outcomes"].items() if o not in ("tp", "tn")] for fid, d in details.items()}
    print("\nErrors by fixture:")
    for fid, fields in wrong.items():
        print(f"- {fid}: {', '.join(fields) or 'none'}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"agent": agent, "model": model, "metrics": metrics, "fixtures": details},
                                        ensure_ascii=False, indent=2, default=str))
        print(f"\nDetailed results written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
