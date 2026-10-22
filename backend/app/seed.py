"""`make seed`: load the policy table and a sample transcript fixture.

    python -m app.seed [--fixture clean_call] [--all-fixtures]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Call, CallStatus, Policy
from app.db.session import session_scope
from app.ingest import store_transcript
from app.logging_setup import configure_logging

log = logging.getLogger("seed")


def load_policies(session: Session, csv_path: Path) -> int:
    with csv_path.open(encoding="utf-8") as fh:
        rows = [{k: (v.strip() or None) for k, v in r.items()} for r in csv.DictReader(fh)]
    if not rows:
        return 0
    stmt = insert(Policy).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=[Policy.policy_number],
        set_={c: stmt.excluded[c] for c in ("holder_name", "plate", "vehicle", "coverage")},
    )
    session.execute(stmt)
    return len(rows)


def load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ingest_fixture(session: Session, fixture: dict[str, Any], source: str = "fixture") -> Call:
    """Create (or refresh) a call from a synthetic transcript, bypassing ASR."""
    filename = f"{source}:{fixture['id']}"
    call = session.scalars(select(Call).where(Call.filename == filename)).first()
    if call is None:
        call = Call(filename=filename, source=source)
        session.add(call)
    call.language = fixture.get("language")
    call.duration = fixture.get("duration_s")
    call.recorded_at = datetime.fromisoformat(fixture["recorded_at"]) if fixture.get("recorded_at") else None
    call.diarization_method = "fixture"
    call.status = CallStatus.TRANSCRIBED
    call.error = None
    session.flush()
    store_transcript(session, call, fixture["segments"])
    return call


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="clean_call", help="eval fixture to load as a sample call")
    parser.add_argument("--all-fixtures", action="store_true", help="load every eval fixture")
    args = parser.parse_args()
    settings = get_settings()

    with session_scope() as s:
        n = load_policies(s, settings.seed_dir / "policies.csv")
        log.info("Upserted %d policies", n)

    fixtures_dir = settings.eval_dir / "fixtures"
    paths = sorted(fixtures_dir.glob("*.json")) if args.all_fixtures else [fixtures_dir / f"{args.fixture}.json"]
    for path in paths:
        with session_scope() as s:
            call = ingest_fixture(s, load_fixture(path))
            log.info("Loaded sample call %s from %s (%d segments)", call.id, path.name, len(call.segments))


if __name__ == "__main__":
    main()
