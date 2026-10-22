"""Test fixtures: a real Postgres+pgvector database, migrated with Alembic.

Set TEST_DATABASE_URL to point elsewhere (default: the `fnol_test` database on
the DATABASE_URL server). Embeddings use the deterministic hashing backend so
no model download is needed. The Anthropic API is never called: agent tests
use a scripted client (tests/fakes.py) while every tool runs against the DB.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

_base = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/fnol")
TEST_DB_URL = os.getenv("TEST_DATABASE_URL") or make_url(_base).set(database="fnol_test").render_as_string(
    hide_password=False
)

os.environ["DATABASE_URL"] = TEST_DB_URL
os.environ["EMBEDDING_BACKEND"] = "hashing"
os.environ.setdefault("SCHEMAS_DIR", str(REPO / "schemas"))
os.environ.setdefault("SEED_DIR", str(REPO / "seed"))
os.environ.setdefault("EVAL_DIR", str(REPO / "eval"))
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ["AUTO_EXTRACT"] = "false"

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()

TABLES = "field_corrections, missing_fields, field_evidence, extractions, chunks, segments, calls, policies"


def _ensure_database(url: str) -> None:
    u = make_url(url)
    admin = create_engine(u.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        exists = conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": u.database}).first()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{u.database}"'))
    admin.dispose()


@pytest.fixture(scope="session")
def engine():
    from alembic import command
    from alembic.config import Config

    _ensure_database(TEST_DB_URL)
    eng = create_engine(TEST_DB_URL)
    with eng.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "alembic"))
    cfg.set_main_option("sqlalchemy.url", TEST_DB_URL)
    command.upgrade(cfg, "head")
    yield eng
    eng.dispose()


@pytest.fixture()
def db(engine, tmp_path, monkeypatch):
    from app.db.session import get_sessionmaker

    monkeypatch.setenv("AUDIO_DIR", str(tmp_path / "audio"))
    get_settings.cache_clear()
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE"))
    session = get_sessionmaker(TEST_DB_URL)()
    yield session
    session.rollback()
    session.close()
    get_settings.cache_clear()


@pytest.fixture()
def seeded(db):
    """Policies + all six eval fixtures ingested as calls. Returns {fixture_id: call_id}."""
    from app.seed import ingest_fixture, load_fixture, load_policies

    settings = get_settings()
    load_policies(db, settings.seed_dir / "policies.csv")
    ids = {}
    for path in sorted((settings.eval_dir / "fixtures").glob("*.json")):
        call = ingest_fixture(db, load_fixture(path))
        ids[path.stem] = call.id
    db.commit()
    return ids


@pytest.fixture()
def client(db):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
