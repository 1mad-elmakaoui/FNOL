"""Runtime configuration, read once from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/fnol"
        )
    )

    # Storage
    audio_dir: Path = field(default_factory=lambda: _path("AUDIO_DIR", _REPO_ROOT / "data" / "audio"))
    schemas_dir: Path = field(default_factory=lambda: _path("SCHEMAS_DIR", _REPO_ROOT / "schemas"))
    seed_dir: Path = field(default_factory=lambda: _path("SEED_DIR", _REPO_ROOT / "seed"))
    eval_dir: Path = field(default_factory=lambda: _path("EVAL_DIR", _REPO_ROOT / "eval"))
    max_upload_mb: int = field(default_factory=lambda: _int("MAX_UPLOAD_MB", 200))

    # ASR / diarization
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "small"))
    whisper_device: str = field(default_factory=lambda: os.getenv("WHISPER_DEVICE", "auto"))
    whisper_compute_type: str = field(default_factory=lambda: os.getenv("WHISPER_COMPUTE_TYPE", "auto"))
    whisper_language: str | None = field(default_factory=lambda: os.getenv("WHISPER_LANGUAGE") or None)
    hf_token: str | None = field(default_factory=lambda: os.getenv("HF_TOKEN") or None)
    pyannote_model: str = field(
        default_factory=lambda: os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")
    )

    # Embeddings / RAG
    embedding_backend: str = field(default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "sentence-transformers"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    embedding_dim: int = 1024
    chunk_max_words: int = field(default_factory=lambda: _int("CHUNK_MAX_WORDS", 80))
    chunk_overlap_segments: int = field(default_factory=lambda: _int("CHUNK_OVERLAP_SEGMENTS", 1))

    # Agent
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY") or None)
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    agent_max_tool_calls: int = field(default_factory=lambda: _int("AGENT_MAX_TOOL_CALLS", 40))
    agent_max_iterations: int = field(default_factory=lambda: _int("AGENT_MAX_ITERATIONS", 25))
    agent_max_tokens: int = field(default_factory=lambda: _int("AGENT_MAX_TOKENS", 8000))
    agent_api_retries: int = field(default_factory=lambda: _int("AGENT_API_RETRIES", 5))
    default_schema: str = field(default_factory=lambda: os.getenv("DEFAULT_SCHEMA", "fnol"))
    auto_extract: bool = field(default_factory=lambda: _bool("AUTO_EXTRACT", True))
    call_timezone: str = field(default_factory=lambda: os.getenv("CALL_TIMEZONE", "Europe/Paris"))

    # Privacy
    redaction_enabled: bool = field(default_factory=lambda: _bool("REDACTION_ENABLED", True))

    # Worker
    worker_poll_seconds: float = field(default_factory=lambda: float(os.getenv("WORKER_POLL_SECONDS", "2")))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
