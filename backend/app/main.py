"""FastAPI application."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.agent.loop import AgentAPIError, AgentConfigError
from app.agent.schema import SchemaNotFound, list_schemas, load_schema
from app.api import calls, extractions
from app.config import get_settings
from app.db.session import get_engine
from app.logging_setup import configure_logging

configure_logging()
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    if not settings.anthropic_api_key:
        log.warning("ANTHROPIC_API_KEY is not set: transcription works, extraction requests will return 503.")
    log.info("Schemas available: %s", ", ".join(list_schemas()) or "(none)")
    yield


app = FastAPI(title="FNOL voice-to-form", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)
app.include_router(calls.router)
app.include_router(extractions.router)


@app.exception_handler(AgentConfigError)
def _config_error(_: Request, exc: AgentConfigError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"error": "configuration", "detail": str(exc)})


@app.exception_handler(AgentAPIError)
def _api_error(_: Request, exc: AgentAPIError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"error": "anthropic_api", "detail": str(exc)})


@app.exception_handler(SchemaNotFound)
def _schema_not_found(_: Request, exc: SchemaNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": "schema_not_found", "detail": str(exc)})


@app.get("/health")
def health() -> dict:
    with get_engine().connect() as conn:
        conn.execute(text("SELECT 1"))
    settings = get_settings()
    return {
        "status": "ok",
        "anthropic_api_key": bool(settings.anthropic_api_key),
        "model": settings.anthropic_model,
        "embedding_backend": settings.embedding_backend,
        "whisper_model": settings.whisper_model,
        "diarization": "pyannote" if settings.hf_token else "energy-fallback",
    }


@app.get("/schemas")
def get_schemas() -> list[dict]:
    out = []
    for name in list_schemas():
        s = load_schema(name)
        out.append({"name": name, "title": s.title, "fields": len(s.fields)})
    return out


@app.get("/schemas/{name}")
def get_schema(name: str) -> dict:
    try:
        return load_schema(name).raw
    except SchemaNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
