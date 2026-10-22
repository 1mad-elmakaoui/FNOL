"""SQLAlchemy ORM models. The Alembic migration in alembic/versions mirrors these."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 1024


class Base(DeclarativeBase):
    pass


class CallStatus:
    QUEUED = "queued"
    TRANSCRIBING = "transcribing"
    TRANSCRIBED = "transcribed"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    FAILED = "failed"


class Call(Base):
    __tablename__ = "calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(512))
    audio_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)  # seconds
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default=CallStatus.QUEUED, index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="upload")  # upload | fixture | eval
    diarization_method: Mapped[str | None] = mapped_column(String(64), nullable=True)
    recorded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    segments: Mapped[list[Segment]] = relationship(
        back_populates="call", cascade="all, delete-orphan", order_by="Segment.start_ms"
    )
    extractions: Mapped[list[Extraction]] = relationship(
        back_populates="call", cascade="all, delete-orphan", order_by="Extraction.id"
    )


class Segment(Base):
    __tablename__ = "segments"
    __table_args__ = (Index("ix_segments_call_time", "call_id", "start_ms", "end_ms"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    call_id: Mapped[int] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"), index=True)
    speaker: Mapped[str] = mapped_column(String(64))
    start_ms: Mapped[int] = mapped_column(Integer)
    end_ms: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    # Word-level timestamps: [{"w": "plate", "s": 1200, "e": 1480, "p": 0.93}, ...]
    words: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)

    call: Mapped[Call] = relationship(back_populates="segments")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    call_id: Mapped[int] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"), index=True)
    text: Mapped[str] = mapped_column(Text)
    start_ms: Mapped[int] = mapped_column(Integer)
    end_ms: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))


class Extraction(Base):
    __tablename__ = "extractions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    call_id: Mapped[int] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"), index=True)
    form_schema_name: Mapped[str] = mapped_column(String(128))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="running")  # running | completed | failed
    termination_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    iterations: Mapped[int] = mapped_column(Integer, default=0)
    tool_calls: Mapped[int] = mapped_column(Integer, default=0)
    trace: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    followup_script: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    call: Mapped[Call] = relationship(back_populates="extractions")
    evidence: Mapped[list[FieldEvidence]] = relationship(
        back_populates="extraction", cascade="all, delete-orphan", order_by="FieldEvidence.id"
    )
    missing: Mapped[list[MissingField]] = relationship(
        back_populates="extraction", cascade="all, delete-orphan", order_by="MissingField.id"
    )
    corrections: Mapped[list[FieldCorrection]] = relationship(
        back_populates="extraction", cascade="all, delete-orphan", order_by="FieldCorrection.id"
    )


class FieldEvidence(Base):
    __tablename__ = "field_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    extraction_id: Mapped[int] = mapped_column(ForeignKey("extractions.id", ondelete="CASCADE"), index=True)
    field_path: Mapped[str] = mapped_column(String(256))
    value: Mapped[Any] = mapped_column(JSONB, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    start_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quote: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_trace: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    source: Mapped[str] = mapped_column(String(16), default="agent")  # agent | forced | human
    warnings: Mapped[list[str]] = mapped_column(JSONB, default=list)
    original_value: Mapped[Any] = mapped_column(JSONB, nullable=True)
    corrected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    extraction: Mapped[Extraction] = relationship(back_populates="evidence")


class MissingField(Base):
    __tablename__ = "missing_fields"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    extraction_id: Mapped[int] = mapped_column(ForeignKey("extractions.id", ondelete="CASCADE"), index=True)
    field_path: Mapped[str] = mapped_column(String(256))
    reason: Mapped[str] = mapped_column(Text)
    suggested_question: Mapped[str] = mapped_column(Text)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    extraction: Mapped[Extraction] = relationship(back_populates="missing")


class FieldCorrection(Base):
    __tablename__ = "field_corrections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    extraction_id: Mapped[int] = mapped_column(ForeignKey("extractions.id", ondelete="CASCADE"), index=True)
    field_path: Mapped[str] = mapped_column(String(256))
    original_value: Mapped[Any] = mapped_column(JSONB, nullable=True)
    corrected_value: Mapped[Any] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    extraction: Mapped[Extraction] = relationship(back_populates="corrections")


class Policy(Base):
    __tablename__ = "policies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    policy_number: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    holder_name: Mapped[str] = mapped_column(String(256))
    plate: Mapped[str | None] = mapped_column(String(32), nullable=True)
    vehicle: Mapped[str | None] = mapped_column(String(256), nullable=True)
    coverage: Mapped[str | None] = mapped_column(String(128), nullable=True)
