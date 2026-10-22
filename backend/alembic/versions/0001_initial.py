"""initial schema: calls, segments, chunks (pgvector), extractions, evidence, policies

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-24
"""
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "calls",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("audio_path", sa.String(1024)),
        sa.Column("duration", sa.Float),
        sa.Column("language", sa.String(16)),
        sa.Column("status", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("error", sa.Text),
        sa.Column("source", sa.String(32), nullable=False, server_default="upload"),
        sa.Column("diarization_method", sa.String(64)),
        sa.Column("recorded_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_calls_status", "calls", ["status"])

    op.create_table(
        "segments",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("call_id", sa.Integer, sa.ForeignKey("calls.id", ondelete="CASCADE"), nullable=False),
        sa.Column("speaker", sa.String(64), nullable=False),
        sa.Column("start_ms", sa.Integer, nullable=False),
        sa.Column("end_ms", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("words", JSONB, nullable=False, server_default="[]"),
    )
    op.create_index("ix_segments_call_id", "segments", ["call_id"])
    op.create_index("ix_segments_call_time", "segments", ["call_id", "start_ms", "end_ms"])

    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("call_id", sa.Integer, sa.ForeignKey("calls.id", ondelete="CASCADE"), nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("start_ms", sa.Integer, nullable=False),
        sa.Column("end_ms", sa.Integer, nullable=False),
        sa.Column("embedding", Vector(1024), nullable=False),
    )
    op.create_index("ix_chunks_call_id", "chunks", ["call_id"])
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
    )

    op.create_table(
        "extractions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("call_id", sa.Integer, sa.ForeignKey("calls.id", ondelete="CASCADE"), nullable=False),
        sa.Column("form_schema_name", sa.String(128), nullable=False),
        sa.Column("payload", JSONB, nullable=False, server_default="{}"),
        sa.Column("status", sa.String(32), nullable=False, server_default="running"),
        sa.Column("termination_reason", sa.String(64)),
        sa.Column("language", sa.String(16)),
        sa.Column("model", sa.String(128)),
        sa.Column("iterations", sa.Integer, nullable=False, server_default="0"),
        sa.Column("tool_calls", sa.Integer, nullable=False, server_default="0"),
        sa.Column("trace", JSONB, nullable=False, server_default="[]"),
        sa.Column("followup_script", sa.Text),
        sa.Column("error", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_extractions_call_id", "extractions", ["call_id"])

    op.create_table(
        "field_evidence",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("extraction_id", sa.Integer, sa.ForeignKey("extractions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("field_path", sa.String(256), nullable=False),
        sa.Column("value", JSONB),
        sa.Column("confidence", sa.Float, nullable=False, server_default="0"),
        sa.Column("start_ms", sa.Integer),
        sa.Column("end_ms", sa.Integer),
        sa.Column("quote", sa.Text),
        sa.Column("tool_trace", JSONB, nullable=False, server_default="[]"),
        sa.Column("source", sa.String(16), nullable=False, server_default="agent"),
        sa.Column("warnings", JSONB, nullable=False, server_default="[]"),
        sa.Column("original_value", JSONB),
        sa.Column("corrected_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_field_evidence_extraction_id", "field_evidence", ["extraction_id"])

    op.create_table(
        "missing_fields",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("extraction_id", sa.Integer, sa.ForeignKey("extractions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("field_path", sa.String(256), nullable=False),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column("suggested_question", sa.Text, nullable=False),
        sa.Column("resolved", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_missing_fields_extraction_id", "missing_fields", ["extraction_id"])

    op.create_table(
        "field_corrections",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("extraction_id", sa.Integer, sa.ForeignKey("extractions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("field_path", sa.String(256), nullable=False),
        sa.Column("original_value", JSONB),
        sa.Column("corrected_value", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_field_corrections_extraction_id", "field_corrections", ["extraction_id"])

    op.create_table(
        "policies",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("policy_number", sa.String(64), nullable=False),
        sa.Column("holder_name", sa.String(256), nullable=False),
        sa.Column("plate", sa.String(32)),
        sa.Column("vehicle", sa.String(256)),
        sa.Column("coverage", sa.String(128)),
    )
    op.create_index("ix_policies_policy_number", "policies", ["policy_number"], unique=True)


def downgrade() -> None:
    for table in (
        "policies",
        "field_corrections",
        "missing_fields",
        "field_evidence",
        "extractions",
        "chunks",
        "segments",
        "calls",
    ):
        op.drop_table(table)
