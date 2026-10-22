import { useState } from "react";
import type { Extraction, FieldEvidence, SchemaField } from "../api/client";
import { confidenceLevel, formatMs, formatValue } from "../format";
import FieldEditor from "./FieldEditor";

interface Props {
  extraction: Extraction;
  canPlay: boolean;
  activeField: string | null;
  onSeek: (field: FieldEvidence) => void;
  onSave: (fieldPath: string, value: unknown) => Promise<void>;
}

function ConfidenceBadge({ ev }: { ev: FieldEvidence }) {
  if (ev.corrected) return <span className="badge badge--human" title="Corrected by a reviewer">edited</span>;
  const level = confidenceLevel(ev.confidence);
  return (
    <span className={`badge badge--${level}`} title={`Confidence ${ev.confidence.toFixed(2)}`}>
      {Math.round(ev.confidence * 100)}%
    </span>
  );
}

function Trace({ ev }: { ev: FieldEvidence }) {
  if (!ev.tool_trace.length) return null;
  return (
    <details className="trace" onClick={(e) => e.stopPropagation()}>
      <summary>Agent trace ({ev.tool_trace.length})</summary>
      <ol>
        {ev.tool_trace.map((t, i) => (
          <li key={i} className={t.is_error ? "trace__error" : undefined}>
            <code>{t.tool}</code> <span className="muted">{JSON.stringify(t.input)}</span>
          </li>
        ))}
      </ol>
    </details>
  );
}

function FieldCard({
  spec,
  ev,
  canPlay,
  active,
  onSeek,
  onSave,
}: {
  spec: SchemaField;
  ev: FieldEvidence;
  canPlay: boolean;
  active: boolean;
  onSeek: (f: FieldEvidence) => void;
  onSave: (path: string, value: unknown) => Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const seekable = canPlay && ev.start_ms != null;
  return (
    <li
      className={`field ${seekable ? "field--seekable" : ""} ${active ? "field--active" : ""}`}
      onClick={() => seekable && !editing && onSeek(ev)}
      title={seekable ? `Play from ${formatMs(ev.start_ms)}` : undefined}
    >
      <div className="field__head">
        <span className="field__label">
          {spec.label}
          {spec.required && <span className="req">*</span>}
        </span>
        <ConfidenceBadge ev={ev} />
      </div>
      {editing ? (
        <FieldEditor
          field={spec}
          value={ev.value}
          onCancel={() => setEditing(false)}
          onSave={async (v) => {
            await onSave(spec.path, v);
            setEditing(false);
          }}
        />
      ) : (
        <div className="field__value">
          <span>{formatValue(ev.value)}</span>
          <button
            type="button"
            className="link"
            onClick={(e) => {
              e.stopPropagation();
              setEditing(true);
            }}
          >
            edit
          </button>
        </div>
      )}
      {ev.corrected && (
        <div className="field__orig">
          agent said: <s>{formatValue(ev.original_value)}</s>
        </div>
      )}
      {ev.quote && (
        <blockquote className="field__quote">
          {seekable && <span className="play-icon">▶</span>}
          <span className="mono muted">{formatMs(ev.start_ms)}</span> “{ev.quote}”
        </blockquote>
      )}
      {ev.warnings.map((w) => (
        <div key={w} className="warning">
          ⚠ {w}
        </div>
      ))}
      <Trace ev={ev} />
    </li>
  );
}

export default function FormPanel({ extraction, canPlay, activeField, onSeek, onSave }: Props) {
  const byPath = new Map(extraction.fields.map((f) => [f.field_path, f]));
  const missing = new Set(extraction.missing.filter((m) => !m.resolved).map((m) => m.field_path));
  const filled = extraction.schema_fields.filter((f) => byPath.get(f.path)?.value != null);
  const empty = extraction.schema_fields.filter((f) => byPath.get(f.path)?.value == null && !missing.has(f.path));
  const [addingPath, setAddingPath] = useState<string | null>(null);

  return (
    <section className="panel form-panel">
      <h2>
        Form <span className="muted">· {extraction.schema}</span>
      </h2>
      <p className="muted small">
        {filled.length} filled · {missing.size} missing · click a field to hear the moment it was said
      </p>
      <ul className="fields">
        {filled.map((spec) => (
          <FieldCard
            key={spec.path}
            spec={spec}
            ev={byPath.get(spec.path)!}
            canPlay={canPlay}
            active={activeField === spec.path}
            onSeek={onSeek}
            onSave={onSave}
          />
        ))}
      </ul>
      {empty.length > 0 && (
        <details className="empty-fields">
          <summary>Not mentioned in the call ({empty.length})</summary>
          <ul>
            {empty.map((spec) => (
              <li key={spec.path}>
                <span>{spec.label}</span>
                {addingPath === spec.path ? (
                  <FieldEditor
                    field={spec}
                    value={null}
                    onCancel={() => setAddingPath(null)}
                    onSave={async (v) => {
                      await onSave(spec.path, v);
                      setAddingPath(null);
                    }}
                  />
                ) : (
                  <button type="button" className="link" onClick={() => setAddingPath(spec.path)}>
                    add
                  </button>
                )}
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}
