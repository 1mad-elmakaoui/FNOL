import { useState } from "react";
import type { Extraction } from "../api/client";
import FieldEditor from "./FieldEditor";

interface Props {
  extraction: Extraction;
  onSave: (fieldPath: string, value: unknown) => Promise<void>;
}

export default function MissingFields({ extraction, onSave }: Props) {
  const [editing, setEditing] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const specs = new Map(extraction.schema_fields.map((f) => [f.path, f]));
  const open = extraction.missing.filter((m) => !m.resolved);

  return (
    <section className="panel missing-panel">
      <h2>
        Missing fields <span className="count">{open.length}</span>
      </h2>
      {open.length === 0 && <p className="muted">Nothing missing.</p>}
      <ul className="missing">
        {open.map((m) => {
          const spec = specs.get(m.field_path);
          return (
            <li key={m.field_path}>
              <div className="missing__head">
                <strong>{spec?.label ?? m.field_path}</strong>
                {spec?.required && <span className="req">required</span>}
              </div>
              <div className="missing__q">“{m.suggested_question}”</div>
              <div className="muted small">{m.reason}</div>
              {spec &&
                (editing === m.field_path ? (
                  <FieldEditor
                    field={spec}
                    value={null}
                    onCancel={() => setEditing(null)}
                    onSave={async (v) => {
                      await onSave(m.field_path, v);
                      setEditing(null);
                    }}
                  />
                ) : (
                  <button type="button" className="link" onClick={() => setEditing(m.field_path)}>
                    fill in
                  </button>
                ))}
            </li>
          );
        })}
      </ul>
      {extraction.followup_script && (
        <details className="script" open={open.length > 0}>
          <summary>Follow-up call script</summary>
          <pre dir="auto">{extraction.followup_script}</pre>
          <button
            type="button"
            onClick={() => {
              void navigator.clipboard?.writeText(extraction.followup_script ?? "").then(() => {
                setCopied(true);
                setTimeout(() => setCopied(false), 1500);
              });
            }}
          >
            {copied ? "Copied ✓" : "Copy script"}
          </button>
        </details>
      )}
    </section>
  );
}
