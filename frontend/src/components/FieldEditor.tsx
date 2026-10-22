import { useState } from "react";
import type { SchemaField } from "../api/client";

interface Props {
  field: SchemaField;
  value: unknown;
  onSave: (value: unknown) => Promise<void>;
  onCancel: () => void;
}

function toText(field: SchemaField, value: unknown): string {
  if (value === null || value === undefined) return field.type === "array" ? "[]" : "";
  if (field.type === "array" || field.type === "object") return JSON.stringify(value, null, 1);
  return String(value);
}

export default function FieldEditor({ field, value, onSave, onCancel }: Props) {
  const [text, setText] = useState(toText(field, value));
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const parse = (): unknown => {
    if (field.type === "boolean") return text === "" ? null : text === "true";
    if (field.type === "array" || field.type === "object") return text.trim() ? JSON.parse(text) : null;
    if (field.type === "number" || field.type === "integer") return text.trim() ? Number(text) : null;
    return text.trim() ? text.trim() : null;
  };

  const save = async () => {
    setError(null);
    let parsed: unknown;
    try {
      parsed = parse();
    } catch {
      setError("Invalid JSON");
      return;
    }
    setSaving(true);
    try {
      await onSave(parsed);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const common = {
    autoFocus: true,
    onKeyDown: (e: React.KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
      if (e.key === "Enter" && !(e.target instanceof HTMLTextAreaElement)) void save();
    },
  };

  let input: JSX.Element;
  if (field.type === "boolean") {
    input = (
      <select value={text} onChange={(e) => setText(e.target.value)} {...common}>
        <option value="">— (clear)</option>
        <option value="true">Yes</option>
        <option value="false">No</option>
      </select>
    );
  } else if (field.enum) {
    input = (
      <select value={text} onChange={(e) => setText(e.target.value)} {...common}>
        <option value="">— (clear)</option>
        {field.enum.map((v) => (
          <option key={v} value={v}>
            {v}
          </option>
        ))}
      </select>
    );
  } else if (field.type === "array" || field.type === "object") {
    input = <textarea rows={4} value={text} onChange={(e) => setText(e.target.value)} {...common} />;
  } else {
    input = <input value={text} onChange={(e) => setText(e.target.value)} placeholder={field.format ?? ""} {...common} />;
  }

  return (
    <div className="editor" onClick={(e) => e.stopPropagation()}>
      {input}
      {field.type === "array" && <div className="hint">JSON list, e.g. [{`{"name": "Julie Martin", "phone": "0687451236"}`}]</div>}
      <div className="editor__actions">
        <button type="button" className="primary" disabled={saving} onClick={() => void save()}>
          {saving ? "Saving…" : "Save"}
        </button>
        <button type="button" onClick={onCancel} disabled={saving}>
          Cancel
        </button>
      </div>
      {error && <div className="error">{error}</div>}
    </div>
  );
}
