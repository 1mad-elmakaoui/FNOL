import { useCallback, useEffect, useRef, useState } from "react";
import { api, type CallDetail, type Extraction, type FieldEvidence, type SchemaInfo } from "../api/client";
import { formatMs } from "../format";
import FormPanel from "./FormPanel";
import MissingFields from "./MissingFields";
import Transcript from "./Transcript";
import WaveformPlayer, { type PlayerHandle } from "./WaveformPlayer";

const BUSY = new Set(["queued", "transcribing", "extracting"]);

interface Props {
  callId: number;
  schemas: SchemaInfo[];
  onChanged: () => void;
  onDeleted: () => void;
}

export default function CallView({ callId, schemas, onChanged, onDeleted }: Props) {
  const [call, setCall] = useState<CallDetail | null>(null);
  const [extraction, setExtraction] = useState<Extraction | null>(null);
  const [schema, setSchema] = useState("fnol");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentMs, setCurrentMs] = useState(0);
  const [activeField, setActiveField] = useState<string | null>(null);
  const [highlight, setHighlight] = useState<{ start: number; end: number } | null>(null);
  const playerRef = useRef<PlayerHandle>(null);

  const loadExtraction = useCallback(async (id: number | null) => {
    setExtraction(id == null ? null : await api.getExtraction(id));
  }, []);

  // Load the call, then poll while it is being processed.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let lastExtraction: number | null | undefined;
    setCall(null);
    setExtraction(null);
    setError(null);
    setActiveField(null);
    setHighlight(null);
    const tick = async () => {
      try {
        const c = await api.getCall(callId);
        if (cancelled) return;
        setCall(c);
        if (c.latest_extraction_id !== lastExtraction) {
          lastExtraction = c.latest_extraction_id;
          await loadExtraction(c.latest_extraction_id);
          onChanged();
        }
        if (BUSY.has(c.status)) timer = setTimeout(tick, 2000);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [callId, loadExtraction, onChanged]);

  const seekField = (ev: FieldEvidence) => {
    if (ev.start_ms == null) return;
    setActiveField(ev.field_path);
    setHighlight({ start: ev.start_ms, end: ev.end_ms ?? ev.start_ms });
    playerRef.current?.playSnippet(ev.start_ms, ev.end_ms);
  };

  const seekSegment = (start: number, end: number) => {
    setActiveField(null);
    setHighlight(null);
    playerRef.current?.playSnippet(start, end);
  };

  const runExtraction = async () => {
    setRunning(true);
    setError(null);
    try {
      const ext = await api.extract(callId, schema);
      setExtraction(ext);
      setCall(await api.getCall(callId));
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  const saveField = async (fieldPath: string, value: unknown) => {
    if (!extraction) return;
    setExtraction(await api.patchField(extraction.id, fieldPath, value));
  };

  if (!call) return <div className="panel">{error ? <div className="error">{error}</div> : "Loading…"}</div>;
  const canPlay = call.has_audio;
  const canExtract = !BUSY.has(call.status) && call.segments.length > 0;

  return (
    <div className="call-view">
      <header className="call-view__head">
        <div>
          <h1>
            Call #{call.id} <span className="muted">{call.filename}</span>
          </h1>
          <div className="muted small">
            <span className={`status status--${call.status}`}>{call.status}</span> · language {call.language ?? "?"} ·{" "}
            {formatMs((call.duration ?? 0) * 1000)} · diarization {call.diarization_method ?? "–"}
            {call.error && <span className="error-inline"> · {call.error}</span>}
          </div>
        </div>
        <div className="actions">
          <select value={schema} onChange={(e) => setSchema(e.target.value)} disabled={running}>
            {schemas.map((s) => (
              <option key={s.name} value={s.name}>
                {s.name} — {s.title}
              </option>
            ))}
          </select>
          <button type="button" className="primary" disabled={!canExtract || running} onClick={() => void runExtraction()}>
            {running ? "Extracting… (agent running)" : extraction ? "Re-run extraction" : "Run extraction"}
          </button>
          <button
            type="button"
            className="danger"
            onClick={async () => {
              if (!confirm(`Delete call #${call.id}, its audio, transcript and extractions?`)) return;
              await api.deleteCall(call.id);
              onDeleted();
            }}
          >
            Delete
          </button>
        </div>
      </header>

      {error && <div className="error banner">{error}</div>}
      {BUSY.has(call.status) && (
        <div className="info banner">
          {call.status === "extracting" ? "The agent is filling the form…" : "Transcribing locally (faster-whisper + diarization)…"}
        </div>
      )}

      <WaveformPlayer
        ref={playerRef}
        callId={call.id}
        hasAudio={call.has_audio}
        segments={call.segments}
        onTime={setCurrentMs}
      />

      <div className="columns">
        <section className="panel transcript-panel">
          <h2>Transcript</h2>
          <Transcript
            segments={call.segments}
            currentMs={currentMs}
            highlight={highlight}
            canPlay={canPlay}
            onSeek={seekSegment}
          />
        </section>
        <div className="right-col">
          {extraction ? (
            <>
              <div className="run-meta muted small">
                Extraction #{extraction.id} · {extraction.model} · {extraction.termination_reason ?? extraction.status} ·{" "}
                {extraction.iterations} iterations · {extraction.tool_calls} tool calls
                {extraction.error && <span className="error-inline"> · {extraction.error}</span>}
              </div>
              <FormPanel
                extraction={extraction}
                canPlay={canPlay}
                activeField={activeField}
                onSeek={seekField}
                onSave={saveField}
              />
              <MissingFields extraction={extraction} onSave={saveField} />
            </>
          ) : (
            <section className="panel">
              <h2>Form</h2>
              <p className="muted">
                {canExtract
                  ? "No extraction yet. Pick a schema and run the agent."
                  : "Waiting for the transcript before the agent can run."}
              </p>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
