import { useCallback, useEffect, useState } from "react";
import { api, type CallSummary, type SchemaInfo } from "./api/client";
import CallList from "./components/CallList";
import CallView from "./components/CallView";
import UploadArea from "./components/UploadArea";

export default function App() {
  const [calls, setCalls] = useState<CallSummary[]>([]);
  const [schemas, setSchemas] = useState<SchemaInfo[]>([]);
  const [selected, setSelected] = useState<number | null>(() => {
    const fromHash = Number(window.location.hash.replace("#call-", ""));
    return Number.isFinite(fromHash) && fromHash > 0 ? fromHash : null;
  });
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setCalls(await api.listCalls());
      setError(null);
    } catch (e) {
      setError(`API unreachable: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, []);

  useEffect(() => {
    void refresh();
    api.listSchemas().then(setSchemas).catch(() => setSchemas([{ name: "fnol", title: "FNOL", fields: 16 }]));
    const t = setInterval(() => void refresh(), 4000);
    return () => clearInterval(t);
  }, [refresh]);

  useEffect(() => {
    window.location.hash = selected ? `call-${selected}` : "";
  }, [selected]);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          FNOL <span className="muted">voice → form</span>
        </div>
        <UploadArea
          onUploaded={(id) => {
            setSelected(id);
            void refresh();
          }}
        />
        {error && <div className="error small">{error}</div>}
        <h3>Calls</h3>
        <CallList calls={calls} selectedId={selected} onSelect={setSelected} />
      </aside>
      <main className="main">
        {selected ? (
          <CallView
            key={selected}
            callId={selected}
            schemas={schemas}
            onChanged={refresh}
            onDeleted={() => {
              setSelected(null);
              void refresh();
            }}
          />
        ) : (
          <div className="placeholder">
            <h1>Review extracted claims</h1>
            <p className="muted">
              Upload a recorded call on the left. It is transcribed locally, then the agent fills the form with a
              timestamp for every value. Click any field to hear the caller say it.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
