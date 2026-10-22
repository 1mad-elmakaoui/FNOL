import { useRef, useState } from "react";
import { api } from "../api/client";

interface Props {
  onUploaded: (callId: number) => void;
}

export default function UploadArea({ onUploaded }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const upload = async (file: File | undefined) => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const { call_id } = await api.uploadCall(file);
      onUploaded(call_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  return (
    <div
      className={`upload ${dragging ? "upload--drag" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        void upload(e.dataTransfer.files[0]);
      }}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
    >
      <input
        ref={inputRef}
        type="file"
        accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac,.webm,.opus"
        hidden
        onChange={(e) => void upload(e.target.files?.[0])}
      />
      <strong>{busy ? "Uploading…" : "Drop a call recording"}</strong>
      <span className="muted small">or click to choose · wav, mp3, m4a, ogg, flac · stays on this machine</span>
      {error && <div className="error">{error}</div>}
    </div>
  );
}
