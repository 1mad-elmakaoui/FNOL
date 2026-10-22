import type { CallSummary } from "../api/client";
import { formatMs } from "../format";

interface Props {
  calls: CallSummary[];
  selectedId: number | null;
  onSelect: (id: number) => void;
}

export default function CallList({ calls, selectedId, onSelect }: Props) {
  if (!calls.length) return <p className="muted small">No calls yet. Upload one, or run `make seed`.</p>;
  return (
    <ul className="calls">
      {calls.map((c) => (
        <li key={c.id}>
          <button
            type="button"
            className={`call ${c.id === selectedId ? "call--selected" : ""}`}
            onClick={() => onSelect(c.id)}
          >
            <span className="call__name">
              #{c.id} {c.filename}
            </span>
            <span className="call__meta">
              <span className={`status status--${c.status}`}>{c.status}</span>
              {c.language && <span className="muted">{c.language}</span>}
              {c.duration != null && <span className="muted mono">{formatMs(c.duration * 1000)}</span>}
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}
