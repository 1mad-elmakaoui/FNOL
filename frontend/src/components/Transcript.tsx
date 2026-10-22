import { useEffect, useRef } from "react";
import type { Segment } from "../api/client";
import { formatMs, speakerColor } from "../format";

interface Props {
  segments: Segment[];
  currentMs: number;
  highlight: { start: number; end: number } | null;
  canPlay: boolean;
  onSeek: (startMs: number, endMs: number) => void;
}

export default function Transcript({ segments, currentMs, highlight, canPlay, onSeek }: Props) {
  const activeRef = useRef<HTMLLIElement>(null);
  const activeIdx = segments.findIndex((s) => currentMs >= s.start_ms && currentMs < s.end_ms);

  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [activeIdx]);

  if (!segments.length) return <p className="muted">No transcript yet.</p>;
  return (
    <ol className="transcript" dir="auto">
      {segments.map((s, i) => {
        const inEvidence = highlight && s.end_ms >= highlight.start && s.start_ms <= highlight.end;
        return (
          <li
            key={s.id}
            ref={i === activeIdx ? activeRef : undefined}
            className={`seg ${i === activeIdx ? "seg--active" : ""} ${inEvidence ? "seg--evidence" : ""} ${canPlay ? "seg--seekable" : ""}`}
            style={{ borderLeftColor: speakerColor(s.speaker) }}
            onClick={() => canPlay && onSeek(s.start_ms, s.end_ms)}
          >
            <span className="seg__meta">
              <span className="seg__speaker" style={{ color: speakerColor(s.speaker) }}>
                {s.speaker}
              </span>
              <span className="mono muted">{formatMs(s.start_ms)}</span>
            </span>
            <span className="seg__text">{s.text}</span>
          </li>
        );
      })}
    </ol>
  );
}
