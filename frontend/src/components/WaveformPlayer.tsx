import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin, { type Region } from "wavesurfer.js/dist/plugins/regions.esm.js";
import { api, type Segment } from "../api/client";
import { formatMs, speakerColor } from "../format";

/** A clicked field plays at least this long… */
export const SNIPPET_MIN_MS = 4000;
/** …and at most this long, even if the cited evidence is longer. */
export const SNIPPET_MAX_MS = 8000;

export interface PlayerHandle {
  /** Seek to startMs and play ~4 s (longer if the evidence span is longer, capped at 8 s). */
  playSnippet: (startMs: number, endMs?: number | null) => void;
}

interface Props {
  callId: number;
  hasAudio: boolean;
  segments: Segment[];
  onTime?: (ms: number) => void;
}

interface Snippet {
  start: number; // seconds
  stop: number; // seconds
}

export function snippetWindow(startMs: number, endMs: number | null | undefined, durationS: number): Snippet {
  const span = Math.min(Math.max((endMs ?? startMs) - startMs, SNIPPET_MIN_MS), SNIPPET_MAX_MS);
  const start = Math.min(Math.max(startMs / 1000, 0), Math.max(durationS - 0.05, 0));
  return { start, stop: Math.min(start + span / 1000, durationS) };
}

const WaveformPlayer = forwardRef<PlayerHandle, Props>(function WaveformPlayer(
  { callId, hasAudio, segments, onTime },
  ref,
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);
  const regionsRef = useRef<RegionsPlugin | null>(null);
  const readyRef = useRef(false);
  const pendingRef = useRef<(() => void) | null>(null);
  const snippetRef = useRef<Snippet | null>(null);
  const snippetRegionRef = useRef<Region | null>(null);
  const onTimeRef = useRef(onTime);
  const lastTickRef = useRef(0);
  onTimeRef.current = onTime;

  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [zoom, setZoom] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const clearSnippet = useCallback(() => {
    snippetRef.current = null;
    snippetRegionRef.current?.remove();
    snippetRegionRef.current = null;
  }, []);

  // Create / destroy the player when the call changes.
  useEffect(() => {
    readyRef.current = false;
    pendingRef.current = null;
    snippetRef.current = null;
    setReady(false);
    setPlaying(false);
    setTime(0);
    setError(null);
    if (!hasAudio || !containerRef.current) return;

    const regions = RegionsPlugin.create();
    const ws = WaveSurfer.create({
      container: containerRef.current,
      url: api.audioUrl(callId),
      height: 96,
      waveColor: "#94a3b8",
      progressColor: "#475569",
      cursorColor: "#ef4444",
      cursorWidth: 2,
      normalize: true,
      dragToSeek: true,
      autoCenter: true,
      plugins: [regions],
    });
    wsRef.current = ws;
    regionsRef.current = regions;

    ws.on("ready", (d) => {
      readyRef.current = true;
      setReady(true);
      setDuration(d);
      const pending = pendingRef.current;
      pendingRef.current = null;
      pending?.();
    });
    ws.on("error", (e) => setError(`Audio could not be loaded: ${String(e)}`));
    ws.on("play", () => setPlaying(true));
    ws.on("pause", () => setPlaying(false));
    ws.on("finish", () => {
      snippetRef.current = null;
      setPlaying(false);
    });
    // The user clicked/dragged on the waveform: they take over, no auto-stop.
    ws.on("interaction", () => clearSnippet());
    ws.on("timeupdate", (t) => {
      const snip = snippetRef.current;
      // Ignore stale ticks from before the seek; stop once the snippet window is played.
      if (snip && ws.isPlaying() && t >= snip.stop && t < snip.stop + 1.0) {
        ws.pause();
        snippetRef.current = null;
      }
      const now = performance.now();
      if (now - lastTickRef.current > 100 || !ws.isPlaying()) {
        lastTickRef.current = now;
        setTime(t);
        onTimeRef.current?.(Math.round(t * 1000));
      }
    });

    return () => {
      ws.destroy();
      wsRef.current = null;
      regionsRef.current = null;
      snippetRegionRef.current = null;
    };
  }, [callId, hasAudio, clearSnippet]);

  // Speaker-coloured regions, one per transcript segment.
  useEffect(() => {
    const regions = regionsRef.current;
    if (!ready || !regions) return;
    regions.getRegions().forEach((r) => r.id !== "snippet" && r.remove());
    for (const s of segments) {
      regions.addRegion({
        id: `seg-${s.id}`,
        start: s.start_ms / 1000,
        end: s.end_ms / 1000,
        color: speakerColor(s.speaker, 0.22),
        drag: false,
        resize: false,
      });
    }
  }, [ready, segments]);

  useEffect(() => {
    if (ready) wsRef.current?.zoom(zoom);
  }, [ready, zoom]);

  const playSnippet = useCallback(
    (startMs: number, endMs?: number | null) => {
      const run = () => {
        const ws = wsRef.current;
        const regions = regionsRef.current;
        if (!ws || !regions) return;
        const snip = snippetWindow(startMs, endMs, ws.getDuration());
        clearSnippet();
        ws.setTime(snip.start);
        snippetRef.current = snip; // set after the seek so pre-seek ticks cannot stop it
        snippetRegionRef.current = regions.addRegion({
          id: "snippet",
          start: snip.start,
          end: snip.stop,
          color: "rgba(239, 68, 68, 0.28)",
          drag: false,
          resize: false,
        });
        ws.play().catch((e: unknown) => {
          snippetRef.current = null;
          setError(`Playback was blocked by the browser (${String(e)}). Press play.`);
        });
      };
      if (readyRef.current) run();
      else pendingRef.current = run; // audio still loading: play as soon as it is ready
    },
    [clearSnippet],
  );

  useImperativeHandle(ref, () => ({ playSnippet }), [playSnippet]);

  if (!hasAudio) {
    return (
      <div className="player player--empty">
        No audio for this call (transcript-only fixture). Click-to-play is disabled; timestamps are still shown.
      </div>
    );
  }

  const speakers = Array.from(new Set(segments.map((s) => s.speaker)));
  return (
    <div className="player">
      <div ref={containerRef} className="player__wave" aria-label="Call waveform" />
      <div className="player__controls">
        <button
          type="button"
          disabled={!ready}
          onClick={() => {
            clearSnippet();
            void wsRef.current?.playPause();
          }}
        >
          {playing ? "❚❚ Pause" : "▶ Play"}
        </button>
        <span className="mono">
          {formatMs(time * 1000)} / {formatMs(duration * 1000)}
        </span>
        <label className="player__zoom">
          Zoom
          <input type="range" min={0} max={200} step={10} value={zoom} onChange={(e) => setZoom(Number(e.target.value))} />
        </label>
        <span className="player__legend">
          {speakers.map((s) => (
            <span key={s} className="legend-item">
              <span className="swatch" style={{ background: speakerColor(s, 0.7) }} />
              {s}
            </span>
          ))}
        </span>
        {!ready && !error && <span className="muted">Loading audio…</span>}
      </div>
      {error && <div className="error">{error}</div>}
    </div>
  );
});

export default WaveformPlayer;
