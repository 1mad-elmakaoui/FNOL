export function formatMs(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms)) return "–:––";
  const total = Math.max(0, ms) / 1000;
  const m = Math.floor(total / 60);
  const s = total - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

const SPEAKER_RGB: Record<string, [number, number, number]> = {
  agent: [59, 130, 246],
  caller: [249, 115, 22],
};
const EXTRA_RGB: [number, number, number][] = [
  [16, 185, 129],
  [168, 85, 247],
  [236, 72, 153],
];

export function speakerColor(speaker: string, alpha = 1): string {
  let rgb = SPEAKER_RGB[speaker];
  if (!rgb) {
    let h = 0;
    for (const ch of speaker) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    rgb = EXTRA_RGB[h % EXTRA_RGB.length];
  }
  return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
}

export type ConfidenceLevel = "high" | "medium" | "low";

/** green > 0.85, amber 0.6–0.85, red < 0.6 */
export function confidenceLevel(c: number): ConfidenceLevel {
  if (c > 0.85) return "high";
  if (c >= 0.6) return "medium";
  return "low";
}

export function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (Array.isArray(value)) {
    if (value.length === 0) return "None";
    return value
      .map((v) =>
        v && typeof v === "object"
          ? Object.values(v as Record<string, unknown>).filter((x) => x != null && x !== "").join(", ")
          : String(v),
      )
      .join(" · ");
  }
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
