// Typed client for the FastAPI backend. In Docker, nginx serves the UI and proxies /api -> api:8000.
export const API_BASE: string = import.meta.env.VITE_API_BASE ?? "/api";

export type CallStatus = "queued" | "transcribing" | "transcribed" | "extracting" | "extracted" | "failed";

export interface CallSummary {
  id: number;
  filename: string;
  duration: number | null;
  language: string | null;
  status: CallStatus;
  error: string | null;
  source: string;
  has_audio: boolean;
  diarization_method: string | null;
  recorded_at: string | null;
  created_at: string | null;
  latest_extraction_id: number | null;
}

export interface Word {
  w: string;
  s: number;
  e: number;
  p: number;
}

export interface Segment {
  id: number;
  speaker: string;
  start_ms: number;
  end_ms: number;
  text: string;
  words: Word[];
}

export interface CallDetail extends CallSummary {
  transcript: string;
  segments: Segment[];
  extractions: { id: number; schema: string; status: string; termination_reason: string | null; created_at: string }[];
}

export interface TraceEntry {
  n?: number;
  iteration?: number;
  tool: string;
  input: Record<string, unknown>;
  output?: string;
  is_error?: boolean;
}

export interface FieldEvidence {
  field_path: string;
  value: unknown;
  confidence: number;
  start_ms: number | null;
  end_ms: number | null;
  quote: string | null;
  source: "agent" | "forced" | "human";
  warnings: string[];
  original_value: unknown;
  corrected: boolean;
  tool_trace: TraceEntry[];
}

export interface MissingField {
  field_path: string;
  reason: string;
  suggested_question: string;
  resolved: boolean;
}

export interface SchemaField {
  path: string;
  label: string;
  type: "string" | "boolean" | "number" | "integer" | "array" | "object";
  required: boolean;
  format: string | null;
  enum: string[] | null;
  description: string | null;
}

export interface Extraction {
  id: number;
  call_id: number;
  schema: string;
  status: string;
  termination_reason: string | null;
  language: string | null;
  model: string | null;
  iterations: number;
  tool_calls: number;
  error: string | null;
  payload: Record<string, unknown>;
  fields: FieldEvidence[];
  missing: MissingField[];
  corrections: { field_path: string; original_value: unknown; corrected_value: unknown; created_at: string }[];
  followup_script: string | null;
  schema_fields: SchemaField[];
}

export interface SchemaInfo {
  name: string;
  title: string;
  fields: number;
}

export class ApiError extends Error {
  constructor(public status: number, message: string, public body: unknown) {
    super(message);
  }
}

function messageFrom(body: unknown, fallback: string): string {
  if (body && typeof body === "object") {
    const b = body as Record<string, unknown>;
    const detail = b.detail;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object") {
      const d = detail as Record<string, unknown>;
      const errors = Array.isArray(d.errors) ? `: ${d.errors.join("; ")}` : "";
      if (typeof d.message === "string") return d.message + errors;
    }
    if (Array.isArray(detail)) return detail.map((d) => (d as { msg?: string }).msg ?? String(d)).join("; ");
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  const text = await res.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (!res.ok) throw new ApiError(res.status, messageFrom(body, `${res.status} ${res.statusText}`), body);
  return body as T;
}

const json = (method: string, data: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(data),
});

export const api = {
  listCalls: () => request<CallSummary[]>("/calls"),
  getCall: (id: number) => request<CallDetail>(`/calls/${id}`),
  deleteCall: (id: number) => request<null>(`/calls/${id}`, { method: "DELETE" }),
  uploadCall: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ call_id: number; status: CallStatus }>("/calls", { method: "POST", body: form });
  },
  extract: (callId: number, schema: string) => request<Extraction>(`/calls/${callId}/extract`, json("POST", { schema })),
  getExtraction: (id: number) => request<Extraction>(`/extractions/${id}`),
  patchField: (extractionId: number, fieldPath: string, value: unknown) =>
    request<Extraction>(`/extractions/${extractionId}/fields/${fieldPath}`, json("PATCH", { value })),
  listSchemas: () => request<SchemaInfo[]>("/schemas"),
  audioUrl: (callId: number) => `${API_BASE}/calls/${callId}/audio`,
};
