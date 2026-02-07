export function getApiBase(): string {
  try {
    const saved = localStorage.getItem('api_base');
    if (saved && saved.trim()) return saved.trim();
  } catch { }
  return (import.meta.env.VITE_API_BASE as string) || 'http://localhost:8001';
}

export async function apiGet<T = any>(path: string): Promise<T> {
  const r = await fetch(`${getApiBase()}${path}`);
  if (!r.ok) throw new Error(`GET ${path} failed: ${r.status}`);
  return r.json();
}

export async function apiPostJson<T = any>(path: string, body: any, headers: Record<string, string> = {}): Promise<T> {
  const r = await fetch(`${getApiBase()}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...headers },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`POST ${path} failed: ${r.status}`);
  return r.json();
}

export async function apiPostForm<T = any>(path: string, form: FormData): Promise<T> {
  const r = await fetch(`${getApiBase()}${path}`, {
    method: 'POST',
    body: form,
  });
  if (!r.ok) throw new Error(`POST ${path} failed: ${r.status}`);
  return r.json();
}

export async function apiDelete<T = any>(path: string): Promise<T> {
  const r = await fetch(`${getApiBase()}${path}`, { method: 'DELETE' });
  if (!r.ok) throw new Error(`DELETE ${path} failed: ${r.status}`);
  return r.json();
}

// Backend helpers
export async function health(): Promise<any> {
  return apiGet('/health');
}

export type Stats = {
  subjects_total: number;
  embeddings_total: number;
  last_24h_enrolls: number;
  last_24h_searches: number;
  qdrant_enabled: boolean;
  qdrant_collection?: string | null;
};

export async function stats(): Promise<Stats> {
  return apiGet('/v1/stats');
}

export async function facesSubjects(): Promise<{ subjects: string[] }> {
  return apiGet('/v1/faces/subjects');
}

// Phase 2 endpoints
export type SubjectItem = { subject_id: string; embeddings_count: number };
export type SubjectsListResponse = { items: SubjectItem[]; cursor?: string | null };

export async function subjects(params: { cursor?: string | null; limit?: number; with_counts?: boolean } = {}): Promise<SubjectsListResponse> {
  const q = new URLSearchParams();
  if (params.cursor) q.set('cursor', params.cursor);
  if (params.limit != null) q.set('limit', String(params.limit));
  if (params.with_counts != null) q.set('with_counts', String(params.with_counts));
  const qs = q.toString();
  return apiGet(`/v1/subjects${qs ? `?${qs}` : ''}`);
}

export type SubjectImageItem = { image_id: string; thumb_path?: string | null; image_path?: string | null; created_at?: string | null; source?: string | null };
export type SubjectImagesResponse = { items: SubjectImageItem[]; cursor?: string | null };

export async function subjectImages(subjectId: string, params: { cursor?: string | null; limit?: number } = {}): Promise<SubjectImagesResponse> {
  const q = new URLSearchParams();
  if (params.cursor) q.set('cursor', params.cursor);
  if (params.limit != null) q.set('limit', String(params.limit));
  const qs = q.toString();
  return apiGet(`/v1/subjects/${encodeURIComponent(subjectId)}/images${qs ? `?${qs}` : ''}`);
}

export async function deleteSubject(subjectId: string): Promise<{ subject_id: string; deleted: boolean }> {
  return apiDelete(`/v1/faces/subjects/${encodeURIComponent(subjectId)}`);
}

export async function facesAddUpload(subjectId: string, files: File[]): Promise<any> {
  const form = new FormData();
  form.append('subject_id', subjectId);
  for (const f of files) form.append('files', f);
  return apiPostForm('/v1/faces/add_upload', form);
}

export async function facesSearchUpload(file: File, topK: number = 5): Promise<any> {
  const form = new FormData();
  form.append('file', file);
  form.append('top_k', String(topK));
  return apiPostForm('/v1/faces/search_upload', form);
}

export async function facesRecognizeUpload(file: File, topK: number = 5): Promise<any> {
  const form = new FormData();
  form.append('file', file);
  form.append('top_k', String(topK));
  return apiPostForm('/v1/faces/recognize_upload', form);
}

// Recognition events
export type RecognitionEvent = {
  event_id: string;
  ts: number;
  camera: string;
  source_path: string;
  decision: string;
  subject_id?: string | null;
  similarity?: number | null;
  rejected_reason?: string | null;
  bbox?: number[] | null;
  det_score?: number | null;
  image_path: string;
  thumb_path: string;
  image_saved_at?: number | null;
  meta?: any;
};

export type RecognitionEventsListResponse = { items: RecognitionEvent[]; cursor?: number | null };

export async function recognitionEvents(params: {
  decision?: string;
  camera?: string;
  subject_id?: string;
  limit?: number;
  cursor?: number | null;
} = {}): Promise<RecognitionEventsListResponse> {
  const q = new URLSearchParams();
  if (params.decision) q.set('decision', params.decision);
  if (params.camera) q.set('camera', params.camera);
  if (params.subject_id) q.set('subject_id', params.subject_id);
  if (params.limit != null) q.set('limit', String(params.limit));
  if (params.cursor != null) q.set('cursor', String(params.cursor));
  const qs = q.toString();
  return apiGet(`/v1/events/recognition${qs ? `?${qs}` : ''}`);
}
