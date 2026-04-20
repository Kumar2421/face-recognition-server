export function getApiBase(): string {
  try {
    const saved = localStorage.getItem('api_base');
    if (saved && saved.trim()) return saved.trim();
  } catch { }
  return (import.meta.env.VITE_API_BASE as string) || 'http://localhost:8001';
}

export async function apiGet<T = any>(path: string): Promise<T> {
  const r = await fetch(`${getApiBase()}${path}`, {
    cache: 'no-store',
    headers: { 'Cache-Control': 'no-cache' },
  });
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
  if (!r.ok) {
    let detail = '';
    try {
      detail = await r.text();
    } catch { }
    const suffix = detail ? ` ${detail.slice(0, 500)}` : '';
    throw new Error(`POST ${path} failed: ${r.status}${suffix}`);
  }
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
export type SubjectItem = { subject_id: string; embeddings_count: number; embeddings_cap?: number | null; embeddings_capped?: boolean | null };
export type SubjectsListResponse = { items: SubjectItem[]; cursor?: string | null };

export type DateFilter = {
  day?: string | null;
  from_day?: string | null;
  to_day?: string | null;
};

export async function subjects(params: { cursor?: string | null; limit?: number; with_counts?: boolean; q?: string | null } & DateFilter = {}): Promise<SubjectsListResponse> {
  const q = new URLSearchParams();
  if (params.cursor) q.set('cursor', params.cursor);
  if (params.limit != null) q.set('limit', String(params.limit));
  if (params.with_counts != null) q.set('with_counts', String(params.with_counts));
  if (params.q) q.set('q', String(params.q));
  if (params.day) q.set('day', String(params.day));
  if (params.from_day) q.set('from_day', String(params.from_day));
  if (params.to_day) q.set('to_day', String(params.to_day));
  const qs = q.toString();
  return apiGet(`/v1/subjects${qs ? `?${qs}` : ''}`);
}

export async function getSubject(subjectId: string): Promise<SubjectItem> {
  return apiGet(`/v1/subjects/${encodeURIComponent(subjectId)}`);
}

export type SubjectImageItem = { image_id: string; thumb_path?: string | null; image_path?: string | null; created_at?: string | null; source?: string | null };
export type SubjectImagesResponse = { items: SubjectImageItem[]; cursor?: string | null };

export async function subjectImages(subjectId: string, params: { cursor?: string | null; limit?: number } & DateFilter = {}): Promise<SubjectImagesResponse> {
  const q = new URLSearchParams();
  if (params.cursor) q.set('cursor', params.cursor);
  if (params.limit != null) q.set('limit', String(params.limit));
  if (params.day) q.set('day', String(params.day));
  if (params.from_day) q.set('from_day', String(params.from_day));
  if (params.to_day) q.set('to_day', String(params.to_day));
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

export async function facesSearchUpload(file: File, topK: number = 5, filter: DateFilter = {}): Promise<any> {
  const form = new FormData();
  form.append('file', file, file.name || 'query.jpg');
  form.append('top_k', String(topK));
  if (filter.day) form.append('day', String(filter.day));
  if (filter.from_day) form.append('from_day', String(filter.from_day));
  if (filter.to_day) form.append('to_day', String(filter.to_day));
  return apiPostForm('/v1/faces/search_upload', form);
}

export async function facesRecognizeUpload(file: File, topK: number = 5, filter: DateFilter = {}): Promise<any> {
  const form = new FormData();
  form.append('file', file, file.name || 'query.jpg');
  form.append('top_k', String(topK));
  if (filter.day) form.append('day', String(filter.day));
  if (filter.from_day) form.append('from_day', String(filter.from_day));
  if (filter.to_day) form.append('to_day', String(filter.to_day));
  return apiPostForm('/v1/faces/recognize_upload', form);
}

export async function qualityCheckUpload(file: File): Promise<any> {
  const form = new FormData();
  form.append('file', file, file.name || 'query.jpg');
  return apiPostForm('/v1/quality/check_upload', form);
}

export async function crossMatch(subjectId: string): Promise<any> {
  return apiGet(`/v1/faces/cross_match/${encodeURIComponent(subjectId)}`);
}

export type CrossCheckHit = {
  employee_subject_id: string;
  visitor_event_id: string;
  visitor_subject_id: string;
  similarity: number;
  top2_second?: number | null;
  top2_margin?: number | null;
  visitor_ts?: number | null;
  visitor_camera?: string | null;
  visitor_image_path?: string | null;
  visitor_thumb_path?: string | null;
};

export type CrossCheckResponse = { items: CrossCheckHit[] };

export async function crossCheckVisitorsVsEmployees(params: {
  camera?: string;
  day?: string | null;
  from_day?: string | null;
  to_day?: string | null;
  since_ts?: number;
  until_ts?: number;
  limit?: number;
} = {}): Promise<CrossCheckResponse> {
  const q = new URLSearchParams();
  if (params.camera) q.set('camera', String(params.camera));
  if (params.day) q.set('day', String(params.day));
  if (params.from_day) q.set('from_day', String(params.from_day));
  if (params.to_day) q.set('to_day', String(params.to_day));
  if (params.since_ts != null) q.set('since_ts', String(params.since_ts));
  if (params.until_ts != null) q.set('until_ts', String(params.until_ts));
  if (params.limit != null) q.set('limit', String(params.limit));
  const qs = q.toString();
  return apiGet(`/v1/cross_check/visitors_vs_employees${qs ? `?${qs}` : ''}`);
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
  processing_ms?: number | null;
  rejected_reason?: string | null;
  bbox?: number[] | null;
  det_score?: number | null;
  image_path: string;
  thumb_path?: string | null;
  image_saved_at?: number | null;
  meta?: any;
  feedback_label?: string | null;
  feedback_note?: string | null;
  feedback_updated_at?: number | null;
};

export type RecognitionEventsListResponse = { items: RecognitionEvent[]; cursor?: number | null };

export type RecognitionCamerasResponse = { items: string[] };

export type RecognitionStatsResponse = {
  total: number;
  match: number;
  no_match: number;
  rejection: number;
  by_camera: Record<string, Record<string, number>>;
};

export async function recognitionStats(params: {
  day?: string | null;
  from_day?: string | null;
  to_day?: string | null;
  since_ts?: number;
  until_ts?: number;
  camera?: string;
} = {}): Promise<RecognitionStatsResponse> {
  const q = new URLSearchParams();
  if (params.day) q.set('day', params.day);
  if (params.from_day) q.set('from_day', params.from_day);
  if (params.to_day) q.set('to_day', params.to_day);
  if (params.since_ts != null) q.set('since_ts', String(params.since_ts));
  if (params.until_ts != null) q.set('until_ts', String(params.until_ts));
  if (params.camera) q.set('camera', params.camera);
  const qs = q.toString();
  return apiGet(`/v1/events/recognition/stats${qs ? `?${qs}` : ''}`);
}

export async function recognitionEvents(params: {
  decision?: string;
  camera?: string;
  subject_id?: string;
  min_similarity?: number;
  max_similarity?: number;
  since_ts?: number;
  until_ts?: number;
  day?: string | null;
  from_day?: string | null;
  to_day?: string | null;
  limit?: number;
  cursor?: number | null;
} = {}): Promise<RecognitionEventsListResponse> {
  const q = new URLSearchParams();
  q.set('_cb', String(Date.now()));
  if (params.decision) q.set('decision', params.decision);
  if (params.camera) q.set('camera', params.camera);
  if (params.subject_id) q.set('subject_id', params.subject_id);
  if (params.min_similarity != null) q.set('min_similarity', String(params.min_similarity));
  if (params.max_similarity != null) q.set('max_similarity', String(params.max_similarity));
  if (params.since_ts != null) q.set('since_ts', String(params.since_ts));
  if (params.until_ts != null) q.set('until_ts', String(params.until_ts));
  if (params.day) q.set('day', params.day);
  if (params.from_day) q.set('from_day', params.from_day);
  if (params.to_day) q.set('to_day', params.to_day);
  if (params.limit != null) q.set('limit', String(params.limit));
  if (params.cursor != null) q.set('cursor', String(params.cursor));
  const qs = q.toString();
  return apiGet(`/v1/events/recognition${qs ? `?${qs}` : ''}`);
}

export async function recognitionCameras(params: { limit?: number } = {}): Promise<RecognitionCamerasResponse> {
  const q = new URLSearchParams();
  if (params.limit != null) q.set('limit', String(params.limit));
  const qs = q.toString();
  return apiGet(`/v1/events/recognition/cameras${qs ? `?${qs}` : ''}`);
}

export type EventFeedbackLabel = 'tp' | 'fp' | 'fn' | 'ignore' | '';

export type EventFeedbackResponse = {
  event_id: string;
  updated: boolean;
  feedback_label?: string | null;
  feedback_note?: string | null;
  feedback_updated_at?: number | null;
};

export async function setRecognitionEventFeedback(eventId: string, payload: { label?: EventFeedbackLabel | null; note?: string | null }): Promise<EventFeedbackResponse> {
  return apiPostJson(`/v1/events/recognition/${encodeURIComponent(eventId)}/feedback`, payload);
}

export type FeedbackStatsResponse = {
  total: number;
  labeled: number;
  unlabeled: number;
  tp: number;
  fp: number;
  fn: number;
  ignore: number;
  fp_rate_match?: number | null;
  by_decision: Record<string, Record<string, number>>;
};

export async function recognitionFeedbackStats(params: { since_ts?: number; until_ts?: number; camera?: string } = {}): Promise<FeedbackStatsResponse> {
  const q = new URLSearchParams();
  if (params.since_ts != null) q.set('since_ts', String(params.since_ts));
  if (params.until_ts != null) q.set('until_ts', String(params.until_ts));
  if (params.camera) q.set('camera', String(params.camera));
  const qs = q.toString();
  return apiGet(`/v1/events/recognition/feedback_stats${qs ? `?${qs}` : ''}`);
}

// Search events
export type SearchEventResult = {
  subject_id: string;
  similarity: number;
  point_id?: string | null;
  thumb_path?: string | null;
  image_path?: string | null;
  bbox?: number[] | null;
  det_score?: number | null;
  meta?: any;
};

export type SearchEvent = {
  event_id: string;
  ts: number;
  query_image_path: string;
  query_thumb_path: string;
  top_subject_id?: string | null;
  top_similarity?: number | null;
  results: SearchEventResult[];
  meta?: any;
};

export type SearchEventsListResponse = { items: SearchEvent[]; cursor?: number | null };

export type SearchEventsStatsResponse = { match: number; no_match: number; total: number };

export async function searchEvents(params: {
  limit?: number;
  cursor?: number | null;
  day?: string | null;
  from_day?: string | null;
  to_day?: string | null;
  since_ts?: number;
  until_ts?: number;
} = {}): Promise<SearchEventsListResponse> {
  const q = new URLSearchParams();
  if (params.limit != null) q.set('limit', String(params.limit));
  if (params.cursor != null) q.set('cursor', String(params.cursor));
  if (params.day) q.set('day', String(params.day));
  if (params.from_day) q.set('from_day', String(params.from_day));
  if (params.to_day) q.set('to_day', String(params.to_day));
  if (params.since_ts != null) q.set('since_ts', String(params.since_ts));
  if (params.until_ts != null) q.set('until_ts', String(params.until_ts));
  const qs = q.toString();
  return apiGet(`/v1/search_history${qs ? `?${qs}` : ''}`);
}

export async function searchEventsStats(params: {
  match_threshold: number;
  day?: string | null;
  from_day?: string | null;
  to_day?: string | null;
  since_ts?: number;
  until_ts?: number;
}): Promise<SearchEventsStatsResponse> {
  const q = new URLSearchParams();
  q.set('match_threshold', String(params.match_threshold));
  if (params.day) q.set('day', String(params.day));
  if (params.from_day) q.set('from_day', String(params.from_day));
  if (params.to_day) q.set('to_day', String(params.to_day));
  if (params.since_ts != null) q.set('since_ts', String(params.since_ts));
  if (params.until_ts != null) q.set('until_ts', String(params.until_ts));
  const qs = q.toString();
  return apiGet(`/v1/search_history/stats${qs ? `?${qs}` : ''}`);
}
