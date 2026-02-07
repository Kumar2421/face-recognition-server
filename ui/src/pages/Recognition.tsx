import { useEffect, useMemo, useState } from 'react';
import { getApiBase, recognitionEvents, subjectImages, type RecognitionEvent } from '../lib/api';

function fmtTs(ts: number): string {
  try {
    return new Date(ts * 1000).toLocaleString();
  } catch {
    return String(ts);
  }
}

function fmtSavedAt(ev: { image_saved_at?: number | null; ts: number }): string {
  const t = ev.image_saved_at != null ? Number(ev.image_saved_at) : Number(ev.ts);
  return fmtTs(t);
}

type DecisionFilter = '' | 'match' | 'no_match' | 'rejected';

export default function Recognition() {
  const [items, setItems] = useState<RecognitionEvent[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingMore, setLoadingMore] = useState<boolean>(false);
  const [camera, setCamera] = useState<string>('');
  const [decision, setDecision] = useState<DecisionFilter>('');
  const [subjectId, setSubjectId] = useState<string>('');
  const [nextCursor, setNextCursor] = useState<number | null>(null);
  const [pageSize, setPageSize] = useState<number>(200);
  const [subjectImgById, setSubjectImgById] = useState<Record<string, string>>({});

  const cameras = useMemo(() => {
    const s = new Set<string>();
    for (const it of items) if (it.camera) s.add(it.camera);
    return Array.from(s).sort();
  }, [items]);

  async function load(reset: boolean = true) {
    setLoading(true);
    setErr(null);
    try {
      const cur = reset ? null : nextCursor;
      const resp = await recognitionEvents({
        decision: decision || undefined,
        camera: camera || undefined,
        subject_id: subjectId || undefined,
        limit: pageSize,
        cursor: cur,
      });
      if (reset) {
        setItems(resp.items || []);
      } else {
        setItems(prev => [...(prev || []), ...(resp.items || [])]);
      }
      setNextCursor(resp.cursor != null ? Number(resp.cursor) : null);
    } catch (e: any) {
      setErr(String(e));
      if (reset) setItems([]);
      setNextCursor(null);
    } finally {
      setLoading(false);
    }
  }

  async function loadMore() {
    if (loadingMore) return;
    if (nextCursor == null) return;
    setLoadingMore(true);
    setErr(null);
    try {
      const resp = await recognitionEvents({
        decision: decision || undefined,
        camera: camera || undefined,
        subject_id: subjectId || undefined,
        limit: pageSize,
        cursor: nextCursor,
      });
      setItems(prev => [...(prev || []), ...(resp.items || [])]);
      setNextCursor(resp.cursor != null ? Number(resp.cursor) : null);
    } catch (e: any) {
      setErr(String(e));
    } finally {
      setLoadingMore(false);
    }
  }

  useEffect(() => {
    load(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    setNextCursor(null);
    load(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [decision, camera, subjectId, pageSize]);

  useEffect(() => {
    let cancelled = false;

    async function prefetchSubjectImages() {
      const want: string[] = [];
      for (const ev of items) {
        if (String(ev.decision || '') !== 'match') continue;
        const sid = String(ev.subject_id || '').trim();
        if (!sid) continue;
        if (subjectImgById[sid]) continue;
        want.push(sid);
      }

      const uniq = Array.from(new Set(want)).slice(0, 50);
      if (!uniq.length) return;

      for (const sid of uniq) {
        try {
          const resp = await subjectImages(sid, { limit: 1 });
          const first = resp?.items?.[0];
          const p = (first?.thumb_path || first?.image_path || '').trim();
          if (!p) continue;
          if (cancelled) return;
          setSubjectImgById(prev => (prev[sid] ? prev : { ...prev, [sid]: p }));
        } catch {
          // ignore
        }
      }
    }

    prefetchSubjectImages();
    return () => {
      cancelled = true;
    };
  }, [items, subjectImgById]);

  const filtered = useMemo(() => {
    let out = items;
    if (decision) out = out.filter(i => String(i.decision || '') === decision);
    if (camera) out = out.filter(i => String(i.camera || '') === camera);
    if (subjectId) out = out.filter(i => String(i.subject_id || '') === subjectId);
    return out;
  }, [items, decision, camera, subjectId]);

  const counts = useMemo(() => {
    let match = 0;
    let noMatch = 0;
    let rejected = 0;
    for (const it of filtered) {
      const d = String(it.decision || '');
      if (d === 'match') match += 1;
      else if (d === 'no_match') noMatch += 1;
      else if (d === 'rejected') rejected += 1;
    }
    return { match, noMatch, rejected, total: filtered.length };
  }, [filtered]);

  return (
    <div>
      <h2>Recognition</h2>
      <div style={{ marginBottom: 12, color: '#9ca3af' }}>API Base: {getApiBase()}</div>

      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 12, flexWrap: 'wrap' }}>
        <button onClick={() => load(true)} style={{ background: '#a3e635', color: '#111', border: 'none', padding: '6px 10px', borderRadius: 8, cursor: 'pointer', fontWeight: 700 }}>
          Refresh
        </button>

        <button
          onClick={() => {
            setNextCursor(null);
            load(true);
          }}
          style={{ background: 'transparent', color: '#e5e5e5', border: '1px solid #222', padding: '6px 10px', borderRadius: 8, cursor: 'pointer', fontWeight: 700 }}
        >
          Reset
        </button>

        <label style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span style={{ color: '#9ca3af' }}>Page size</span>
          <input
            type="number"
            min={50}
            max={500}
            value={pageSize}
            onChange={e => setPageSize(Math.max(50, Math.min(500, Number(e.target.value) || 200)))}
            style={{ width: 90, background: '#111', color: '#e5e5e5', border: '1px solid #222', borderRadius: 8, padding: '6px 10px' }}
          />
        </label>

        <label style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span style={{ color: '#9ca3af' }}>Decision</span>
          <select value={decision} onChange={e => setDecision(e.target.value as DecisionFilter)} style={{ background: '#111', color: '#e5e5e5', border: '1px solid #222', borderRadius: 8, padding: '6px 10px' }}>
            <option value="">All</option>
            <option value="match">Match</option>
            <option value="no_match">No Match</option>
            <option value="rejected">Rejected</option>
          </select>
        </label>

        <label style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span style={{ color: '#9ca3af' }}>Camera</span>
          <select value={camera} onChange={e => setCamera(e.target.value)} style={{ background: '#111', color: '#e5e5e5', border: '1px solid #222', borderRadius: 8, padding: '6px 10px' }}>
            <option value="">All</option>
            {cameras.map(c => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </label>

        <label style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span style={{ color: '#9ca3af' }}>Subject</span>
          <input value={subjectId} onChange={e => setSubjectId(e.target.value)} placeholder="subject_id" style={{ background: '#111', color: '#e5e5e5', border: '1px solid #222', borderRadius: 8, padding: '6px 10px' }} />
        </label>

        <div style={{ color: '#9ca3af' }}>
          {loading ? 'Loading…' : `${counts.total} events | match ${counts.match} | no_match ${counts.noMatch} | rejected ${counts.rejected}`}
        </div>
      </div>

      {err && <div style={{ color: '#ef4444', marginBottom: 12 }}>Error: {err}</div>}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12 }}>
        {filtered.map(ev => (
          <div key={ev.event_id} style={{ border: '1px solid #222', borderRadius: 12, padding: 12, background: '#0f0f0f' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, marginBottom: 8 }}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <div style={{ fontWeight: 800 }}>{ev.camera || '—'}</div>
                <div style={{ padding: '2px 8px', borderRadius: 999, border: '1px solid #222', background: '#111', fontSize: 12, color: '#d1d5db' }}>{ev.decision}</div>
              </div>
              <div style={{ color: '#9ca3af', fontSize: 12 }}>{fmtSavedAt(ev)}</div>
            </div>

            {String(ev.decision || '') === 'match' && ev.subject_id && subjectImgById[String(ev.subject_id)] ? (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 8 }}>
                <img
                  src={`${getApiBase()}${ev.image_path}`}
                  style={{ width: '100%', height: 180, objectFit: 'cover', borderRadius: 10, border: '1px solid #111' }}
                />
                <img
                  src={`${getApiBase()}${subjectImgById[String(ev.subject_id)]}`}
                  style={{ width: '100%', height: 180, objectFit: 'cover', borderRadius: 10, border: '1px solid #111' }}
                />
              </div>
            ) : ev.image_path ? (
              <img
                src={`${getApiBase()}${ev.image_path}`}
                style={{ width: '100%', height: 180, objectFit: 'cover', borderRadius: 10, border: '1px solid #111', marginBottom: 8 }}
              />
            ) : (
              <div style={{ height: 180, borderRadius: 10, border: '1px dashed #333', marginBottom: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#9ca3af' }}>
                No image stored
              </div>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: '90px 1fr', gap: 6, fontSize: 13 }}>
              <div style={{ color: '#9ca3af' }}>Subject</div>
              <div style={{ wordBreak: 'break-all' }}>{ev.subject_id || '—'}</div>

              <div style={{ color: '#9ca3af' }}>Similarity</div>
              <div>{ev.similarity != null ? ev.similarity.toFixed(4) : '—'}</div>

              <div style={{ color: '#9ca3af' }}>Reason</div>
              <div style={{ color: '#ef4444', fontWeight: 700 }}>{ev.rejected_reason || '—'}</div>

              <div style={{ color: '#9ca3af' }}>Source</div>
              <div style={{ wordBreak: 'break-all' }}>{ev.source_path || '—'}</div>

              <div style={{ color: '#9ca3af' }}>Event ID</div>
              <div style={{ wordBreak: 'break-all', color: '#9ca3af' }}>{ev.event_id}</div>
            </div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 14, display: 'flex', gap: 8, alignItems: 'center' }}>
        <button
          disabled={nextCursor == null || loadingMore}
          onClick={loadMore}
          style={{
            background: nextCursor == null ? '#111' : '#0ea5e9',
            color: '#111',
            border: 'none',
            padding: '6px 10px',
            borderRadius: 8,
            cursor: nextCursor == null ? 'not-allowed' : 'pointer',
            opacity: nextCursor == null ? 0.6 : 1,
            fontWeight: 800
          }}
        >
          {loadingMore ? 'Loading…' : (nextCursor == null ? 'No more' : 'Load more')}
        </button>
        <div style={{ color: '#9ca3af', fontSize: 12 }}>
          Loaded: {filtered.length}
        </div>
      </div>
    </div>
  );
}
