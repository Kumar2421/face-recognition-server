import { useEffect, useMemo, useState } from 'react';
import { getApiBase, recognitionEvents, type RecognitionEvent } from '../lib/api';

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

export default function Rejections() {
  const [items, setItems] = useState<RecognitionEvent[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [camera, setCamera] = useState<string>('');
  const [mode, setMode] = useState<'rejected' | 'no_match'>('rejected');

  const cameras = useMemo(() => {
    const s = new Set<string>();
    for (const it of items) if (it.camera) s.add(it.camera);
    return Array.from(s).sort();
  }, [items]);

  async function load() {
    setLoading(true);
    setErr(null);
    try {
      const resp = await recognitionEvents({ decision: mode, limit: 200 });
      setItems(resp.items || []);
    } catch (e: any) {
      setErr(String(e));
      setItems([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, [mode]);

  const filtered = useMemo(() => {
    if (!camera) return items;
    return items.filter(i => String(i.camera || '') === camera);
  }, [items, camera]);

  return (
    <div>
      <h2>Review</h2>
      <div style={{ marginBottom: 12, color: '#9ca3af' }}>API Base: {getApiBase()}</div>

      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 12, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button
            onClick={() => setMode('rejected')}
            style={{
              background: mode === 'rejected' ? '#a3e635' : 'transparent',
              color: mode === 'rejected' ? '#111' : '#e5e5e5',
              border: '1px solid #222',
              padding: '6px 10px',
              borderRadius: 8,
              cursor: 'pointer',
              fontWeight: 700
            }}
          >
            Rejected
          </button>
          <button
            onClick={() => setMode('no_match')}
            style={{
              background: mode === 'no_match' ? '#a3e635' : 'transparent',
              color: mode === 'no_match' ? '#111' : '#e5e5e5',
              border: '1px solid #222',
              padding: '6px 10px',
              borderRadius: 8,
              cursor: 'pointer',
              fontWeight: 700
            }}
          >
            No Match
          </button>
        </div>

        <button onClick={load} style={{ background: '#a3e635', color: '#111', border: 'none', padding: '6px 10px', borderRadius: 8, cursor: 'pointer', fontWeight: 700 }}>
          Refresh
        </button>

        <label style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span style={{ color: '#9ca3af' }}>Camera</span>
          <select value={camera} onChange={e => setCamera(e.target.value)} style={{ background: '#111', color: '#e5e5e5', border: '1px solid #222', borderRadius: 8, padding: '6px 10px' }}>
            <option value="">All</option>
            {cameras.map(c => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </label>

        <div style={{ color: '#9ca3af' }}>
          {loading ? 'Loading…' : `${filtered.length} ${mode} events`}
        </div>
      </div>

      {err && <div style={{ color: '#ef4444', marginBottom: 12 }}>Error: {err}</div>}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12 }}>
        {filtered.map(ev => (
          <div key={ev.event_id} style={{ border: '1px solid #222', borderRadius: 12, padding: 12, background: '#0f0f0f' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, marginBottom: 8 }}>
              <div style={{ fontWeight: 700 }}>{ev.camera || '—'}</div>
              <div style={{ color: '#9ca3af', fontSize: 12 }}>{fmtSavedAt(ev)}</div>
            </div>

            {ev.image_path ? (
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
              <div style={{ color: '#9ca3af' }}>Decision</div>
              <div style={{ fontWeight: 700 }}>{ev.decision || '—'}</div>

              <div style={{ color: '#9ca3af' }}>Reason</div>
              <div style={{ color: '#ef4444', fontWeight: 700 }}>{ev.rejected_reason || '—'}</div>

              <div style={{ color: '#9ca3af' }}>Similarity</div>
              <div>{ev.similarity != null ? ev.similarity.toFixed(4) : '—'}</div>

              <div style={{ color: '#9ca3af' }}>Subject</div>
              <div style={{ wordBreak: 'break-all' }}>{ev.subject_id || '—'}</div>

              <div style={{ color: '#9ca3af' }}>Source</div>
              <div style={{ wordBreak: 'break-all' }}>{ev.source_path || '—'}</div>

              <div style={{ color: '#9ca3af' }}>Event ID</div>
              <div style={{ wordBreak: 'break-all', color: '#9ca3af' }}>{ev.event_id}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
