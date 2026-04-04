import { useEffect, useMemo, useState } from 'react';
import { getApiBase, recognitionCameras, recognitionEvents, type RecognitionEvent } from '../lib/api';

function parseCamParts(cam: string): Record<string, string> {
  const out: Record<string, string> = {};
  const s = String(cam || '').trim();
  if (!s) return out;
  for (const seg of s.split('__')) {
    const idx = seg.indexOf('=');
    if (idx <= 0) continue;
    const k = seg.slice(0, idx).trim();
    const v = seg.slice(idx + 1).trim();
    if (k) out[k] = v;
  }
  return out;
}

function camGroupLabel(cam: string): string {
  const p = parseCamParts(cam);
  if (p.BR && p.ZN) return `BR=${p.BR}__ZN=${p.ZN}`;
  if (p.BR) return `BR=${p.BR}`;
  if (p.ZN) return `ZN=${p.ZN}`;
  return 'Other';
}

function camOptionLabel(cam: string): string {
  const p = parseCamParts(cam);
  if (p.IO && p.CAM) return `IO=${p.IO}__CAM=${p.CAM}`;
  if (p.CAM) return `CAM=${p.CAM}`;
  return cam;
}

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

function getNoMatchAutoEnroll(ev: any): any {
  try {
    return ev?.meta?.decision?.no_match_auto_enroll || null;
  } catch {
    return null;
  }
}

export default function Rejections() {
  const [items, setItems] = useState<RecognitionEvent[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [camera, setCamera] = useState<string>('');
  const [mode, setMode] = useState<'rejected' | 'no_match'>('rejected');
  const [cameraOptions, setCameraOptions] = useState<string[]>([]);

  const cameras = useMemo(() => {
    const s = new Set<string>();
    for (const it of items) if (it.camera) s.add(it.camera);
    return Array.from(s).sort();
  }, [items]);

  const cameraDropdownItems = useMemo(() => {
    const src = (cameraOptions && cameraOptions.length ? cameraOptions : cameras) || [];
    return Array.from(new Set(src.filter(Boolean))).sort();
  }, [cameraOptions, cameras]);

  const cameraGrouped = useMemo(() => {
    const groups = new Map<string, string[]>();
    for (const cam of cameraDropdownItems) {
      const g = camGroupLabel(cam);
      const arr = groups.get(g) || [];
      arr.push(cam);
      groups.set(g, arr);
    }
    const out = Array.from(groups.entries())
      .map(([g, cams]) => [g, Array.from(new Set(cams)).sort()] as const)
      .sort((a, b) => String(a[0]).localeCompare(String(b[0])));
    return out;
  }, [cameraDropdownItems]);

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

  useEffect(() => {
    let cancelled = false;
    async function loadCameras() {
      try {
        const resp = await recognitionCameras({ limit: 50000 });
        const list = (resp?.items || []).map(s => String(s || '').trim()).filter(Boolean);
        if (cancelled) return;
        setCameraOptions(list);
      } catch {
        if (cancelled) return;
        // keep any previous list; fall back to events-derived cameras if none
      }
    }
    loadCameras();
    return () => {
      cancelled = true;
    };
  }, []);

  const filtered = useMemo(() => {
    if (!camera) return items;
    return items.filter(i => String(i.camera || '') === camera);
  }, [items, camera]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h2 style={{ fontSize: '1.875rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '4px' }}>Event Review</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9375rem' }}>Analyze rejected attempts and unrecognised individuals.</p>
        </div>
        <button onClick={load} className="primary" style={{ fontWeight: 600 }}>
          {loading ? 'Refreshing...' : 'Refresh Events'}
        </button>
      </header>

      <div className="card" style={{ display: 'flex', gap: '20px', alignItems: 'center', flexWrap: 'wrap', background: 'var(--bg-primary)' }}>
        <div style={{ display: 'flex', borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border)' }}>
          <button
            onClick={() => setMode('rejected')}
            style={{
              padding: '10px 20px',
              background: mode === 'rejected' ? 'var(--primary)' : 'var(--bg-primary)',
              color: mode === 'rejected' ? 'white' : 'var(--text-secondary)',
              border: 'none',
              borderRadius: 0,
              fontWeight: 600,
              transition: 'all 0.2s'
            }}
          >
            Rejected
          </button>
          <button
            onClick={() => setMode('no_match')}
            style={{
              padding: '10px 20px',
              background: mode === 'no_match' ? 'var(--primary)' : 'var(--bg-primary)',
              color: mode === 'no_match' ? 'white' : 'var(--text-secondary)',
              border: 'none',
              borderRadius: 0,
              fontWeight: 600,
              transition: 'all 0.2s'
            }}
          >
            No Match
          </button>
        </div>

        <div style={{ flex: 1, minWidth: '200px', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-muted)' }}>Camera Filter</span>
          <select value={camera} onChange={e => setCamera(e.target.value)} style={{ flex: 1, padding: '10px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}>
            <option value="">All Cameras</option>
            {cameraGrouped.map(([g, cams]) => (
              <optgroup key={g} label={g}>
                {cams.map(c => (
                  <option key={c} value={c}>{camOptionLabel(c)}</option>
                ))}
              </optgroup>
            ))}
          </select>
        </div>

        <div style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', fontWeight: 500 }}>
          Found {filtered.length} {mode} events
        </div>
      </div>

      {err && (
        <div style={{ padding: '16px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--error)', borderRadius: 'var(--radius-md)', color: 'var(--error)', fontWeight: 500 }}>
          {err}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))', gap: '20px' }}>
        {filtered.map(ev => (
          <div key={ev.event_id} className="card" style={{ display: 'flex', flexDirection: 'column', gap: '10px', padding: '10px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div style={{ overflow: 'hidden' }}>
                <h4 style={{ fontWeight: 700, fontSize: '0.875rem', color: 'var(--text-primary)', whiteSpace: 'nowrap', textOverflow: 'ellipsis' }}>{ev.camera || 'Unknown'}</h4>
                <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', fontWeight: 500 }}>{fmtSavedAt(ev)}</div>
              </div>
            </div>

            {mode === 'no_match' && (() => {
              const nm = getNoMatchAutoEnroll(ev);
              if (!nm || !nm.enabled) return null;
              const enrolled = Boolean(nm.enrolled);
              const sid = String(nm.subject_id || '');
              return (
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '10px', 
                  padding: '8px 12px', 
                  background: enrolled ? 'rgba(16, 185, 129, 0.05)' : 'rgba(107, 114, 128, 0.05)',
                  borderRadius: 'var(--radius-md)',
                  border: enrolled ? '1px solid rgba(16, 185, 129, 0.1)' : '1px solid var(--border)'
                }}>
                  <div style={{ 
                    width: '8px', 
                    height: '8px', 
                    borderRadius: '50%', 
                    background: enrolled ? 'var(--success)' : 'var(--text-muted)' 
                  }}></div>
                  <span style={{ fontSize: '0.75rem', fontWeight: 700, color: enrolled ? 'var(--success)' : 'var(--text-secondary)' }}>
                    {enrolled ? 'System Auto-Enrolled' : 'Not Auto-Enrolled'}
                  </span>
                  {enrolled && sid && <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ID: {sid}</span>}
                </div>
              );
            })()}

            <div style={{ height: '150px', borderRadius: 'var(--radius-md)', overflow: 'hidden', background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
              {ev.image_path ? (
                <img src={`${getApiBase()}${ev.image_path}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              ) : (
                <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.875rem' }}>No image stored</div>
              )}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '70px 1fr', gap: '4px', fontSize: '0.75rem' }}>
              <span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>Decision</span>
              <span style={{ 
                fontWeight: 800, 
                color: ev.decision === 'rejected' ? 'var(--error)' : 'var(--text-secondary)',
                textTransform: 'uppercase'
              }}>{ev.decision}</span>

              {ev.rejected_reason && (
                <>
                  <span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>Reason</span>
                  <span style={{ color: 'var(--error)', fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={ev.rejected_reason || undefined}>{ev.rejected_reason}</span>
                </>
              )}

              <span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>Sim</span>
              <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{ev.similarity != null ? ev.similarity.toFixed(4) : '—'}</span>

              <span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>Subject</span>
              <span style={{ wordBreak: 'break-all', color: 'var(--text-primary)', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={ev.subject_id || undefined}>{ev.subject_id || '—'}</span>
            </div>

            <div style={{ borderTop: '1px solid var(--border)', paddingTop: '12px', opacity: 0.6 }}>
              <div style={{ fontSize: '0.625rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <span style={{ wordBreak: 'break-all' }}>Source: {ev.source_path}</span>
                <span>ID: {ev.event_id}</span>
              </div>
            </div>
          </div>
        ))}
        {filtered.length === 0 && !loading && (
          <div style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '80px', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-lg)', border: '2px dashed var(--border)', color: 'var(--text-muted)' }}>
            No {mode} events found for the selected camera.
          </div>
        )}
      </div>
    </div>
  );
}
