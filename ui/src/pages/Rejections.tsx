import { useEffect, useMemo, useState } from 'react';
import { getApiBase, recognitionCameras, recognitionEvents, recognitionStats, type RecognitionEvent } from '../lib/api';

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
  const [selectedDate, setSelectedDate] = useState<string>(new Date().toLocaleDateString('en-CA'));

  useEffect(() => {
    // Sync selectedDate with the latest event date if available
    if (items.length > 0) {
      const validTss = items.map(it => Number(it.ts || 0)).filter(t => t > 0);
      if (validTss.length > 0) {
        const latestTs = Math.max(...validTss);
        const latestDate = new Intl.DateTimeFormat('en-CA', {
          timeZone: 'Asia/Kolkata',
          year: 'numeric',
          month: '2-digit',
          day: '2-digit'
        }).format(new Date(latestTs * 1000));
        setSelectedDate(latestDate);
      }
    }
  }, [items]);

  const [cursor, setCursor] = useState<number | null>(null);
  const [hasMore, setHasMore] = useState<boolean>(true);

  const [uniqueCount, setUniqueCount] = useState<number | null>(null);
  const [uniqueCountLoading, setUniqueCountLoading] = useState<boolean>(false);

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

  useEffect(() => {
    let cancelled = false;
    async function loadUniqueCount() {
      if (!selectedDate) return;
      setUniqueCountLoading(true);
      try {
        const resp = await recognitionStats({
          day: selectedDate,
          camera: camera || undefined
        });
        if (cancelled) return;
        setUniqueCount(resp.unique_matches ?? resp.match ?? 0);
      } catch (e) {
        console.error('Failed to load unique count', e);
      } finally {
        if (!cancelled) setUniqueCountLoading(false);
      }
    }
    loadUniqueCount();
    return () => {
      cancelled = true;
    };
  }, [selectedDate, camera]);

  async function load(isMore = false) {
    if (loading) return;
    setLoading(true);
    setErr(null);
    try {
      const limit = 50;
      const resp = await recognitionEvents({
        decision: mode,
        limit,
        cursor: isMore ? cursor : undefined,
        day: selectedDate || undefined,
        camera: camera || undefined
      });

      const newItems = resp.items || [];
      if (isMore) {
        setItems(prev => [...prev, ...newItems]);
      } else {
        setItems(newItems);
      }

      setCursor(resp.cursor || null);
      setHasMore(newItems.length >= limit);
    } catch (e: any) {
      setErr(String(e));
      if (!isMore) setItems([]);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    setCursor(null);
    setHasMore(true);
    load(false);
  }, [mode, selectedDate]);

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
      }
    }
    loadCameras();
    return () => {
      cancelled = true;
    };
  }, []);

  const filtered = useMemo(() => {
    const src = camera ? items.filter(i => String(i.camera || '') === camera) : items;
    const groups = new Map<string, RecognitionEvent & { all_images: string[], minTs: number, maxTs: number }>();

    for (const ev of src) {
      const sid = String(ev.subject_id || `unknown_${ev.event_id}`).trim().toLowerCase();
      const ts = ev.image_saved_at != null ? Number(ev.image_saved_at) : Number(ev.ts);

      if (!groups.has(sid)) {
        groups.set(sid, {
          ...ev,
          subject_id: ev.subject_id || sid,
          all_images: ev.image_path ? [ev.image_path] : [],
          minTs: ts,
          maxTs: ts
        });
      } else {
        const g = groups.get(sid)!;
        if (ev.image_path && !g.all_images.includes(ev.image_path)) {
          g.all_images.push(ev.image_path);
        }
        if (ts < g.minTs) g.minTs = ts;
        if (ts > g.maxTs) g.maxTs = ts;
        if (ev.similarity != null && (g.similarity == null || ev.similarity > g.similarity)) {
          g.similarity = ev.similarity;
          g.camera = ev.camera;
          g.rejected_reason = ev.rejected_reason;
          g.subject_id = ev.subject_id || g.subject_id;
        }
      }
    }

    return Array.from(groups.values()).sort((a, b) => b.maxTs - a.maxTs);
  }, [items, camera]);

  function fmtDuration(s: number, e: number): string {
    const diff = Math.max(0, e - s);
    const mins = Math.floor(diff / 60);
    const secs = Math.floor(diff % 60);
    if (mins > 0) return `${mins}m ${secs}s`;
    return `${secs}s`;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h2 style={{ fontSize: '1.875rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '4px' }}>Event Review</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9375rem' }}>Analyze rejected attempts and unrecognised individuals (grouped by ID).</p>
        </div>
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <div className="card" style={{ padding: '8px 20px', display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: '140px', background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)' }}>
            <span style={{ fontSize: '0.65rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Unique Count</span>
            <span style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--primary)' }}>
              {uniqueCountLoading ? '...' : (uniqueCount ?? 0)}
            </span>
          </div>
          <button onClick={() => load(false)} className="primary" style={{ fontWeight: 600 }}>
            {loading ? 'Refreshing...' : 'Refresh Events'}
          </button>
        </div>
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

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-muted)' }}>Date</span>
          <input
            type="date"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            style={{ padding: '8px 12px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}
          />
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
          Found {filtered.length} {mode} groups
        </div>
      </div>

      {err && (
        <div style={{ padding: '16px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--error)', borderRadius: 'var(--radius-md)', color: 'var(--error)', fontWeight: 500 }}>
          {err}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: '20px' }}>
        {filtered.map(ev => (
          <div key={`${ev.subject_id}_${ev.minTs}`} className="card" style={{ display: 'flex', flexDirection: 'column', gap: '10px', padding: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div style={{ overflow: 'hidden' }}>
                <h4 style={{ fontWeight: 800, fontSize: '0.9375rem', color: 'var(--text-primary)', whiteSpace: 'nowrap', textOverflow: 'ellipsis' }}>{ev.subject_id || 'Unknown'}</h4>
                <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', fontWeight: 500 }}>
                  {new Date(ev.minTs * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} - {new Date(ev.maxTs * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            </div>

            {mode === 'no_match' && (() => {
              const nm = getNoMatchAutoEnroll(ev);
              if (!nm || !nm.enabled) return null;
              const enrolled = Boolean(nm.enrolled);
              return (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '6px 10px',
                  background: enrolled ? 'rgba(16, 185, 129, 0.05)' : 'rgba(107, 114, 128, 0.05)',
                  borderRadius: 'var(--radius-md)',
                  border: enrolled ? '1px solid rgba(16, 185, 129, 0.1)' : '1px solid var(--border)'
                }}>
                  <div style={{
                    width: '6px',
                    height: '6px',
                    borderRadius: '50%',
                    background: enrolled ? 'var(--success)' : 'var(--text-muted)'
                  }}></div>
                  <span style={{ fontSize: '0.7rem', fontWeight: 700, color: enrolled ? 'var(--success)' : 'var(--text-secondary)' }}>
                    {enrolled ? 'Auto-Enrolled' : 'No Auto-Enroll'}
                  </span>
                </div>
              );
            })()}

            <div style={{ height: '160px', borderRadius: 'var(--radius-md)', overflow: 'hidden', background: 'var(--bg-secondary)', border: '1px solid var(--border)', position: 'relative' }}>
              {ev.all_images && ev.all_images.length > 0 ? (
                <div style={{ display: 'flex', overflowX: 'auto', scrollSnapType: 'x mandatory', height: '100%', gap: '2px' }}>
                  {ev.all_images.map((img, idx) => (
                    <img
                      key={idx}
                      src={`${getApiBase()}${img}`}
                      style={{ height: '100%', minWidth: '100%', objectFit: 'cover', scrollSnapAlign: 'start' }}
                    />
                  ))}
                  {ev.all_images.length > 1 && (
                    <div style={{ position: 'absolute', bottom: '8px', right: '8px', background: 'rgba(0,0,0,0.6)', color: 'white', padding: '2px 6px', borderRadius: '10px', fontSize: '0.625rem', fontWeight: 700 }}>
                      1 / {ev.all_images.length}
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.875rem' }}>No image</div>
              )}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '70px 1fr', gap: '4px', fontSize: '0.75rem' }}>
              <span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>Duration</span>
              <span style={{ fontWeight: 800, color: 'var(--text-primary)' }}>{fmtDuration(ev.minTs, ev.maxTs)}</span>

              <span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>Sim (Max)</span>
              <span style={{ color: 'var(--text-primary)', fontWeight: 700 }}>{ev.similarity != null ? ev.similarity.toFixed(3) : '—'}</span>

              <span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>Camera</span>
              <span style={{ color: 'var(--text-primary)', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{ev.camera || '—'}</span>

              {ev.rejected_reason && (
                <>
                  <span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>Reason</span>
                  <span style={{ color: 'var(--error)', fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{ev.rejected_reason}</span>
                </>
              )}
            </div>

            <div style={{ borderTop: '1px solid var(--border)', paddingTop: '10px', marginTop: '4px', opacity: 0.7 }}>
              <div style={{ fontSize: '0.625rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
                <span>{ev.all_images.length} detection{ev.all_images.length !== 1 ? 's' : ''}</span>
                <span style={{ fontWeight: 600 }}>ID: {ev.event_id.slice(-6)}</span>
              </div>
            </div>
          </div>
        ))}
        {filtered.length === 0 && !loading && (
          <div style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '80px', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-lg)', border: '2px dashed var(--border)', color: 'var(--text-muted)' }}>
            No groups found for the selected filter.
          </div>
        )}
      </div>

      {hasMore && (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '20px' }}>
          <button
            onClick={() => load(true)}
            className="secondary"
            disabled={loading}
            style={{ minWidth: '200px', fontWeight: 600 }}
          >
            {loading ? 'Loading...' : 'Load More Events'}
          </button>
        </div>
      )}
    </div>
  );
}
