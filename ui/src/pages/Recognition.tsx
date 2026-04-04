import { useEffect, useMemo, useState } from 'react';
import { getApiBase, recognitionCameras, recognitionEvents, setRecognitionEventFeedback, subjectImages, type EventFeedbackLabel, type RecognitionEvent } from '../lib/api';

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

function fmtMs(v: any): string {
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  return String(Math.round(n));
}

function getTiming(ev: any): any {
  try {
    return ev?.meta?.timing || null;
  } catch {
    return null;
  }
}

type DecisionFilter = '' | 'match' | 'no_match' | 'rejected';

export default function Recognition() {
  const [items, setItems] = useState<RecognitionEvent[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingMore, setLoadingMore] = useState<boolean>(false);
  const [camera, setCamera] = useState<string>('');
  const [cameraOptions, setCameraOptions] = useState<string[]>([]);
  const [decision, setDecision] = useState<DecisionFilter>('');
  const [subjectId, setSubjectId] = useState<string>('');
  const [minSim, setMinSim] = useState<string>('');
  const [maxSim, setMaxSim] = useState<string>('');
  const [nextCursor, setNextCursor] = useState<number | null>(null);
  const [pageSize, setPageSize] = useState<number>(200);
  const [matchRefPath, setMatchRefPath] = useState<string | null>(null);
  const [subjectImgById, setSubjectImgById] = useState<Record<string, string>>({});
  const [dateMode, setDateMode] = useState<'all' | 'day' | 'range'>('all');
  const [day, setDay] = useState<string>('');
  const [fromDay, setFromDay] = useState<string>('');
  const [toDay, setToDay] = useState<string>('');

  async function setFeedback(evId: string, label: EventFeedbackLabel) {
    const eventId = String(evId || '').trim();
    if (!eventId) return;
    try {
      const lab = String(label || '').trim() as EventFeedbackLabel;
      const payload = { label: lab ? lab : null };
      const resp = await setRecognitionEventFeedback(eventId, payload);
      setItems(prev =>
        (prev || []).map(it =>
          it.event_id === eventId
            ? {
              ...it,
              feedback_label: resp.feedback_label ?? payload.label ?? null,
              feedback_note: resp.feedback_note ?? null,
              feedback_updated_at: resp.feedback_updated_at ?? null,
            }
            : it
        )
      );
    } catch (e: any) {
      setErr(String(e));
    }
  }

  const minSimNum = useMemo(() => {
    const v = Number(minSim);
    return Number.isFinite(v) && minSim.trim() !== '' ? v : null;
  }, [minSim]);

  const maxSimNum = useMemo(() => {
    const v = Number(maxSim);
    return Number.isFinite(v) && maxSim.trim() !== '' ? v : null;
  }, [maxSim]);

  const cameras = useMemo(() => {
    const s = new Set<string>();
    for (const c of cameraOptions || []) {
      const v = String(c || '').trim();
      if (v) s.add(v);
    }
    for (const it of items) {
      const v = String(it.camera || '').trim();
      if (v) s.add(v);
    }
    return Array.from(s).sort();
  }, [cameraOptions, items]);

  async function load(reset: boolean = true) {
    setLoading(true);
    setErr(null);
    try {
      const cur = reset ? null : nextCursor;
      const resp = await recognitionEvents({
        decision: decision || undefined,
        camera: camera || undefined,
        subject_id: subjectId || undefined,
        min_similarity: minSimNum != null ? minSimNum : undefined,
        max_similarity: maxSimNum != null ? maxSimNum : undefined,
        limit: pageSize,
        cursor: cur,
        day: dateMode === 'day' ? (day || null) : null,
        from_day: dateMode === 'range' ? (fromDay || null) : null,
        to_day: dateMode === 'range' ? (toDay || null) : null,
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
        min_similarity: minSimNum != null ? minSimNum : undefined,
        max_similarity: maxSimNum != null ? maxSimNum : undefined,
        limit: pageSize,
        cursor: nextCursor,
        day: dateMode === 'day' ? (day || null) : null,
        from_day: dateMode === 'range' ? (fromDay || null) : null,
        to_day: dateMode === 'range' ? (toDay || null) : null,
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
    let cancelled = false;
    (async () => {
      try {
        const resp = await recognitionCameras({ limit: 50000 });
        if (cancelled) return;
        setCameraOptions((resp as any) || []);
      } catch { /* ignore */ }
    })();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    setNextCursor(null);
    load(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [decision, camera, subjectId, pageSize, dateMode, day, fromDay, toDay]);

  useEffect(() => {
    setNextCursor(null);
    load(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [minSim, maxSim]);

  const filtered = useMemo(() => {
    let out = items;
    if (decision) out = out.filter(i => String(i.decision || '') === decision);
    if (camera) out = out.filter(i => String(i.camera || '') === camera);
    if (subjectId) out = out.filter(i => String(i.subject_id || '') === subjectId);
    return out;
  }, [items, decision, camera, subjectId]);

  const matchRefSubjectId = useMemo(() => {
    if (decision !== 'match') return null;
    const sid = String(subjectId || '').trim();
    return sid || null;
  }, [decision, subjectId]);

  useEffect(() => {
    let cancelled = false;
    async function prefetchRefsForMatchCards() {
      const group = String(subjectId || '').trim();
      if (decision !== 'match' || group) return;

      const want: string[] = [];
      for (const ev of filtered) {
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
        } catch { /* ignore */ }
      }
    }
    prefetchRefsForMatchCards();
    return () => { cancelled = true; };
  }, [decision, subjectId, filtered, subjectImgById]);

  useEffect(() => {
    let cancelled = false;
    async function loadMatchReference() {
      if (decision !== 'match') {
        setMatchRefPath(null);
        return;
      }
      const sid = String(matchRefSubjectId || '').trim();
      if (!sid) {
        setMatchRefPath(null);
        return;
      }
      try {
        const resp = await subjectImages(sid, { limit: 1 });
        const first = resp?.items?.[0];
        const p = (first?.thumb_path || first?.image_path || '').trim();
        if (cancelled) return;
        setMatchRefPath(p || null);
      } catch {
        if (cancelled) return;
        setMatchRefPath(null);
      }
    }
    loadMatchReference();
    return () => { cancelled = true; };
  }, [decision, matchRefSubjectId]);

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
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h2 style={{ fontSize: '1.875rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '4px' }}>Recognition Events</h2>
          <div style={{ color: 'var(--text-secondary)', fontSize: '0.9375rem' }}>
            Explore full recognition history with advanced filters.
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={() => {
              setNextCursor(null);
              setCamera('');
              setDecision('');
              setSubjectId('');
              setMinSim('');
              setMaxSim('');
              setDateMode('all');
              load(true);
            }}
            style={{ fontWeight: 600 }}
          >
            Reset
          </button>
          <button onClick={() => load(true)} className="primary" style={{ fontWeight: 600 }}>
            {loading ? 'Refreshing...' : 'Refresh Feed'}
          </button>
        </div>
      </header>

      <div className="card" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', background: 'var(--bg-primary)' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Decision</span>
          <select value={decision} onChange={e => setDecision(e.target.value as DecisionFilter)} style={{ padding: '8px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}>
            <option value="">All Decisions</option>
            <option value="match">Match</option>
            <option value="no_match">No Match</option>
            <option value="rejected">Rejected</option>
          </select>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Camera</span>
          <select value={camera} onChange={e => setCamera(e.target.value)} style={{ padding: '8px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}>
            <option value="">All Cameras</option>
            {cameras.map(c => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Date Mode</span>
          <select value={dateMode} onChange={(e) => setDateMode(e.target.value as any)} style={{ padding: '8px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}>
            <option value="all">All Dates</option>
            <option value="day">Single Day</option>
            <option value="range">Range</option>
          </select>
        </div>

        {dateMode !== 'all' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              {dateMode === 'day' ? 'Select Date' : 'Select Range'}
            </span>
            <div style={{ display: 'flex', gap: '4px' }}>
              {dateMode === 'day' && (
                <input type="date" value={day} onChange={(e) => setDay(e.target.value)} style={{ width: '100%', padding: '8px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }} />
              )}
              {dateMode === 'range' && (
                <>
                  <input type="date" value={fromDay} onChange={(e) => setFromDay(e.target.value)} style={{ width: '100%', padding: '8px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }} />
                  <input type="date" value={toDay} onChange={(e) => setToDay(e.target.value)} style={{ width: '100%', padding: '8px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }} />
                </>
              )}
            </div>
          </div>
        )}

        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Subject ID</span>
          <input value={subjectId} onChange={e => setSubjectId(e.target.value)} placeholder="Search..." style={{ padding: '8px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }} />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Similarity Control</span>
          <div style={{ display: 'flex', gap: '4px' }}>
            <input value={minSim} onChange={e => setMinSim(e.target.value)} placeholder="Min" style={{ width: '100%', padding: '8px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', fontSize: '0.8125rem' }} />
            <input value={maxSim} onChange={e => setMaxSim(e.target.value)} placeholder="Max" style={{ width: '100%', padding: '8px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', fontSize: '0.8125rem' }} />
          </div>
        </div>
      </div>

      <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
        {loading ? 'Fetching records...' : `Showing ${counts.total} events ( ${counts.match} matches / ${counts.noMatch} unknown / ${counts.rejected} rejected )`}
      </div>

      {err && (
        <div style={{ padding: '16px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--error)', borderRadius: 'var(--radius-md)', color: 'var(--error)', fontWeight: 500 }}>
          {err}
        </div>
      )}

      {decision === 'match' && String(subjectId || '').trim() !== '' && (
        <div className="card" style={{ maxWidth: '400px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h4 style={{ fontSize: '0.875rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Reference Image</h4>
            <span style={{ background: 'var(--bg-secondary)', padding: '2px 8px', borderRadius: '99px', fontSize: '0.75rem', fontWeight: 600 }}>{String(subjectId || '').trim()}</span>
          </div>
          {matchRefPath ? (
            <img
              src={`${getApiBase()}${matchRefPath}`}
              style={{ width: '100%', height: '200px', objectFit: 'contain', borderRadius: 'var(--radius-md)', background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}
            />
          ) : (
            <div style={{ height: '200px', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)', border: '2px dashed var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
              No reference available
            </div>
          )}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))', gap: '20px' }}>
        {items.map(ev => (
          <div key={ev.event_id} className="card" style={{ display: 'flex', flexDirection: 'column', gap: '10px', padding: '10px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div style={{ overflow: 'hidden' }}>
                <h4 style={{ fontWeight: 700, fontSize: '0.875rem', color: 'var(--text-primary)', whiteSpace: 'nowrap', textOverflow: 'ellipsis' }}>{ev.camera || 'Unknown'}</h4>
                <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', fontWeight: 500 }}>{fmtSavedAt(ev)}</div>
              </div>
              <div style={{
                padding: '1px 8px',
                borderRadius: '99px',
                fontSize: '0.625rem',
                fontWeight: 700,
                textTransform: 'uppercase',
                background: ev.decision === 'match' ? 'rgba(16, 185, 129, 0.1)' : ev.decision === 'no_match' ? 'rgba(107, 114, 128, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                color: ev.decision === 'match' ? 'var(--success)' : ev.decision === 'no_match' ? 'var(--text-secondary)' : 'var(--error)',
                border: `1px solid ${ev.decision === 'match' ? 'rgba(16, 185, 129, 0.2)' : ev.decision === 'no_match' ? 'var(--border)' : 'rgba(239, 68, 68, 0.2)'}`
              }}>
                {ev.decision}
              </div>
            </div>

            <div style={{ height: '150px', borderRadius: 'var(--radius-md)', overflow: 'hidden', background: 'var(--bg-secondary)', display: 'flex', gap: '2px' }}>
              {(() => {
                const group = String(subjectId || '').trim();
                const isMatch = String(ev.decision || '') === 'match';
                const sid = String(ev.subject_id || '').trim();
                const ref = !group && isMatch && sid ? (subjectImgById[sid] || '') : '';

                if (ref) {
                  return (
                    <>
                      <div style={{ flex: 1, position: 'relative' }}>
                        <img src={`${getApiBase()}${ref}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                        <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, padding: '4px', background: 'rgba(0,0,0,0.5)', color: 'white', fontSize: '0.625rem', textAlign: 'center' }}>REF</div>
                      </div>
                      <div style={{ flex: 1, position: 'relative' }}>
                        {ev.image_path ? (
                          <img src={`${getApiBase()}${ev.image_path}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                        ) : (
                          <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem', color: 'var(--text-muted)' }}>NO IMG</div>
                        )}
                        <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, padding: '4px', background: 'rgba(0,0,0,0.5)', color: 'white', fontSize: '0.625rem', textAlign: 'center' }}>LIVE</div>
                      </div>
                    </>
                  );
                }
                return ev.image_path ? (
                  <img src={`${getApiBase()}${ev.image_path}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                ) : (
                  <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.875rem' }}>No Image</div>
                );
              })()}
            </div>

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', paddingBottom: '8px', borderBottom: '1px solid var(--border)' }}>
              <button
                onClick={() => setFeedback(ev.event_id, 'tp')}
                style={{ flex: 1, fontSize: '0.625rem', padding: '2px', background: String(ev.feedback_label || '') === 'tp' ? 'var(--success)' : 'transparent', color: String(ev.feedback_label || '') === 'tp' ? 'white' : 'var(--text-secondary)' }}
              >TP</button>
              <button
                onClick={() => setFeedback(ev.event_id, 'fp')}
                style={{ flex: 1, fontSize: '0.625rem', padding: '2px', background: String(ev.feedback_label || '') === 'fp' ? 'var(--error)' : 'transparent', color: String(ev.feedback_label || '') === 'fp' ? 'white' : 'var(--text-secondary)' }}
              >FP</button>
              <button onClick={() => setFeedback(ev.event_id, '')} style={{ flex: 0.4, fontSize: '0.625rem', padding: '2px' }}>✕</button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '70px 1fr', gap: '4px', fontSize: '0.75rem' }}>
              <span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>Subject</span>
              <span style={{ fontWeight: 600, color: 'var(--text-primary)', wordBreak: 'break-all', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={ev.subject_id || undefined}>{ev.subject_id || '—'}</span>

              <span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>Sim</span>
              <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{ev.similarity != null ? ev.similarity.toFixed(4) : '—'}</span>
            </div>

            <details style={{ cursor: 'pointer' }}>
              <summary style={{ fontSize: '0.625rem', color: 'var(--primary)', fontWeight: 600 }}>Performance Info</summary>
              <div style={{ marginTop: '4px', padding: '6px', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', fontSize: '0.625rem', color: 'var(--text-secondary)' }}>
                Processing: {ev.processing_ms || '—'}ms
              </div>
            </details>
          </div>
        ))}
      </div>

      <footer style={{ display: 'flex', justifyContent: 'center', padding: '16px' }}>
        <button
          disabled={nextCursor == null || loadingMore}
          onClick={loadMore}
          className={nextCursor != null ? 'primary' : ''}
          style={{ padding: '12px 48px', minWidth: '200px' }}
        >
          {loadingMore ? 'Loading...' : (nextCursor == null ? 'No More Records' : 'Load More Results')}
        </button>
      </footer>
    </div>
  );
}
