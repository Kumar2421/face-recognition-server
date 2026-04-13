import { useEffect, useMemo, useState } from 'react';
import { crossCheckVisitorsVsEmployees, getApiBase, recognitionCameras, recognitionEvents, setRecognitionEventFeedback, subjectImages, type CrossCheckHit, type EventFeedbackLabel, type RecognitionEvent } from '../lib/api';

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

export default function Employees() {
  const [items, setItems] = useState<RecognitionEvent[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingMore, setLoadingMore] = useState<boolean>(false);
  const [camera, setCamera] = useState<string>('');
  const [cameraOptions, setCameraOptions] = useState<string[]>([]);
  const [branch, setBranch] = useState<string>('');
  const [subjectId, setSubjectId] = useState<string>('');
  const [nextCursor, setNextCursor] = useState<number | null>(null);
  const [pageSize] = useState<number>(1000);
  const [subjectImgById, setSubjectImgById] = useState<Record<string, string>>({});
  const [crossHitsByEmployee, setCrossHitsByEmployee] = useState<Record<string, CrossCheckHit[]>>({});
  const [selectedEmployee, setSelectedEmployee] = useState<{ sid: string, events: RecognitionEvent[], hits: CrossCheckHit[] } | null>(null);
  const [dateMode, setDateMode] = useState<'all' | 'day' | 'range'>('all');
  const [day, setDay] = useState<string>('');
  const [fromDay, setFromDay] = useState<string>('');
  const [toDay, setToDay] = useState<string>('');

  const [employeePage, setEmployeePage] = useState<number>(1);
  const employeesPerPage = 20;

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

  function getBranchFromSid(sid: string): string {
    const parts = sid.split('-');
    if (parts.length <= 2) return 'Main';
    // Skip 'employee', Skip 'tmj', take the rest but filter out numeric parts
    const branchParts = parts.slice(2).filter(p => !/^\d+$/.test(p));
    return branchParts.join('-') || 'Main';
  }

  async function load(reset: boolean = true) {
    setLoading(true);
    setErr(null);
    try {
      const cur = reset ? null : nextCursor;
      const resp = await recognitionEvents({
        decision: 'match',
        camera: camera || undefined,
        subject_id: subjectId || undefined,
        limit: pageSize,
        cursor: cur,
        day: dateMode === 'day' ? (day || null) : null,
        from_day: dateMode === 'range' ? (fromDay || null) : null,
        to_day: dateMode === 'range' ? (toDay || null) : null,
      });

      const emps = (resp.items || []).filter(it =>
        String(it.subject_id || '').toLowerCase().startsWith('employee-')
      );

      if (reset) {
        setItems(emps);
      } else {
        setItems(prev => [...(prev || []), ...emps]);
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
    if (loadingMore || nextCursor == null) return;
    setLoadingMore(true);
    setErr(null);
    try {
      const resp = await recognitionEvents({
        decision: 'match',
        camera: camera || undefined,
        subject_id: subjectId || undefined,
        limit: pageSize,
        cursor: nextCursor,
        day: dateMode === 'day' ? (day || null) : null,
        from_day: dateMode === 'range' ? (fromDay || null) : null,
        to_day: dateMode === 'range' ? (toDay || null) : null,
      });

      const emps = (resp.items || []).filter(it =>
        String(it.subject_id || '').toLowerCase().startsWith('employee-')
      );

      setItems(prev => [...(prev || []), ...emps]);
      setNextCursor(resp.cursor != null ? Number(resp.cursor) : null);
    } catch (e: any) {
      setErr(String(e));
    } finally {
      setLoadingMore(false);
    }
  }


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
    setEmployeePage(1);
    load(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camera, pageSize, dateMode, day, fromDay, toDay, subjectId]);

  // Derived filters and grouping
  const filtered = useMemo(() => {
    let out = items;
    if (subjectId) {
      out = out.filter(i => String(i.subject_id || '').toLowerCase().includes(subjectId.toLowerCase()));
    }
    if (branch) {
      out = out.filter(i => getBranchFromSid(String(i.subject_id || '')) === branch);
    }
    return out;
  }, [items, subjectId, branch]);

  const grouped = useMemo(() => {
    const map = new Map<string, RecognitionEvent[]>();

    // Add official sightings
    for (const it of filtered) {
      const sid = String(it.subject_id || '').trim();
      if (!sid) continue;
      const arr = map.get(sid) || [];
      arr.push(it);
      map.set(sid, arr);
    }

    return Array.from(map.entries()).sort((a, b) => a[0].localeCompare(b[0])).map(([s, evs]) => {
      // Sort each employee's events by time (newest first)
      const sortedEvs = evs.sort((e1, e2) => e2.ts - e1.ts);
      return [s, sortedEvs] as [string, RecognitionEvent[]];
    });
  }, [filtered, subjectId]);

  useEffect(() => {
    setEmployeePage(1);
  }, [branch, subjectId, grouped.length]);

  const totalEmployeePages = useMemo(() => {
    return Math.max(1, Math.ceil(grouped.length / employeesPerPage));
  }, [grouped.length]);

  const pagedGrouped = useMemo(() => {
    const p = Math.min(Math.max(1, employeePage), totalEmployeePages);
    const start = (p - 1) * employeesPerPage;
    const end = start + employeesPerPage;
    return grouped.slice(start, end);
  }, [grouped, employeePage, totalEmployeePages]);

  const branches = useMemo(() => {
    const s = new Set<string>();
    for (const it of items) {
      s.add(getBranchFromSid(String(it.subject_id || '')));
    }
    return Array.from(s).sort();
  }, [items]);

  useEffect(() => {
    let cancelled = false;
    async function prefetchRefs() {
      const want: string[] = [];
      for (const [sid] of grouped) {
        if (subjectImgById[sid]) continue;
        want.push(sid);
      }
      const uniq = Array.from(new Set(want)).slice(0, 50);
      if (!uniq.length) return;

      for (const sid of uniq) {
        if (cancelled) break;
        try {
          const resp = await subjectImages(sid, { limit: 1 });
          const first = resp?.items?.[0];
          if (first?.thumb_path && !cancelled) {
            setSubjectImgById(prev => ({ ...prev, [sid]: first.thumb_path! }));
          }
        } catch (e) { }
      }
    }
    prefetchRefs();
    return () => { cancelled = true; };
  }, [grouped, subjectImgById]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const resp = await crossCheckVisitorsVsEmployees({
          camera: camera || undefined,
          day: dateMode === 'day' ? (day || null) : null,
          from_day: dateMode === 'range' ? (fromDay || null) : null,
          to_day: dateMode === 'range' ? (toDay || null) : null,
          limit: 2000,
        });
        if (cancelled) return;
        const map: Record<string, CrossCheckHit[]> = {};
        for (const h of resp.items || []) {
          const emp = String(h.employee_subject_id || '').trim();
          if (!emp) continue;
          (map[emp] = map[emp] || []).push(h);
        }
        setCrossHitsByEmployee(map);
      } catch {
        if (cancelled) return;
        setCrossHitsByEmployee({});
      }
    })();
    return () => { cancelled = true; };
  }, [camera, dateMode, day, fromDay, toDay]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h2 style={{ fontSize: '1.875rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '4px' }}>Employee Activity</h2>
          <div style={{ color: 'var(--text-secondary)', fontSize: '0.9375rem' }}>
            Modular grid view. Grouped per employee. Fixed branch filtering.
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={() => {
              setNextCursor(null);
              setCamera('');
              setBranch('');
              setSubjectId('');
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

      {/* Filters */}
      <div className="card" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '20px', background: 'var(--bg-primary)' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Branch</span>
          <select value={branch} onChange={e => setBranch(e.target.value)} style={{ padding: '8px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}>
            <option value="">All Branches</option>
            {branches.map(b => (
              <option key={b} value={b}>{b}</option>
            ))}
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
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Employee ID Search</span>
          <input value={subjectId} onChange={e => setSubjectId(e.target.value)} placeholder="Search..." style={{ padding: '8px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }} />
        </div>
      </div>

      <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
        {loading
          ? 'Fetching records...'
          : `Showing ${pagedGrouped.length} of ${grouped.length} employees. Page ${employeePage} / ${totalEmployeePages}.`}
      </div>

      {err && (
        <div style={{ padding: '16px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--error)', borderRadius: 'var(--radius-md)', color: 'var(--error)', fontWeight: 500 }}>
          {err}
        </div>
      )}

      {/* Main Modular Grid: 5 columns on desktop */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))', gap: '20px' }}>
        {pagedGrouped.map(([sid, events]) => {
          const ref = subjectImgById[sid] || '';
          const lastEv = events[0];
          const hits = (crossHitsByEmployee[sid] || []).slice(0, 6);

          return (
            <div key={sid} className="card" style={{ display: 'flex', flexDirection: 'column', gap: '12px', padding: '12px', position: 'relative', minHeight: '340px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div style={{ maxWidth: '140px' }}>
                  <h3 style={{ fontSize: '0.875rem', fontWeight: 800, color: 'var(--text-primary)', wordBreak: 'break-all', lineHeight: 1.2 }}>{sid}</h3>
                  <div style={{ fontSize: '0.625rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700, marginTop: '2px' }}>
                    Branch: {getBranchFromSid(sid)}
                  </div>
                </div>
                <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                  <div style={{
                    padding: '2px 8px',
                    background: 'rgba(37, 99, 235, 0.1)',
                    color: 'var(--primary)',
                    borderRadius: '4px',
                    fontSize: '0.625rem',
                    fontWeight: 700,
                    border: '1px solid rgba(37, 99, 235, 0.2)'
                  }}>
                    {events.length} SIGHTINGS
                  </div>
                  <button
                    onClick={() => setSelectedEmployee({ sid, events, hits: crossHitsByEmployee[sid] || [] })}
                    style={{ padding: '6px', minWidth: 'auto', background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: '6px', cursor: 'pointer' }}
                    title="Expand View"
                  >
                    <span style={{ fontSize: '0.875rem' }}>⛶</span>
                  </button>
                </div>
              </div>

              <div style={{ height: '150px', display: 'flex', gap: '4px', borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border)' }}>
                <div style={{ flex: 1, position: 'relative', background: 'var(--bg-secondary)' }}>
                  {ref ? (
                    <img src={`${getApiBase()}${ref}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                  ) : (
                    <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.625rem', color: 'var(--text-muted)' }}>No Ref</div>
                  )}
                  <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, padding: '4px', background: 'rgba(0,0,0,0.5)', color: 'white', fontSize: '0.5625rem', textAlign: 'center', fontWeight: 700 }}>DATABASE</div>
                </div>
                <div style={{ flex: 1, position: 'relative', background: 'var(--bg-secondary)', borderLeft: '1px solid var(--border)' }}>
                  {lastEv.image_path ? (
                    <img src={`${getApiBase()}${lastEv.image_path}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                  ) : (
                    <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.625rem', color: 'var(--text-muted)' }}>No Live</div>
                  )}
                  <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, padding: '4px', background: 'var(--success)', color: 'white', fontSize: '0.5625rem', textAlign: 'center', fontWeight: 700 }}>LIVE MATCH</div>
                </div>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem' }}>
                  <div style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>Cross-Check Status</div>
                  <div style={{
                    fontWeight: 700,
                    color: lastEv.similarity && lastEv.similarity > 0.9 ? 'var(--success)' : 'var(--warning)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px'
                  }}>
                    {lastEv.similarity && lastEv.similarity > 0.9 ? '✓ Verified' : '⚠ Caution'}
                    <span style={{ fontSize: '0.6875rem', opacity: 0.8 }}>({lastEv.similarity ? (lastEv.similarity * 100).toFixed(1) : '--'}%)</span>
                  </div>
                </div>

                <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
                  Latest: {lastEv.camera} ({fmtSavedAt(lastEv).split(',')[1]})
                  <div style={{ fontSize: '0.625rem', opacity: 0.8, marginTop: '2px', fontFamily: 'monospace' }}>
                    {`id:${lastEv.event_id.slice(0, 8)}...`}
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '4px', overflowX: 'auto', paddingBottom: '4px' }}>
                  {hits.map(h => {
                    const img = String(h.visitor_image_path || h.visitor_thumb_path || '').trim();
                    const vid = String(h.visitor_subject_id || '').trim();
                    return (
                      <div key={h.visitor_event_id} style={{
                        width: '40px',
                        height: '40px',
                        borderRadius: '4px',
                        overflow: 'hidden',
                        flexShrink: 0,
                        border: '2px solid var(--warning)',
                        position: 'relative'
                      }}>
                        {img ? (
                          <img src={`${getApiBase()}${img}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} title={`${vid} sim=${Number(h.similarity).toFixed(4)}`} />
                        ) : (
                          <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.625rem', color: 'var(--text-muted)' }}>No Img</div>
                        )}
                        {vid && (
                          <div style={{ position: 'absolute', top: 0, right: 0, background: 'var(--warning)', color: 'black', fontSize: '8px', padding: '0 2px', fontWeight: 900, borderRadius: '0 0 0 4px' }}>V</div>
                        )}
                      </div>
                    );
                  })}
                  {hits.length > 6 && (
                    <div
                      onClick={() => setSelectedEmployee({ sid, events, hits: crossHitsByEmployee[sid] || [] })}
                      style={{ width: '40px', height: '40px', borderRadius: '4px', background: 'var(--bg-secondary)', border: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.625rem', fontWeight: 800, color: 'var(--text-muted)', cursor: 'pointer' }}
                    >
                      +{hits.length - 6}
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Pagination */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: '12px', alignItems: 'center' }}>
        <button
          onClick={() => setEmployeePage(p => Math.max(1, p - 1))}
          disabled={employeePage <= 1}
          style={{ padding: '8px 14px' }}
        >
          Prev
        </button>
        <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', fontWeight: 600 }}>
          Page {employeePage} / {totalEmployeePages}
        </div>
        <button
          onClick={() => setEmployeePage(p => Math.min(totalEmployeePages, p + 1))}
          disabled={employeePage >= totalEmployeePages}
          style={{ padding: '8px 14px' }}
        >
          Next
        </button>
      </div>

      <footer style={{ display: 'flex', justifyContent: 'center', padding: '16px 0' }}>
        <button
          disabled={nextCursor == null || loadingMore}
          onClick={loadMore}
          className={nextCursor != null ? 'primary' : ''}
          style={{ padding: '12px 48px', minWidth: '200px' }}
        >
          {loadingMore ? 'Loading...' : (nextCursor == null ? 'No More Results' : 'Load More Employees')}
        </button>
      </footer>

      {/* Full View Modal */}
      {selectedEmployee && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000, padding: '40px' }}>
          <div className="card" style={{ width: '100%', maxWidth: '1000px', maxHeight: '90vh', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '24px', padding: '32px', position: 'relative' }}>
            <button
              onClick={() => setSelectedEmployee(null)}
              style={{ position: 'absolute', top: 16, right: 16, border: 'none', background: 'transparent', cursor: 'pointer', fontSize: '1.5rem', color: 'var(--text-muted)' }}
            >
              ✕
            </button>

            <div style={{ display: 'flex', gap: '24px', alignItems: 'center' }}>
              <div style={{ width: '120px', height: '120px', borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '2px solid var(--primary)', background: 'var(--bg-secondary)' }}>
                {subjectImgById[selectedEmployee.sid] && <img src={`${getApiBase()}${subjectImgById[selectedEmployee.sid]}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />}
              </div>
              <div>
                <h2 style={{ fontSize: '1.5rem', fontWeight: 800 }}>{selectedEmployee.sid}</h2>
                <div style={{ color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', fontSize: '0.875rem' }}>
                  Branch: {getBranchFromSid(selectedEmployee.sid)}
                </div>
                <div style={{ marginTop: '8px', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                  Showing {selectedEmployee.events.length} employee records and {(selectedEmployee.hits || []).length} visitor cross-check hits.
                </div>
              </div>
            </div>

            {(selectedEmployee.hits || []).length > 0 && (
              <div>
                <div style={{ fontSize: '0.875rem', fontWeight: 800, marginBottom: '12px', color: 'var(--warning)' }}>Visitor Cross-Check Hits</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '20px' }}>
                  {(selectedEmployee.hits || []).map(h => {
                    const img = String(h.visitor_image_path || h.visitor_thumb_path || '').trim();
                    const vid = String(h.visitor_subject_id || '').trim();
                    return (
                      <div key={h.visitor_event_id} style={{ borderRadius: 'var(--radius-md)', overflow: 'hidden', background: 'var(--bg-secondary)', border: '2px solid var(--warning)' }}>
                        <div style={{ height: '160px', position: 'relative' }}>
                          {img ? (
                            <img src={`${getApiBase()}${img}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} title={`${vid}`} />
                          ) : (
                            <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem', color: 'var(--text-muted)' }}>No Img</div>
                          )}
                          <div style={{ position: 'absolute', top: 8, right: 8, padding: '4px 8px', background: 'rgba(0,0,0,0.6)', color: 'white', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 700 }}>
                            {(Number(h.similarity) * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                          <div style={{ fontSize: '0.8125rem', fontWeight: 700 }}>{h.visitor_camera || 'Unknown Camera'}</div>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{h.visitor_ts != null ? fmtTs(Number(h.visitor_ts)) : '--'}</div>
                          <div style={{ fontSize: '0.625rem', color: 'var(--warning)', fontFamily: 'monospace', opacity: 0.9, marginTop: '2px', fontWeight: 800 }}>
                            Visitor ID: {vid || '--'}
                          </div>
                          <div style={{ fontSize: '0.625rem', color: 'var(--text-muted)', fontFamily: 'monospace', opacity: 0.8 }}>
                            Event: {String(h.visitor_event_id || '')}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            <div style={{ fontSize: '0.875rem', fontWeight: 800, marginBottom: '-8px' }}>Employee Activity</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '20px' }}>
              {selectedEmployee.events.map(ev => (
                <div key={ev.event_id} style={{ borderRadius: 'var(--radius-md)', overflow: 'hidden', background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
                  <div style={{ height: '160px', position: 'relative' }}>
                    <img src={`${getApiBase()}${ev.image_path}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                    <div style={{ position: 'absolute', top: 8, right: 8, padding: '4px 8px', background: 'rgba(0,0,0,0.6)', color: 'white', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 700 }}>
                      {ev.similarity != null ? (ev.similarity * 100).toFixed(1) : '--'}%
                    </div>
                  </div>
                  <div style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <div style={{ fontSize: '0.8125rem', fontWeight: 700 }}>{ev.camera}</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{fmtSavedAt(ev)}</div>
                    <div style={{ fontSize: '0.625rem', color: 'var(--text-muted)', fontFamily: 'monospace', opacity: 0.8, marginTop: '2px' }}>
                      ID: {ev.event_id}
                    </div>
                    <div style={{ display: 'flex', gap: '4px', marginTop: '8px' }}>
                      <button onClick={() => setFeedback(ev.event_id, 'tp')} style={{ flex: 1, padding: '4px', fontSize: '0.625rem', background: ev.feedback_label === 'tp' ? 'var(--success)' : 'transparent', color: ev.feedback_label === 'tp' ? 'white' : 'inherit' }}>TP</button>
                      <button onClick={() => setFeedback(ev.event_id, 'fp')} style={{ flex: 1, padding: '4px', fontSize: '0.625rem', background: ev.feedback_label === 'fp' ? 'var(--error)' : 'transparent', color: ev.feedback_label === 'fp' ? 'white' : 'inherit' }}>FP</button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
