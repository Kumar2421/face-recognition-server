import { useEffect, useMemo, useState } from 'react';
import { getApiBase, recognitionCameras, recognitionEvents, recognitionStats, setRecognitionEventFeedback, subjectImages, type EventFeedbackLabel, type RecognitionEvent, type RecognitionStatsResponse } from '../lib/api';
import StatCard from '../components/StatCard';

function fmtTs(ts: number): string {
  try {
    return new Intl.DateTimeFormat('en-IN', {
      timeZone: 'Asia/Kolkata',
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      second: '2-digit',
      hour12: true
    }).format(new Date(ts * 1000));
  } catch {
    return String(ts);
  }
}

function fmtTime(ts: number): string {
  try {
    return new Intl.DateTimeFormat('en-IN', {
      timeZone: 'Asia/Kolkata',
      hour: 'numeric',
      minute: '2-digit',
      second: '2-digit',
      hour12: true
    }).format(new Date(ts * 1000));
  } catch { return '--'; }
}

function fmtDate(ts: number): string {
  try {
    return new Intl.DateTimeFormat('en-IN', {
      timeZone: 'Asia/Kolkata',
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    }).format(new Date(ts * 1000));
  } catch { return '--'; }
}

function fmtDuration(seconds: number): string {
  if (seconds <= 0) return '0m';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

type DecisionFilter = '' | 'match' | 'no_match' | 'rejected';

export default function Recognition() {
  const [items, setItems] = useState<RecognitionEvent[]>([]);
  const [stats, setStats] = useState<RecognitionStatsResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingStats, setLoadingStats] = useState<boolean>(false);
  const [loadingMore, setLoadingMore] = useState<boolean>(false);
  const [camera, setCamera] = useState<string>('');
  const [cameraOptions, setCameraOptions] = useState<string[]>([]);
  const [decision, setDecision] = useState<DecisionFilter>('');
  const [subjectId, setSubjectId] = useState<string>('');
  const [minSim, setMinSim] = useState<string>('');
  const [maxSim, setMaxSim] = useState<string>('');
  const [nextCursor, setNextCursor] = useState<number | null>(null);
  const pageSize = 100;

  const [uniqueCount, setUniqueCount] = useState<number | null>(null);
  const [uniqueCountLoading, setUniqueCountLoading] = useState<boolean>(false);
  const [selectedEvent, setSelectedEvent] = useState<{ sid: string, events: RecognitionEvent[] } | null>(null);
  const [matchRefPath, setMatchRefPath] = useState<string | null>(null);
  const [subjectImgById, setSubjectImgById] = useState<Record<string, string>>({});
  const [dateMode, setDateMode] = useState<'all' | 'day' | 'range'>('all');
  const [day, setDay] = useState<string>(new Date().toLocaleDateString('en-CA'));
  const [fromDay, setFromDay] = useState<string>(new Date().toLocaleDateString('en-CA'));
  const [toDay, setToDay] = useState<string>(new Date().toLocaleDateString('en-CA'));

  useEffect(() => {
    // Attempt to find the most recent date from items if dateMode is 'all'
    if (dateMode === 'all' && items.length > 0) {
      // Use IST for date extraction to match the display
      const validTss = items.map(it => Number(it.ts || 0)).filter(t => t > 0);
      if (validTss.length > 0) {
        const latestTs = Math.max(...validTss);
        const latestDate = new Intl.DateTimeFormat('en-CA', {
          timeZone: 'Asia/Kolkata',
          year: 'numeric',
          month: '2-digit',
          day: '2-digit'
        }).format(new Date(latestTs * 1000));

        setDay(latestDate);
        setFromDay(latestDate);
        setToDay(latestDate);
      }
    }
  }, [items, dateMode]);

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

  useEffect(() => {
    let cancelled = false;
    async function loadUniqueCount() {
      setUniqueCountLoading(true);
      try {
        const resp = await recognitionStats({
          day: dateMode === 'day' ? (day || null) : null,
          from_day: dateMode === 'range' ? (fromDay || null) : null,
          to_day: dateMode === 'range' ? (toDay || null) : null,
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
  }, [day, fromDay, toDay, dateMode, camera]);

  async function loadStats() {
    setLoadingStats(true);
    try {
      const s = await recognitionStats({
        day: dateMode === 'day' ? (day || null) : null,
        from_day: dateMode === 'range' ? (fromDay || null) : null,
        to_day: dateMode === 'range' ? (toDay || null) : null,
        camera: camera || undefined,
      });
      setStats(s);
    } catch (e) {
      console.error('Failed to load recognition stats:', e);
    } finally {
      setLoadingStats(false);
    }
  }

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
  }, [decision, camera, subjectId, dateMode, day, fromDay, toDay]);

  useEffect(() => {
    loadStats();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camera, dateMode, day, fromDay, toDay]);

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

  const displayRows = useMemo(() => {
    const groupMatches = decision === '' || decision === 'match';
    if (groupMatches) {
      const map = new Map<string, RecognitionEvent[]>();
      const singles: { type: 'single', key: string, event: RecognitionEvent }[] = [];

      for (const it of filtered) {
        const d = String(it.decision || '').trim();
        if (d !== 'match') {
          singles.push({ type: 'single', key: it.event_id, event: it });
          continue;
        }

        const sid = String(it.subject_id || '').trim();
        if (!sid) {
          singles.push({ type: 'single', key: it.event_id, event: it });
          continue;
        }

        const dateKey = new Date(Number(it.ts || 0) * 1000).toLocaleDateString('en-CA');
        const key = `${sid}:${dateKey}`;
        const arr = map.get(key) || [];
        arr.push(it);
        map.set(key, arr);
      }

      const groups = Array.from(map.entries()).map(([key, evs]) => {
        const sorted = evs.sort((e1, e2) => e1.ts - e2.ts); // Chronological order
        const sid = key.split(':')[0];
        const duration = sorted[sorted.length - 1].ts - sorted[0].ts;
        return { type: 'grouped' as const, key, sid, events: sorted, duration };
      });

      const all = [...groups, ...singles.map(s => ({ ...s, duration: 0 }))];
      all.sort((a, b) => {
        // Primary sort: Duration (highest first)
        const durA = (a as any).duration || 0;
        const durB = (b as any).duration || 0;
        if (durB !== durA) return durB - durA;

        // Secondary sort: Timestamp (newest first)
        const ta = a.type === 'grouped' ? (a.events?.[0]?.ts ?? 0) : (a.event?.ts ?? 0);
        const tb = b.type === 'grouped' ? (b.events?.[0]?.ts ?? 0) : (b.event?.ts ?? 0);
        return Number(tb) - Number(ta);
      });
      return all;
    } else {
      return filtered.map(ev => ({ type: 'single' as const, key: ev.event_id, event: ev }));
    }
  }, [filtered, decision]);

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
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <div className="card" style={{ padding: '8px 16px', display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: '140px', background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)' }}>
            <span style={{ fontSize: '0.65rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Unique Count</span>
            <span style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--primary)' }}>
              {uniqueCountLoading ? '...' : (uniqueCount ?? 0)}
            </span>
          </div>
          <button
            onClick={() => {
              const today = new Date().toLocaleDateString('en-CA');
              setNextCursor(null);
              setCamera('');
              setDecision('');
              setSubjectId('');
              setMinSim('');
              setMaxSim('');
              setDateMode('all');
              setDay(today);
              setFromDay(today);
              setToDay(today);
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

      <section>
        <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '20px' }}>
          <StatCard
            title="Total Events"
            value={loadingStats ? '…' : (stats?.total.toLocaleString() || '0')}
            hint="All detected faces"
          />
          <StatCard
            title="Total Match"
            value={loadingStats ? '…' : (stats?.match.toLocaleString() || '0')}
            tone="good"
            hint="Successfully recognized"
          />
          <StatCard
            title="Total No Match"
            value={loadingStats ? '…' : (stats?.no_match.toLocaleString() || '0')}
            tone="warn"
            hint="Unknown subjects"
          />
          <StatCard
            title="Total Rejections"
            value={loadingStats ? '…' : (stats?.rejection.toLocaleString() || '0')}
            tone="bad"
            hint="Failed quality check"
          />
        </div>
      </section>

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

      {/* Table Header */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '40px 100px 1.2fr 180px 1fr 150px 150px 80px 60px',
        gap: '12px',
        padding: '12px 20px',
        background: 'var(--bg-secondary)',
        borderRadius: 'var(--radius-md)',
        fontWeight: 700,
        fontSize: '0.75rem',
        color: 'var(--text-muted)',
        textTransform: 'uppercase',
        border: '1px solid var(--border)',
        alignItems: 'center'
      }}>
        <span>#</span>
        <span>Status</span>
        <span>Subject ID</span>
        <span>Customer Images</span>
        <span>Camera Name</span>
        <span>Entry Time</span>
        <span>Exit Time</span>
        <span>Duration</span>
        <span style={{ textAlign: 'center' }}>View</span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {displayRows.map((row, idx) => {
          const isGrouped = row.type === 'grouped';
          const events = isGrouped ? row.events : [row.event];
          const firstEv = events[0];
          const lastEv = events[events.length - 1];
          const sid = isGrouped ? row.sid : (lastEv.subject_id || '—');
          const ref = subjectImgById[sid] || '';
          const duration = lastEv.ts - firstEv.ts;

          const cameras = Array.from(new Set(events.map(e => e.camera))).join(', ');

          return (
            <div
              key={row.key}
              className="card"
              style={{
                display: 'grid',
                gridTemplateColumns: '40px 100px 1.2fr 180px 1fr 150px 150px 80px 60px',
                gap: '12px',
                padding: '12px 20px',
                alignItems: 'center',
                transition: 'transform 0.2s, box-shadow 0.2s',
                cursor: 'default'
              }}
            >
              <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', fontWeight: 600 }}>{idx + 1}</div>

              {/* Status */}
              <div style={{
                padding: '1px 8px',
                borderRadius: '4px',
                fontSize: '0.625rem',
                fontWeight: 700,
                textTransform: 'uppercase',
                textAlign: 'center',
                background: lastEv.decision === 'match' ? 'rgba(16, 185, 129, 0.1)' : lastEv.decision === 'no_match' ? 'rgba(107, 114, 128, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                color: lastEv.decision === 'match' ? 'var(--success)' : lastEv.decision === 'no_match' ? 'var(--text-secondary)' : 'var(--error)',
                border: `1px solid ${lastEv.decision === 'match' ? 'rgba(16, 185, 129, 0.2)' : lastEv.decision === 'no_match' ? 'var(--border)' : 'rgba(239, 68, 68, 0.2)'}`
              }}>
                {lastEv.decision}
              </div>

              {/* Subject ID */}
              <div style={{ fontWeight: 700, fontSize: '0.875rem', color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {sid}
              </div>

              {/* Customer Images */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <div style={{ display: 'flex', gap: '4px', height: '60px' }}>
                  <div style={{ flex: 1, borderRadius: '4px', overflow: 'hidden', border: '1px solid var(--border)', background: 'var(--bg-secondary)', position: 'relative' }}>
                    {ref ? (
                      <img src={`${getApiBase()}${ref}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                    ) : (
                      <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.5rem' }}>REF</div>
                    )}
                  </div>
                  <div style={{ flex: 1, borderRadius: '4px', overflow: 'hidden', border: '1px solid var(--border)', background: 'var(--bg-secondary)', position: 'relative' }}>
                    {lastEv.image_path ? (
                      <img src={`${getApiBase()}${lastEv.image_path}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                    ) : (
                      <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.5rem' }}>LIVE</div>
                    )}
                  </div>
                </div>
                {/* Similarity Bar */}
                {lastEv.decision === 'match' && (
                  <div style={{ height: '14px', background: 'var(--bg-secondary)', borderRadius: '2px', overflow: 'hidden', border: '1px solid var(--border)', position: 'relative' }}>
                    <div style={{
                      width: `${(lastEv.similarity || 0) * 100}%`,
                      height: '100%',
                      background: 'var(--success)',
                      opacity: 0.8
                    }} />
                    <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '9px', fontWeight: 800, color: (lastEv.similarity || 0) > 0.5 ? 'white' : 'var(--text-primary)' }}>
                      {((lastEv.similarity || 0) * 100).toFixed(0)}%
                    </div>
                  </div>
                )}
              </div>

              {/* Camera Name */}
              <div style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)', fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={cameras}>
                {cameras}
              </div>

              {/* Entry Time */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                <div style={{ fontSize: '0.8125rem', fontWeight: 600 }}>{fmtDate(firstEv.ts)}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{fmtTime(firstEv.ts)}</div>
              </div>

              {/* Exit Time */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                <div style={{ fontSize: '0.8125rem', fontWeight: 600 }}>{fmtDate(lastEv.ts)}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{fmtTime(lastEv.ts)}</div>
              </div>

              {/* Duration */}
              <div style={{ fontSize: '0.875rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                {lastEv.decision === 'match' ? fmtDuration(duration) : '—'}
              </div>

              {/* Action */}
              <div style={{ display: 'flex', justifyContent: 'center' }}>
                <button
                  onClick={() => setSelectedEvent({ sid, events })}
                  style={{ padding: '6px', minWidth: 'auto', background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: '6px', cursor: 'pointer' }}
                  title="Expand View"
                >
                  <span style={{ fontSize: '1rem' }}>⛶</span>
                </button>
              </div>
            </div>
          );
        })}
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

      {/* Full View Modal */}
      {selectedEvent && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000, padding: '40px' }}>
          <div className="card" style={{ width: '100%', maxWidth: '1000px', maxHeight: '90vh', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '24px', padding: '32px', position: 'relative' }}>
            <button
              onClick={() => setSelectedEvent(null)}
              style={{ position: 'absolute', top: 16, right: 16, border: 'none', background: 'transparent', cursor: 'pointer', fontSize: '1.5rem', color: 'var(--text-muted)' }}
            >
              ✕
            </button>

            <div style={{ display: 'flex', gap: '24px', alignItems: 'center' }}>
              <div style={{ width: '120px', height: '120px', borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '2px solid var(--primary)', background: 'var(--bg-secondary)' }}>
                {subjectImgById[selectedEvent.sid] && <img src={`${getApiBase()}${subjectImgById[selectedEvent.sid]}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />}
              </div>
              <div>
                <h2 style={{ fontSize: '1.5rem', fontWeight: 800 }}>{selectedEvent.sid}</h2>
                <div style={{ marginTop: '8px', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                  Showing {selectedEvent.events.length} records.
                </div>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '20px' }}>
              {selectedEvent.events.map(ev => (
                <div key={ev.event_id} style={{ borderRadius: 'var(--radius-md)', overflow: 'hidden', background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
                  <div style={{ height: '160px', position: 'relative' }}>
                    <img src={`${getApiBase()}${ev.image_path}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                    <div style={{ position: 'absolute', top: 8, right: 8, padding: '4px 8px', background: 'rgba(0,0,0,0.6)', color: 'white', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 700 }}>
                      {ev.similarity != null ? (ev.similarity * 100).toFixed(1) : '--'}%
                    </div>
                  </div>
                  <div style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <div style={{ fontSize: '0.8125rem', fontWeight: 700 }}>{ev.camera}</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{fmtTs(ev.ts)}</div>
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
