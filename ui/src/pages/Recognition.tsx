import { useEffect, useMemo, useState } from 'react';
import { getApiBase, getBranches, recognitionCameras, recognitionEvents, recognitionStats, setRecognitionEventFeedback, subjectImages, type BranchItem, type EventFeedbackLabel, type RecognitionEvent, type RecognitionStatsResponse } from '../lib/api';

// Resolve a stored relative path (/thumbs/.., /images/.., /v1/...) to a full URL.
function imgUrl(p?: string | null): string {
  const s = String(p || '').trim();
  if (!s) return '';
  return s.startsWith('http') ? s : `${getApiBase()}${s}`;
}
import StatCard from '../components/StatCard';
import { useModalDismiss } from '../lib/useModalDismiss';

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

function num(v: any, d = 1): string {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(d) : '--';
}

// Renders meta.quality + meta.decision metrics as small chips.
function QualityBlock({ ev, compact = false }: { ev: RecognitionEvent; compact?: boolean }) {
  const q: any = (ev as any)?.meta?.quality;
  const decStatus: string = (ev as any)?.meta?.decision?.status || '';
  const hasTiming = ev.similarity != null || ev.processing_ms != null || ev.model_ms != null;
  if (!q && !decStatus && !hasTiming) return null;

  const yaw = Number(q?.yaw), pitch = Number(q?.pitch), blur = Number(q?.blur);
  type M = { label: string; value: string; warn?: boolean };
  const all: M[] = [];
  if (ev.similarity != null) all.push({ label: 'Sim', value: `${num(Number(ev.similarity) * 100, 1)}%` });
  if (ev.processing_ms != null) all.push({ label: 'Proc', value: `${ev.processing_ms}ms` });
  if (ev.model_ms != null) all.push({ label: 'Model', value: `${ev.model_ms}ms` });
  if (q) {
    all.push({ label: 'Blur', value: num(q.blur, 0), warn: Number.isFinite(blur) && blur < 50 });
    all.push({ label: 'Bright', value: num(q.brightness, 0) });
    all.push({ label: 'Face%', value: num(Number(q.face_ratio) * 100, 1) });
    all.push({ label: 'Face px', value: num(q.face_abs_px, 0) });
    all.push({ label: 'Landmark', value: num(q.landmark_score, 2) });
    all.push({ label: 'Yaw', value: `${num(q.yaw, 1)}°`, warn: Number.isFinite(yaw) && Math.abs(yaw) > 45 });
    all.push({ label: 'Pitch', value: `${num(q.pitch, 1)}°`, warn: Number.isFinite(pitch) && Math.abs(pitch) > 45 });
    if (Array.isArray(q.face_crop_shape)) all.push({ label: 'Crop', value: q.face_crop_shape.join('×') });
  }
  const shown = compact ? all.filter(m => ['Sim', 'Proc', 'Model', 'Blur', 'Yaw', 'Pitch'].includes(m.label)) : all;

  const statusTone = (s: string) =>
    s === 'ok' || s === 'match' || s === 'embedded' ? 'var(--success)'
      : s === 'rejected' ? 'var(--error)' : 'var(--text-secondary)';

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', alignItems: 'center' }}>
      {decStatus && (
        <span style={{ padding: '2px 8px', borderRadius: '99px', fontSize: '0.625rem', fontWeight: 800, textTransform: 'uppercase', background: 'var(--bg-primary)', border: `1px solid ${statusTone(decStatus)}`, color: statusTone(decStatus) }}>
          {decStatus}
        </span>
      )}
      {q?.status && !compact && (
        <span style={{ padding: '2px 8px', borderRadius: '99px', fontSize: '0.625rem', fontWeight: 700, background: 'var(--bg-primary)', border: `1px solid ${statusTone(q.status)}`, color: statusTone(q.status) }}>
          Q:{q.status}{q.reason ? ` (${q.reason})` : ''}
        </span>
      )}
      {shown.map(m => (
        <span key={m.label} title={m.label} style={{
          padding: '2px 6px', borderRadius: '4px', fontSize: '0.625rem', fontWeight: 600,
          background: m.warn ? 'rgba(239,68,68,0.12)' : 'var(--bg-secondary)',
          color: m.warn ? 'var(--error)' : 'var(--text-secondary)',
          border: '1px solid var(--border)'
        }}>
          <span style={{ opacity: 0.7 }}>{m.label}</span> {m.value}
        </span>
      ))}
    </div>
  );
}

type DecisionFilter = '' | 'match' | 'no_match' | 'rejected';

export default function Recognition() {
  const [items, setItems] = useState<RecognitionEvent[]>([]);
  const [stats, setStats] = useState<RecognitionStatsResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingStats, setLoadingStats] = useState<boolean>(false);
  const [camera, setCamera] = useState<string>('');
  const [cameraOptions, setCameraOptions] = useState<string[]>([]);
  const [decision, setDecision] = useState<DecisionFilter>('');
  const [subjectId, setSubjectId] = useState<string>('');
  const [minSim, setMinSim] = useState<string>('');
  const [maxSim, setMaxSim] = useState<string>('');
  const [branchList, setBranchList] = useState<BranchItem[]>([]);
  const [selectedBranch, setSelectedBranch] = useState<string>('');
  // Client-side pagination over one large fetch (reliable Prev/Next + branch filter).
  const FETCH_LIMIT = 2000;
  const PAGE_ROWS = 25;
  const [page, setPage] = useState<number>(1);
  // Only matches with similarity above this fraction are shown.
  const MATCH_MIN_SIM = 0.5;

  const [uniqueCount, setUniqueCount] = useState<number | null>(null);
  const [uniqueCountLoading, setUniqueCountLoading] = useState<boolean>(false);
  const [selectedEvent, setSelectedEvent] = useState<{ sid: string, events: RecognitionEvent[] } | null>(null);
  // Enrolled reference image (one per matched subject), keyed by subject_id.
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

  async function load() {
    setLoading(true);
    setErr(null);
    try {
      const resp = await recognitionEvents({
        decision: decision || undefined,
        camera: camera || undefined,
        branch: selectedBranch || undefined,
        subject_id: subjectId || undefined,
        min_similarity: minSimNum != null ? minSimNum : undefined,
        max_similarity: maxSimNum != null ? maxSimNum : undefined,
        limit: FETCH_LIMIT,
        day: dateMode === 'day' ? (day || null) : null,
        from_day: dateMode === 'range' ? (fromDay || null) : null,
        to_day: dateMode === 'range' ? (toDay || null) : null,
      });
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
    (async () => {
      try {
        const r = await getBranches();
        if (cancelled) return;
        setBranchList(r.branches || []);
      } catch { /* ignore */ }
    })();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    setPage(1);
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [decision, camera, selectedBranch, subjectId, dateMode, day, fromDay, toDay, minSim, maxSim]);

  useEffect(() => {
    loadStats();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camera, dateMode, day, fromDay, toDay]);

  const filtered = useMemo(() => {
    let out = items;
    if (decision) out = out.filter(i => String(i.decision || '') === decision);
    if (camera) out = out.filter(i => String(i.camera || '') === camera);
    if (subjectId) out = out.filter(i => String(i.subject_id || '') === subjectId);
    // Match threshold: hide weak matches (similarity <= 50%).
    out = out.filter(i => String(i.decision || '') !== 'match' || Number(i.similarity || 0) > MATCH_MIN_SIM);
    return out;
  }, [items, decision, camera, subjectId, selectedBranch]);

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

  const totalPages = Math.max(1, Math.ceil(displayRows.length / PAGE_ROWS));
  const pageClamped = Math.min(page, totalPages);
  const pagedRows = useMemo(
    () => displayRows.slice((pageClamped - 1) * PAGE_ROWS, pageClamped * PAGE_ROWS),
    [displayRows, pageClamped]
  );
  const rowOffset = (pageClamped - 1) * PAGE_ROWS;

  // Fetch one enrolled reference image per matched subject on the current page.
  useEffect(() => {
    let cancelled = false;
    const sids = Array.from(new Set(
      pagedRows
        .map(r => (r.type === 'grouped' ? r.sid : r.event.subject_id))
        .map(s => String(s || '').trim())
        .filter(s => s && s !== '—' && !subjectImgById[s])
    )).slice(0, 40);
    if (!sids.length) return;
    (async () => {
      for (const sid of sids) {
        try {
          const resp = await subjectImages(sid, { limit: 1 });
          const first = resp?.items?.[0];
          const p = (first?.image_path || first?.thumb_path || '').trim();
          if (!p || cancelled) continue;
          setSubjectImgById(prev => (prev[sid] ? prev : { ...prev, [sid]: p }));
        } catch { /* ignore */ }
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pagedRows]);

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

  useModalDismiss(!!selectedEvent, () => setSelectedEvent(null));

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
              setPage(1);
              setCamera('');
              setSelectedBranch('');
              setDecision('');
              setSubjectId('');
              setMinSim('');
              setMaxSim('');
              setDateMode('all');
              setDay(today);
              setFromDay(today);
              setToDay(today);
              load();
            }}
            style={{ fontWeight: 600 }}
          >
            Reset
          </button>
          <button onClick={() => load()} className="primary" style={{ fontWeight: 600 }}>
            {loading ? <><span className="spinner" />Refreshing...</> : 'Refresh Feed'}
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
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Branch</span>
          <select value={selectedBranch} onChange={e => setSelectedBranch(e.target.value)} style={{ padding: '8px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}>
            <option value="">All Branches</option>
            {branchList.map(b => (
              <option key={b.branch_id} value={b.branch_id}>{b.name || b.branch_id}</option>
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

      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap', fontSize: '0.8125rem', fontWeight: 600 }}>
        {loading ? (
          <span style={{ color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '8px' }}><span className="spinner" />Fetching records...</span>
        ) : (
          <>
            <span style={{ color: 'var(--text-secondary)' }}>Showing {pagedRows.length} of {displayRows.length} rows</span>
            <span style={{ padding: '3px 10px', borderRadius: '99px', background: 'rgba(16,185,129,0.12)', color: 'var(--success)' }}>{counts.match} match</span>
            <span style={{ padding: '3px 10px', borderRadius: '99px', background: 'rgba(107,114,128,0.12)', color: 'var(--text-secondary)' }}>{counts.noMatch} unknown</span>
            <span style={{ padding: '3px 10px', borderRadius: '99px', background: 'rgba(239,68,68,0.12)', color: 'var(--error)' }}>{counts.rejected} rejected</span>
          </>
        )}
      </div>

      {err && (
        <div style={{ padding: '16px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--error)', borderRadius: 'var(--radius-md)', color: 'var(--error)', fontWeight: 500 }}>
          {err}
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
        <span>Reference / Response</span>
        <span>Camera Name</span>
        <span>Entry Time</span>
        <span>Exit Time</span>
        <span>Duration</span>
        <span style={{ textAlign: 'center' }}>View</span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {pagedRows.map((row, idx) => {
          const isGrouped = row.type === 'grouped';
          const events = isGrouped ? row.events : [row.event];
          const firstEv = events[0];
          const lastEv = events[events.length - 1];
          const sid = isGrouped ? row.sid : (lastEv.subject_id || '—');
          const refSrc = imgUrl(subjectImgById[sid]);
          const duration = lastEv.ts - firstEv.ts;

          const cameras = Array.from(new Set(events.map(e => e.camera))).join(', ');

          return (
            <div
              key={row.key}
              className="card hover-lift"
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
              <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', fontWeight: 600 }}>{rowOffset + idx + 1}</div>

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
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', overflow: 'hidden' }}>
                <span style={{ fontWeight: 700, fontSize: '0.875rem', color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {sid}
                </span>
                <QualityBlock ev={lastEv} compact />
              </div>

              {/* Reference (enrolled match) + Response (captured frame) */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <div style={{ display: 'flex', gap: '4px', height: '60px' }}>
                  <div style={{ flex: 1, borderRadius: '4px', overflow: 'hidden', border: '1px solid var(--border)', background: 'var(--bg-secondary)', position: 'relative' }} title="Enrolled reference">
                    {refSrc ? (
                      <img src={refSrc} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                    ) : (
                      <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.5rem', color: 'var(--text-muted)' }}>REF</div>
                    )}
                  </div>
                  <div style={{ flex: 1, borderRadius: '4px', overflow: 'hidden', border: '1px solid var(--border)', background: 'var(--bg-secondary)', position: 'relative' }} title="Captured response">
                    {lastEv.image_path ? (
                      <img src={imgUrl(lastEv.image_path)} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                    ) : (
                      <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.5rem', color: 'var(--text-muted)' }}>LIVE</div>
                    )}
                  </div>
                </div>
                {/* Similarity Bar */}
                {lastEv.decision === 'match' && (() => {
                  const pct = (lastEv.similarity || 0) * 100;
                  const barColor = pct >= 75 ? 'var(--success)' : pct >= 60 ? 'var(--warning)' : 'var(--error)';
                  return (
                    <div style={{ height: '16px', background: 'var(--bg-secondary)', borderRadius: '8px', overflow: 'hidden', border: '1px solid var(--border)', position: 'relative' }} title={`Similarity ${pct.toFixed(1)}%`}>
                      <div style={{
                        width: `${Math.min(100, pct)}%`,
                        height: '100%',
                        background: barColor,
                        borderRadius: '8px',
                        transition: 'width 0.3s ease'
                      }} />
                      <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '9px', fontWeight: 800, color: pct > 55 ? 'white' : 'var(--text-primary)' }}>
                        {pct.toFixed(0)}%
                      </div>
                    </div>
                  );
                })()}
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

      {!loading && displayRows.length === 0 && (
        <div style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '64px', color: 'var(--text-muted)', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)', border: '2px dashed var(--border)' }}>
          No recognition events match your filters.
        </div>
      )}

      {displayRows.length > 0 && (
        <footer style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '12px', padding: '16px' }}>
          <button
            disabled={pageClamped <= 1 || loading}
            onClick={() => setPage(p => Math.max(1, p - 1))}
            style={{ padding: '10px 24px', minWidth: '110px' }}
          >
            ← Prev
          </button>
          <span style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', fontWeight: 700, minWidth: '120px', textAlign: 'center' }}>
            Page {pageClamped} / {totalPages}
          </span>
          <button
            disabled={pageClamped >= totalPages || loading}
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            className={pageClamped < totalPages ? 'primary' : ''}
            style={{ padding: '10px 24px', minWidth: '110px' }}
          >
            Next →
          </button>
        </footer>
      )}

      {/* Full View Modal */}
      {selectedEvent && (
        <div className="modal-overlay" onClick={() => setSelectedEvent(null)} style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000, padding: '40px' }}>
          <div className="card modal-content" onClick={e => e.stopPropagation()} style={{ width: '100%', maxWidth: '1000px', maxHeight: '90vh', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '24px', padding: '32px', position: 'relative' }}>
            <button
              onClick={() => setSelectedEvent(null)}
              className="modal-close"
              style={{ position: 'absolute', top: 16, right: 16, background: 'transparent', cursor: 'pointer', fontSize: '1.5rem', color: 'var(--text-muted)' }}
            >
              ✕
            </button>

            <div style={{ display: 'flex', gap: '24px', alignItems: 'center' }}>
              <div style={{ width: '120px', height: '120px', borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '2px solid var(--primary)', background: 'var(--bg-secondary)' }} title="Enrolled reference">
                {(() => {
                  const src = imgUrl(subjectImgById[selectedEvent.sid]) || imgUrl(selectedEvent.events[0]?.image_path);
                  return src ? <img src={src} style={{ width: '100%', height: '100%', objectFit: 'cover' }} /> : null;
                })()}
              </div>
              <div>
                <h2 style={{ fontSize: '1.5rem', fontWeight: 800 }}>{selectedEvent.sid}</h2>
                <div style={{ marginTop: '8px', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                  Showing {selectedEvent.events.length} records.
                </div>
              </div>
            </div>

            <h4 style={{ fontSize: '0.8125rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              Response Images ({selectedEvent.events.length})
            </h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '20px' }}>
              {selectedEvent.events.map(ev => (
                <div key={ev.event_id} style={{ borderRadius: 'var(--radius-md)', overflow: 'hidden', background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
                  <div style={{ height: '160px', position: 'relative' }}>
                    <img src={imgUrl(ev.image_path)} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
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
                    <div style={{ marginTop: '8px' }}>
                      <QualityBlock ev={ev} />
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
