import { useMemo, useState, useEffect } from 'react';
import { getApiBase, searchEvents, searchEventsStats, type SearchEvent, type SearchEventsStatsResponse } from '../lib/api';

export default function SearchEvents() {
  const [items, setItems] = useState<SearchEvent[]>([]);
  const [cursor, setCursor] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [selected, setSelected] = useState<SearchEvent | null>(null);
  const [matchFilter, setMatchFilter] = useState<'all' | 'match' | 'no_match'>('all');
  const matchThreshold = 0.55;
  const [stats, setStats] = useState<SearchEventsStatsResponse | null>(null);
  const [statsLoading, setStatsLoading] = useState<boolean>(false);

  const [dateMode, setDateMode] = useState<'all' | 'day' | 'range'>('all');
  const [day, setDay] = useState<string>('');
  const [fromDay, setFromDay] = useState<string>('');
  const [toDay, setToDay] = useState<string>('');
  const fallbackImg =
    'data:image/svg+xml;utf8,' +
    encodeURIComponent(
      `<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300"><rect width="100%" height="100%" fill="#0f172a"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#94a3b8" font-family="Arial" font-size="20">No Image</text></svg>`
    );

  const activeDateParams = useMemo(() => {
    return {
      day: dateMode === 'day' ? (day || null) : null,
      from_day: dateMode === 'range' ? (fromDay || null) : null,
      to_day: dateMode === 'range' ? (toDay || null) : null,
    } as { day?: string | null; from_day?: string | null; to_day?: string | null };
  }, [dateMode, day, fromDay, toDay]);

  const load = async () => {
    try {
      setLoading(true);
      const res = await searchEvents({ limit: 50, ...activeDateParams });
      setItems(res.items);
      setCursor(res.cursor || null);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const loadMore = async () => {
    if (!cursor || loadingMore) return;
    try {
      setLoadingMore(true);
      const res = await searchEvents({ limit: 50, cursor, ...activeDateParams });
      setItems(prev => [...prev, ...res.items]);
      setCursor(res.cursor || null);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingMore(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    // when date filters change, reset pagination + reload first page
    setCursor(null);
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeDateParams.day, activeDateParams.from_day, activeDateParams.to_day]);

  const fmtTs = (ts: number) => {
    return new Date(ts * 1000).toLocaleString();
  };

  const isMatch = (ev: SearchEvent) => {
    const sim = ev.top_similarity;
    const sid = String(ev.top_subject_id || '').trim();
    if (!sid) return false;
    if (sim == null || Number.isNaN(Number(sim))) return false;
    return Number(sim) >= matchThreshold;
  };

  const getAnyBbox = (ev: SearchEvent): number[] | null => {
    const m: any = ev.meta || {};
    const candidates = [m?.bbox, m?.det?.bbox, m?.face?.bbox, m?.result?.bbox];
    for (const c of candidates) {
      if (Array.isArray(c) && c.length >= 4 && c.every((v: any) => typeof v === 'number')) return c as number[];
    }
    return null;
  };

  const fmtBbox = (bbox: number[] | null): string => {
    if (!bbox || bbox.length < 4) return '--';
    return bbox.slice(0, 4).map(v => Math.round(Number(v))).join(',');
  };

  const getQueryThumbSrc = (ev: SearchEvent): string => {
    const p = String(ev.query_thumb_path || '').trim();
    // Backend may store a filesystem-style path like /thumbs/<event_id>.jpg but not serve it.
    // Always prefer the explicit search_history asset endpoint when available.
    if (p.startsWith('/v1/')) return `${getApiBase()}${p}`;
    if (p.startsWith('/thumbs/') || p.includes('/thumbs/')) {
      return `${getApiBase()}/v1/search_history/asset/thumb/${ev.event_id}`;
    }
    if (p.startsWith('/')) return `${getApiBase()}${p}`;
    return `${getApiBase()}/v1/search_history/asset/thumb/${ev.event_id}`;
  };

  const filteredItems = (() => {
    if (matchFilter === 'all') return items;
    if (matchFilter === 'match') return items.filter(isMatch);
    return items.filter(ev => !isMatch(ev));
  })();

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setStatsLoading(true);
        const resp = await searchEventsStats({
          match_threshold: matchThreshold,
          ...activeDateParams,
        });
        if (cancelled) return;
        setStats(resp);
      } catch (e) {
        if (cancelled) return;
        setStats(null);
      } finally {
        if (!cancelled) setStatsLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [matchThreshold, activeDateParams]);

  if (loading) {
    return <div style={{ display: 'flex', justifyContent: 'center', padding: '40px' }}>Loading events...</div>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h2 style={{ fontSize: '1.875rem', fontWeight: 800, letterSpacing: '-0.025em', color: 'var(--text-primary)', marginBottom: '4px' }}>Search History</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9375rem', fontWeight: 500 }}>Audit log for manual search and recognition attempts</p>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center', marginTop: '8px', flexWrap: 'wrap' }}>
            <div style={{ padding: '2px 10px', borderRadius: '999px', background: 'rgba(16, 185, 129, 0.12)', border: '1px solid rgba(16, 185, 129, 0.25)', color: 'var(--success)', fontSize: '0.75rem', fontWeight: 900 }}>
              Match: {statsLoading ? '…' : (stats?.match ?? 0)}
            </div>
            <div style={{ padding: '2px 10px', borderRadius: '999px', background: 'rgba(148, 163, 184, 0.16)', border: '1px solid rgba(148, 163, 184, 0.35)', color: 'var(--text-muted)', fontSize: '0.75rem', fontWeight: 900 }}>
              No Match: {statsLoading ? '…' : (stats?.no_match ?? 0)}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 800 }}>
              Threshold: {(matchThreshold * 100).toFixed(0)}%
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <select
            value={dateMode}
            onChange={(e) => setDateMode(e.target.value as any)}
            style={{ padding: '8px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
            title="Date Filter"
          >
            <option value="all">All Dates</option>
            <option value="day">Single Day</option>
            <option value="range">Range</option>
          </select>

          {dateMode === 'day' && (
            <input
              type="date"
              value={day}
              onChange={(e) => setDay(e.target.value)}
              style={{ padding: '7px 8px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
            />
          )}

          {dateMode === 'range' && (
            <>
              <input
                type="date"
                value={fromDay}
                onChange={(e) => setFromDay(e.target.value)}
                style={{ padding: '7px 8px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
              />
              <input
                type="date"
                value={toDay}
                onChange={(e) => setToDay(e.target.value)}
                style={{ padding: '7px 8px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
              />
            </>
          )}

          <select
            value={matchFilter}
            onChange={(e) => setMatchFilter(e.target.value as any)}
            style={{ padding: '8px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
            title={`Match filter (match if similarity >= ${matchThreshold})`}
          >
            <option value="all">All</option>
            <option value="match">Match</option>
            <option value="no_match">No Match</option>
          </select>
          <button onClick={load} className="btn btn-secondary" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            Refresh
          </button>
        </div>
      </header>

      {filteredItems.length === 0 ? (
        <div className="card" style={{ padding: '80px 24px', textAlign: 'center', color: 'var(--text-secondary)' }}>
          <p style={{ fontSize: '1rem', fontWeight: 500 }}>No search events found</p>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))', gap: '20px' }}>
          {filteredItems.map(ev => {
            const matched = isMatch(ev);
            const bbox = getAnyBbox(ev);
            const detScore = (ev.meta as any)?.det_score ?? (ev.meta as any)?.det?.score ?? null;

            return (
              <div key={ev.event_id} className="card" onClick={() => setSelected(ev)} style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', gap: '10px', padding: '10px', position: 'relative' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div style={{ overflow: 'hidden' }}>
                    <h4 style={{ fontWeight: 700, fontSize: '0.875rem', color: 'var(--text-primary)', whiteSpace: 'nowrap', textOverflow: 'ellipsis' }}>
                      {ev.top_subject_id || 'No Match'}
                    </h4>
                    <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', fontWeight: 500 }}>{fmtTs(ev.ts)}</div>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '6px' }}>
                    <button
                      onClick={(e) => { e.stopPropagation(); setSelected(ev); }}
                      style={{ padding: '6px', minWidth: 'auto', background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: '6px', cursor: 'pointer' }}
                      title="Expand View"
                    >
                      <span style={{ fontSize: '0.875rem' }}>⛶</span>
                    </button>

                    <div style={{
                      padding: '2px 8px',
                      borderRadius: '99px',
                      fontSize: '0.625rem',
                      fontWeight: 800,
                      background: matched ? 'rgba(16, 185, 129, 0.12)' : 'rgba(148, 163, 184, 0.16)',
                      color: matched ? 'var(--success)' : 'var(--text-muted)',
                      border: matched ? '1px solid rgba(16, 185, 129, 0.25)' : '1px solid rgba(148, 163, 184, 0.35)',
                    }}>
                      {matched ? 'MATCH' : 'NO MATCH'}
                    </div>

                    {ev.top_similarity != null && (
                      <div style={{
                        padding: '1px 8px',
                        borderRadius: '99px',
                        fontSize: '0.625rem',
                        fontWeight: 700,
                        background: 'rgba(37, 99, 235, 0.1)',
                        color: 'var(--primary)',
                        border: '1px solid rgba(37, 99, 235, 0.2)'
                      }}>
                        {(ev.top_similarity * 100).toFixed(1)}%
                      </div>
                    )}
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '0.625rem', color: 'var(--text-muted)', fontWeight: 700 }}>
                  <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={String((ev.meta as any)?.model || (ev.meta as any)?.model_name || '')}>
                    Model: {String((ev.meta as any)?.model || (ev.meta as any)?.model_name || '--')}
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    TopK: {Array.isArray(ev.results) ? ev.results.length : 0}
                  </div>
                  <div>
                    BBox: {fmtBbox(bbox)}
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    Det: {detScore != null ? Number(detScore).toFixed(3) : '--'}
                  </div>
                </div>

                <div style={{ height: '120px', borderRadius: 'var(--radius-md)', overflow: 'hidden', background: 'var(--bg-secondary)', display: 'flex', gap: '2px' }}>
                  <div style={{ flex: 1, position: 'relative' }}>
                    <img
                      src={getQueryThumbSrc(ev)}
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      onError={(e) => (e.currentTarget.src = fallbackImg)}
                    />
                    <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, padding: '2px', background: 'rgba(0,0,0,0.5)', color: 'white', fontSize: '0.55rem', textAlign: 'center', fontWeight: 600 }}>QUERY</div>
                  </div>
                  {ev.results && ev.results[0] && ev.results[0].thumb_path && (
                    <div style={{ flex: 1, position: 'relative' }}>
                      <img
                        src={`${getApiBase()}${ev.results[0].thumb_path}`}
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                        onError={(e) => (e.currentTarget.src = fallbackImg)}
                      />
                      <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, padding: '2px', background: 'rgba(0,0,0,0.5)', color: 'white', fontSize: '0.55rem', textAlign: 'center', fontWeight: 600 }}>BEST HIT</div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {cursor && (
        <div style={{ display: 'flex', justifyContent: 'center', paddingTop: '20px' }}>
          <button onClick={loadMore} disabled={loadingMore} className="btn btn-secondary" style={{ minWidth: '160px' }}>
            {loadingMore ? 'Loading...' : 'Load More Results'}
          </button>
        </div>
      )}

      {selected && (
        <div
          onClick={() => setSelected(null)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0,0,0,0.6)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '40px',
          }}
        >
          <div
            className="card"
            onClick={e => e.stopPropagation()}
            style={{
              width: '100%',
              maxWidth: '1000px',
              maxHeight: '90vh',
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: '20px',
              padding: '28px',
              position: 'relative',
            }}
          >
            <button
              onClick={() => setSelected(null)}
              style={{ position: 'absolute', top: 12, right: 12, border: 'none', background: 'transparent', cursor: 'pointer', fontSize: '1.5rem', color: 'var(--text-muted)' }}
            >
              ✕
            </button>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '12px', paddingRight: '28px' }}>
              <div>
                <div style={{ fontSize: '1.25rem', fontWeight: 900, color: 'var(--text-primary)' }}>Search Event Detail</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 700, marginTop: '2px', fontFamily: 'monospace' }}>
                  {selected.event_id}
                </div>
              </div>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <div style={{
                  padding: '4px 10px',
                  borderRadius: '99px',
                  fontSize: '0.75rem',
                  fontWeight: 900,
                  background: isMatch(selected) ? 'rgba(16, 185, 129, 0.12)' : 'rgba(148, 163, 184, 0.16)',
                  color: isMatch(selected) ? 'var(--success)' : 'var(--text-muted)',
                  border: isMatch(selected) ? '1px solid rgba(16, 185, 129, 0.25)' : '1px solid rgba(148, 163, 184, 0.35)',
                }}>
                  {isMatch(selected) ? 'MATCH' : 'NO MATCH'}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 800 }}>
                  ≥ {(matchThreshold * 100).toFixed(0)}%
                </div>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ fontSize: '0.75rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Query Image</div>
                <img
                  src={`${getApiBase()}/v1/search_history/asset/image/${selected.event_id}`}
                  style={{ width: '100%', aspectRatio: '1', objectFit: 'contain', borderRadius: 'var(--radius-md)', background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}
                />
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  <div>
                    <div style={{ fontSize: '0.75rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Timestamp</div>
                    <div style={{ fontWeight: 700 }}>{fmtTs(selected.ts)}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.75rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Similarity</div>
                    <div style={{ fontWeight: 800, color: 'var(--primary)' }}>{selected.top_similarity != null ? (selected.top_similarity * 100).toFixed(2) + '%' : 'N/A'}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.75rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Top Subject</div>
                    <div style={{ fontWeight: 900 }}>{selected.top_subject_id || 'No Match'}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.75rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase' }}>BBox</div>
                    <div style={{ fontWeight: 800, fontFamily: 'monospace' }}>{fmtBbox(getAnyBbox(selected))}</div>
                  </div>
                </div>
              </div>
            </div>

            <div>
              <div style={{ fontSize: '0.875rem', fontWeight: 900, marginBottom: '10px' }}>Results</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(190px, 1fr))', gap: '12px' }}>
                {(selected.results || []).map((res, i) => (
                  <div key={i} style={{ borderRadius: 'var(--radius-md)', overflow: 'hidden', background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}>
                    <div style={{ height: '120px', position: 'relative', background: 'var(--bg-primary)' }}>
                      {res.thumb_path ? (
                        <img
                          src={`${getApiBase()}${res.thumb_path}`}
                          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                          onError={(e) => {
                            if (e.currentTarget.src !== fallbackImg) e.currentTarget.src = fallbackImg;
                          }}
                        />
                      ) : (
                        <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 800 }}>No Thumb</div>
                      )}
                      <div style={{ position: 'absolute', top: 8, right: 8, padding: '4px 8px', background: 'rgba(0,0,0,0.6)', color: 'white', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 800 }}>
                        {(Number(res.similarity) * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div style={{ padding: '10px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                      <div style={{ fontSize: '0.8125rem', fontWeight: 900 }}>{res.subject_id}</div>
                      <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                        {res.point_id ? `point:${String(res.point_id).slice(0, 8)}...` : ''}
                      </div>
                      <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                        {res.bbox && Array.isArray(res.bbox) ? `bbox:${fmtBbox(res.bbox as any)}` : ''}
                        {res.det_score != null ? `  det:${Number(res.det_score).toFixed(3)}` : ''}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {selected.meta != null && (
              <div>
                <div style={{ fontSize: '0.875rem', fontWeight: 900, marginBottom: '10px' }}>Model Metadata</div>
                <pre style={{ maxHeight: '260px', overflow: 'auto', background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '12px', fontSize: '0.75rem' }}>
                  {JSON.stringify(selected.meta, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
