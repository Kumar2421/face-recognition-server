import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { deleteSubject, subjects, type SubjectItem, subjectImages, type SubjectImageItem, getApiBase, type DateFilter } from '../lib/api';

export default function Subjects() {
  const [items, setItems] = useState<SubjectItem[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [limit, setLimit] = useState<number>(25);
  const [query, setQuery] = useState<string>('');
  const [previews, setPreviews] = useState<Record<string, SubjectImageItem | null>>({});
  const [dateMode, setDateMode] = useState<'all' | 'day' | 'range'>('all');
  const [day, setDay] = useState<string>('');
  const [fromDay, setFromDay] = useState<string>('');
  const [toDay, setToDay] = useState<string>('');

  const dateFilter: DateFilter = useMemo(() => {
    return {
      day: dateMode === 'day' ? (day || null) : null,
      from_day: dateMode === 'range' ? (fromDay || null) : null,
      to_day: dateMode === 'range' ? (toDay || null) : null,
    };
  }, [dateMode, day, fromDay, toDay]);

  const filtered = useMemo(() => {
    const q = String(query || '').trim().toLowerCase();
    if (!q) return items;
    return (items || []).filter((it) => String(it.subject_id || '').toLowerCase().includes(q));
  }, [items, query]);


  async function load(c: string | null, l: number) {
    setLoading(true);
    setError(null);
    try {
      const r = await subjects({ cursor: c || undefined, limit: l, with_counts: true, q: String(query || '').trim() || undefined, ...dateFilter });
      const list = r.items || [];
      setItems(list);
      setCursor(r.cursor || null);
      // Load one preview image per subject
      loadPreviews(list);
    } catch (e: any) {
      setError(String(e));
      setItems([]);
      setCursor(null);
    } finally {
      setLoading(false);
    }
  }

  async function loadPreviews(subs: SubjectItem[]) {
    const out: Record<string, SubjectImageItem | null> = {};
    try {
      await Promise.all(
        subs.map(async (s) => {
          try {
            const r = await subjectImages(s.subject_id, { limit: 1, ...dateFilter });
            out[s.subject_id] = (r.items && r.items.length > 0) ? r.items[0] : null;
          } catch {
            out[s.subject_id] = null;
          }
        })
      );
      setPreviews(out);
    } finally {
    }
  }

  useEffect(() => {
    load(null, limit);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [limit, dateFilter]);

  useEffect(() => {
    const t = setTimeout(() => {
      load(null, limit);
    }, 250);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query]);

  async function onDelete(s: string) {
    if (!confirm(`Delete subject ${s}?`)) return;
    try {
      await deleteSubject(s);
      await load(null, limit);
    } catch (e: any) {
      alert(`Delete failed: ${String(e)}`);
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h2 style={{ fontSize: '1.875rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '4px' }}>Subjects Directory</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9375rem' }}>Manage and monitor enrolled subjects and their face embeddings.</p>
        </div>
      </header>

      <div className="card" style={{ display: 'flex', flexWrap: 'wrap', gap: '24px', alignItems: 'flex-end', background: 'var(--bg-primary)' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Date Range</span>
          <div style={{ display: 'flex', gap: '8px' }}>
            <select 
              value={dateMode} 
              onChange={(e) => setDateMode(e.target.value as any)} 
              style={{ padding: '8px 12px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: '0.875rem' }}
            >
              <option value="all">All Time</option>
              <option value="day">Single Day</option>
              <option value="range">Range</option>
            </select>
            {dateMode === 'day' && (
              <input type="date" value={day} onChange={(e) => setDay(e.target.value)} style={{ padding: '8px 12px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }} />
            )}
            {dateMode === 'range' && (
              <>
                <input type="date" value={fromDay} onChange={(e) => setFromDay(e.target.value)} style={{ padding: '8px 12px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }} />
                <input type="date" value={toDay} onChange={(e) => setToDay(e.target.value)} style={{ padding: '8px 12px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }} />
              </>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', flex: 1, minWidth: '200px' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Search Subject</span>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Type subject ID..."
            style={{ padding: '8px 12px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', width: '100%' }}
          />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', width: '100px' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Limit</span>
          <input 
            type="number" 
            min={5} 
            max={100} 
            value={limit} 
            onChange={(e) => setLimit(Math.max(5, Math.min(100, Number(e.target.value) || 25)))} 
            style={{ padding: '8px 12px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }} 
          />
        </div>

        <button onClick={() => load(null, limit)} className="primary" style={{ height: '42px', padding: '0 24px' }}>
          Refresh
        </button>
      </div>

      {loading && <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>Loading subjects...</div>}
      {error && <div style={{ color: 'var(--error)', background: 'rgba(239, 68, 68, 0.1)', padding: '16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--error)' }}>{error}</div>}

      {!loading && !error && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '24px' }}>
          {filtered.map((it) => (
            <div key={it.subject_id} className="card" style={{ display: 'flex', flexDirection: 'column', gap: '16px', transition: 'transform 0.2s, box-shadow 0.2s', cursor: 'default' }} onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(-4px)'; e.currentTarget.style.boxShadow = 'var(--shadow-md)'; }} onMouseLeave={(e) => { e.currentTarget.style.transform = 'none'; e.currentTarget.style.boxShadow = 'var(--shadow-sm)'; }}>
              <div style={{ height: '180px', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)', overflow: 'hidden', position: 'relative' }}>
                {(() => {
                  const p = previews[it.subject_id];
                  const src = p?.image_path || p?.thumb_path ? `${getApiBase()}${p.image_path || p.thumb_path}` : '';
                  if (src) {
                    return <img src={src} alt={it.subject_id} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />;
                  }
                  return (
                    <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                      No Preview
                    </div>
                  );
                })()}
                {it.embeddings_capped && (
                  <div style={{ position: 'absolute', top: '12px', right: '12px', background: 'var(--warning)', color: 'white', fontSize: '0.625rem', padding: '2px 8px', borderRadius: '99px', fontWeight: 700 }}>
                    CAPPED
                  </div>
                )}
              </div>
              
              <div>
                <h4 style={{ fontSize: '1.125rem', fontWeight: 700, marginBottom: '4px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{it.subject_id}</h4>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
                  <span>{typeof it.embeddings_cap === 'number' ? `${it.embeddings_count} / ${it.embeddings_cap}` : it.embeddings_count} Embeddings</span>
                </div>
              </div>

              <div style={{ display: 'flex', gap: '8px', marginTop: 'auto' }}>
                <Link
                  to={`/subjects/${encodeURIComponent(it.subject_id)}`}
                  style={{ flex: 1, textAlign: 'center', background: 'var(--primary)', color: 'white', padding: '8px', borderRadius: 'var(--radius-md)', fontWeight: 600, fontSize: '0.875rem' }}
                >
                  View Details
                </Link>
                <button 
                  onClick={() => onDelete(it.subject_id)} 
                  style={{ padding: '8px 12px', borderRadius: 'var(--radius-md)', border: '1px solid var(--error)', color: 'var(--error)', background: 'transparent' }}
                  onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.05)'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>
                </button>
              </div>
            </div>
          ))}
          {filtered.length === 0 && (
            <div style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '80px', color: 'var(--text-muted)', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-lg)', border: '2px dashed var(--border)' }}>
              No subjects found matching your criteria.
            </div>
          )}
        </div>
      )}

      {cursor && !loading && (
        <div style={{ display: 'flex', justifyContent: 'center', marginTop: '16px' }}>
          <button onClick={() => load(cursor, limit)} style={{ padding: '10px 32px' }}>Load More Subjects</button>
        </div>
      )}
    </div>
  );
}
