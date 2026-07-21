import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { deleteSubject, subjects, type SubjectItem, subjectImages, type SubjectImageItem, getApiBase, type DateFilter, getBranches, type BranchItem } from '../lib/api';

const PAGE_SIZE = 24;
// Pull a large batch once, then paginate client-side (reliable Prev/Next).
const FETCH_LIMIT = 1000;

export default function Subjects() {
  const [items, setItems] = useState<SubjectItem[]>([]);
  const [page, setPage] = useState<number>(1);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState<string>('');
  const [previews, setPreviews] = useState<Record<string, SubjectImageItem | null>>({});
  const [dateMode, setDateMode] = useState<'all' | 'day' | 'range'>('all');
  const [day, setDay] = useState<string>('');
  const [fromDay, setFromDay] = useState<string>('');
  const [toDay, setToDay] = useState<string>('');

  const [branchList, setBranchList] = useState<BranchItem[]>([]);
  const [selectedBranch, setSelectedBranch] = useState<string>('');

  // Multi-select for bulk delete.
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [bulkBusy, setBulkBusy] = useState<boolean>(false);

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

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const pageClamped = Math.min(page, totalPages);
  const pageItems = useMemo(
    () => filtered.slice((pageClamped - 1) * PAGE_SIZE, pageClamped * PAGE_SIZE),
    [filtered, pageClamped]
  );

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const r = await subjects({
        limit: FETCH_LIMIT,
        with_counts: true,
        q: String(query || '').trim() || undefined,
        branch: selectedBranch || undefined,
        ...dateFilter
      });
      setItems(r.items || []);
    } catch (e: any) {
      setError(String(e));
      setItems([]);
    } finally {
      setLoading(false);
    }
  }

  // jump back to page 1 and reload (used when filters change)
  function reload() {
    setPage(1);
    load();
  }

  function goNext() {
    if (page >= totalPages || loading) return;
    setPage((p) => Math.min(p + 1, totalPages));
  }

  function goPrev() {
    if (page <= 1 || loading) return;
    setPage((p) => Math.max(1, p - 1));
  }

  async function loadBranches() {
    try {
      const r = await getBranches();
      setBranchList(r.branches || []);
    } catch (e) {
      console.error('Failed to load branches:', e);
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
    loadBranches();
  }, []);

  useEffect(() => {
    reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dateFilter, selectedBranch]);

  useEffect(() => {
    setPage(1);
    const t = setTimeout(() => {
      reload();
    }, 250);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query]);

  // Load preview thumbnails only for the subjects on the current page.
  useEffect(() => {
    loadPreviews(pageItems);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pageItems]);

  async function onDelete(s: string) {
    if (!confirm(`Delete subject ${s}?`)) return;
    try {
      await deleteSubject(s);
      setSelected((prev) => { const n = new Set(prev); n.delete(s); return n; });
      load();
    } catch (e: any) {
      alert(`Delete failed: ${String(e)}`);
    }
  }

  function toggleSelect(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  // Are all subjects on the current page selected?
  const pageAllSelected = pageItems.length > 0 && pageItems.every((it) => selected.has(it.subject_id));

  function toggleSelectPage() {
    setSelected((prev) => {
      const next = new Set(prev);
      if (pageAllSelected) pageItems.forEach((it) => next.delete(it.subject_id));
      else pageItems.forEach((it) => next.add(it.subject_id));
      return next;
    });
  }

  function clearSelection() { setSelected(new Set()); }

  async function onBulkDelete() {
    const ids = Array.from(selected);
    if (ids.length === 0) return;
    if (!confirm(`Delete ${ids.length} subject${ids.length > 1 ? 's' : ''}? This cannot be undone.`)) return;
    setBulkBusy(true);
    try {
      const results = await Promise.allSettled(ids.map((id) => deleteSubject(id)));
      const failed = results.filter((r) => r.status === 'rejected').length;
      clearSelection();
      await load();
      if (failed > 0) alert(`${failed} of ${ids.length} deletions failed.`);
    } finally {
      setBulkBusy(false);
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '16px', flexWrap: 'wrap' }}>
        <div>
          <h2 style={{ fontSize: '1.875rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '4px' }}>Subjects Directory</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9375rem' }}>
            Manage and monitor enrolled subjects and their face embeddings.
            {!loading && !error && (
              <span style={{ marginLeft: '8px', color: 'var(--text-muted)', fontWeight: 600 }}>
                · {filtered.length} subject{filtered.length !== 1 ? 's' : ''}
              </span>
            )}
          </p>
        </div>
        {pageItems.length > 0 && (
          <button
            onClick={toggleSelectPage}
            style={{ height: '38px', padding: '0 16px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600, fontSize: '0.875rem', display: 'flex', alignItems: 'center', gap: '8px' }}
          >
            <input type="checkbox" readOnly checked={pageAllSelected} style={{ width: '16px', height: '16px', accentColor: 'var(--primary)', pointerEvents: 'none' }} />
            {pageAllSelected ? 'Deselect Page' : 'Select Page'}
          </button>
        )}
      </header>

      <div className="card" style={{ display: 'flex', flexWrap: 'wrap', gap: '24px', alignItems: 'flex-end', background: 'var(--bg-primary)' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Branch</span>
          <select 
            value={selectedBranch} 
            onChange={(e) => setSelectedBranch(e.target.value)} 
            style={{ padding: '8px 12px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: '0.875rem', minWidth: '150px' }}
          >
            <option value="">All Branches</option>
            {branchList.map((b) => (
              <option key={b.branch_id} value={b.branch_id}>
                {b.name || b.branch_id} {b.subject_count !== undefined ? `(${b.subject_count})` : ''}
              </option>
            ))}
          </select>
        </div>

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

        <button onClick={reload} className="primary" style={{ height: '42px', padding: '0 24px' }}>
          Refresh
        </button>
      </div>

      {selected.size > 0 && (
        <div
          className="card"
          style={{
            position: 'sticky', top: '12px', zIndex: 10,
            display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px', flexWrap: 'wrap',
            padding: '12px 20px', background: 'var(--bg-primary)', border: '1px solid var(--primary)',
            boxShadow: 'var(--shadow-md, 0 4px 12px rgba(0,0,0,0.15))',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ background: 'var(--primary)', color: 'white', borderRadius: '99px', padding: '2px 12px', fontWeight: 700, fontSize: '0.875rem' }}>
              {selected.size}
            </span>
            <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
              subject{selected.size !== 1 ? 's' : ''} selected
            </span>
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <button onClick={clearSelection} disabled={bulkBusy} style={{ padding: '8px 16px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}>
              Clear
            </button>
            <button
              onClick={onBulkDelete}
              disabled={bulkBusy}
              style={{ padding: '8px 18px', borderRadius: 'var(--radius-sm)', border: 'none', background: 'var(--error)', color: 'white', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px', cursor: bulkBusy ? 'not-allowed' : 'pointer', opacity: bulkBusy ? 0.7 : 1 }}
            >
              {bulkBusy ? <span className="spinner" /> : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
              )}
              Delete Selected ({selected.size})
            </button>
          </div>
        </div>
      )}

      {loading && <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', padding: '40px', color: 'var(--text-muted)', fontWeight: 500 }}><span className="spinner" />Loading subjects...</div>}
      {error && <div style={{ color: 'var(--error)', background: 'rgba(239, 68, 68, 0.1)', padding: '16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--error)' }}>{error}</div>}

      {!loading && !error && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '16px' }}>
          {pageItems.map((it) => {
            const isSel = selected.has(it.subject_id);
            return (
            <div key={it.subject_id} className="card hover-lift" style={{ display: 'flex', flexDirection: 'column', gap: '12px', padding: '12px', borderRadius: 'var(--radius-sm)', cursor: 'default', border: isSel ? '2px solid var(--primary)' : '2px solid transparent', background: isSel ? 'color-mix(in srgb, var(--primary) 6%, var(--bg-primary))' : undefined, transition: 'border-color 0.15s, background 0.15s' }}>
              <div style={{ height: '160px', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', overflow: 'hidden', position: 'relative' }}>
                <label
                  onClick={(e) => e.stopPropagation()}
                  style={{ position: 'absolute', top: '8px', left: '8px', zIndex: 3, display: 'flex', alignItems: 'center', justifyContent: 'center', width: '28px', height: '28px', borderRadius: 'var(--radius-sm)', background: 'rgba(0,0,0,0.35)', backdropFilter: 'blur(2px)', cursor: 'pointer' }}
                >
                  <input
                    type="checkbox"
                    checked={isSel}
                    onChange={() => toggleSelect(it.subject_id)}
                    style={{ width: '18px', height: '18px', accentColor: 'var(--primary)', cursor: 'pointer' }}
                  />
                </label>
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
                  style={{ flex: 1, textAlign: 'center', background: 'var(--primary)', color: 'white', padding: '8px', borderRadius: 'var(--radius-sm)', fontWeight: 600, fontSize: '0.875rem' }}
                >
                  View Details
                </Link>
                <button
                  onClick={() => onDelete(it.subject_id)}
                  style={{ padding: '8px 12px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--error)', color: 'var(--error)', background: 'transparent' }}
                  onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.05)'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>
                </button>
              </div>
            </div>
            );
          })}
          {filtered.length === 0 && (
            <div style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '80px', color: 'var(--text-muted)', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)', border: '2px dashed var(--border)' }}>
              No subjects found matching your criteria.
            </div>
          )}
        </div>
      )}

      {!error && filtered.length > 0 && (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '12px', marginTop: '8px', paddingBottom: '8px' }}>
          <button onClick={goPrev} disabled={pageClamped <= 1 || loading} style={{ padding: '8px 18px', borderRadius: 'var(--radius-sm)' }}>
            Prev
          </button>
          <span style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', fontWeight: 600, minWidth: '90px', textAlign: 'center' }}>
            Page {pageClamped} / {totalPages}
          </span>
          <button onClick={goNext} disabled={pageClamped >= totalPages || loading} className="primary" style={{ padding: '8px 18px', borderRadius: 'var(--radius-sm)' }}>
            Next
          </button>
        </div>
      )}
    </div>
  );
}
