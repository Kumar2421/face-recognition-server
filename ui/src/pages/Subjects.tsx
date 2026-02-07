import { useEffect, useMemo, useRef, useState } from 'react';
import { deleteSubject, subjects, type SubjectItem, subjectImages, type SubjectImageItem, getApiBase } from '../lib/api';

export default function Subjects() {
  const [items, setItems] = useState<SubjectItem[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [limit, setLimit] = useState<number>(25);
  const [query, setQuery] = useState<string>('');
  const [previews, setPreviews] = useState<Record<string, SubjectImageItem | null>>({});
  const [previewsLoading, setPreviewsLoading] = useState<boolean>(false);
  const [searching, setSearching] = useState<boolean>(false);
  const [searchScanned, setSearchScanned] = useState<number>(0);
  const searchRunIdRef = useRef<number>(0);

  const filtered = useMemo(() => {
    const q = String(query || '').trim().toLowerCase();
    if (!q) return items;
    return (items || []).filter((it) => String(it.subject_id || '').toLowerCase().includes(q));
  }, [items, query]);

  const isSearchMode = useMemo(() => {
    return String(query || '').trim().length > 0;
  }, [query]);

  async function load(c: string | null, l: number) {
    setLoading(true);
    setError(null);
    try {
      const r = await subjects({ cursor: c || undefined, limit: l, with_counts: true });
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

  async function loadSearch(qRaw: string, l: number) {
    const q = String(qRaw || '').trim().toLowerCase();
    if (!q) return;

    searchRunIdRef.current += 1;
    const runId = searchRunIdRef.current;

    setSearching(true);
    setError(null);
    setSearchScanned(0);

    const maxMatches = 500;
    const maxPages = 15;
    const pageLimit = Math.max(50, Math.min(500, l || 200));

    let next: string | null = null;
    let page = 0;
    const out: SubjectItem[] = [];
    const seen = new Set<string>();

    try {
      while (page < maxPages && out.length < maxMatches) {
        if (runId !== searchRunIdRef.current) return;
        const r = await subjects({ cursor: next || undefined, limit: pageLimit, with_counts: true });
        const list = r.items || [];
        setSearchScanned(prev => prev + list.length);

        for (const it of list) {
          const sid = String(it.subject_id || '');
          if (!sid) continue;
          if (!sid.toLowerCase().includes(q)) continue;
          if (seen.has(sid)) continue;
          seen.add(sid);
          out.push(it);
          if (out.length >= maxMatches) break;
        }

        next = r.cursor || null;
        page += 1;
        if (!next) break;
      }

      if (runId !== searchRunIdRef.current) return;
      setItems(out);
      setCursor(null);
      loadPreviews(out);
    } catch (e: any) {
      setError(String(e));
      setItems([]);
      setCursor(null);
    } finally {
      if (runId === searchRunIdRef.current) setSearching(false);
    }
  }

  async function loadPreviews(subs: SubjectItem[]) {
    setPreviewsLoading(true);
    const out: Record<string, SubjectImageItem | null> = {};
    try {
      await Promise.all(
        subs.map(async (s) => {
          try {
            const r = await subjectImages(s.subject_id, { limit: 1 });
            out[s.subject_id] = (r.items && r.items.length > 0) ? r.items[0] : null;
          } catch {
            out[s.subject_id] = null;
          }
        })
      );
      setPreviews(out);
    } finally {
      setPreviewsLoading(false);
    }
  }

  useEffect(() => {
    if (String(query || '').trim()) {
      loadSearch(query, limit);
    } else {
      load(null, limit);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [limit]);

  useEffect(() => {
    const q = String(query || '').trim();
    if (!q) {
      searchRunIdRef.current += 1;
      setSearching(false);
      setSearchScanned(0);
      load(null, limit);
      return;
    }
    const t = setTimeout(() => {
      loadSearch(q, limit);
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
    <div>
      <h2>Subjects</h2>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 12 }}>
        <label style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <span>Search</span>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="subject_id"
            style={{ width: 220 }}
          />
        </label>
        <label style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <span>Page size</span>
          <input type="number" min={5} max={100} value={limit} onChange={(e) => setLimit(Math.max(5, Math.min(100, Number(e.target.value) || 25)))} style={{ width: 80 }} />
        </label>
        <button onClick={() => load(null, limit)} style={{ background: '#22c55e', color: '#111', border: 'none', padding: '6px 10px', borderRadius: 6, cursor: 'pointer', fontWeight: 600 }}>Refresh</button>
      </div>
      {isSearchMode && (
        <div style={{ marginBottom: 10, color: '#9ca3af' }}>
          {searching ? `Searching… scanned ${searchScanned} subjects` : `Search results: ${filtered.length}`}
        </div>
      )}
      {loading && <div>Loading...</div>}
      {error && <div style={{ color: 'red' }}>Error: {error}</div>}
      {!loading && !error && (
        <div>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', borderBottom: '1px solid #333', padding: '6px' }}>Subject</th>
                <th style={{ textAlign: 'left', borderBottom: '1px solid #333', padding: '6px' }}>Embeddings</th>
                <th style={{ textAlign: 'left', borderBottom: '1px solid #333', padding: '6px' }}>Image</th>
                <th style={{ textAlign: 'left', borderBottom: '1px solid #333', padding: '6px' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((it) => (
                <tr key={it.subject_id}>
                  <td style={{ borderBottom: '1px solid #222', padding: '6px' }}>{it.subject_id}</td>
                  <td style={{ borderBottom: '1px solid #222', padding: '6px' }}>{it.embeddings_count}</td>
                  <td style={{ borderBottom: '1px solid #222', padding: '6px' }}>
                    {(() => {
                      const p = previews[it.subject_id];
                      if (previewsLoading && p === undefined) return <div style={{ width: 80, height: 60, background: '#111', borderRadius: 6 }} />;
                      if (!p) return <div style={{ width: 80, height: 60, background: '#111', borderRadius: 6 }} />;
                      const src = p.image_path || p.thumb_path ? `${getApiBase()}${p.image_path || p.thumb_path}` : '';
                      return src ? (
                        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                        // @ts-ignore
                        <img src={src} alt={p.image_id} style={{ width: 80, height: 60, objectFit: 'cover', borderRadius: 6 }} />
                      ) : (
                        <div style={{ width: 80, height: 60, background: '#111', borderRadius: 6 }} />
                      );
                    })()}
                  </td>
                  <td style={{ borderBottom: '1px solid #222', padding: '6px' }}>
                    <button onClick={() => onDelete(it.subject_id)} style={{ background: '#ef4444', color: '#fff', border: 'none', padding: '6px 10px', borderRadius: 6, cursor: 'pointer' }}>Delete</button>
                  </td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={3} style={{ padding: '8px' }}>No subjects.</td>
                </tr>
              )}
            </tbody>
          </table>
          <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
            <button disabled={!cursor || isSearchMode} onClick={() => load(cursor, limit)} style={{ background: '#0ea5e9', color: '#111', border: 'none', padding: '6px 10px', borderRadius: 6, cursor: 'pointer', opacity: cursor && !isSearchMode ? 1 : 0.5 }}>Next</button>
          </div>

          {/* Removed selectable inline gallery and open link as requested */}
        </div>
      )}
    </div>
  );
}
