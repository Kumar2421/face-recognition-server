import { useRef, useState } from 'react';
import { facesRecognizeUpload, facesSearchUpload, getApiBase, qualityCheckUpload } from '../lib/api';

type Item = { subject_id: string; similarity: number; point_id?: string; image_id?: string; thumb_path?: string };

export default function Search() {
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [topK, setTopK] = useState<number>(5);
  const [results, setResults] = useState<Item[]>([]);
  const [queryThumb, setQueryThumb] = useState<string | null>(null);
  const [quality, setQuality] = useState<any>(null);
  const [error, setError] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [dateMode, setDateMode] = useState<'all' | 'day' | 'range'>('all');
  const [day, setDay] = useState<string>('');
  const [fromDay, setFromDay] = useState<string>('');
  const [toDay, setToDay] = useState<string>('');

  async function runSearch(kind: 'search' | 'recognize') {
    setError('');
    setResults([]);
    setQuality(null);
    const f = fileRef.current?.files?.[0];
    if (!f) {
      setError('Select an image file');
      return;
    }
    try {
      setLoading(true);
      const filter = {
        day: dateMode === 'day' ? (day || null) : null,
        from_day: dateMode === 'range' ? (fromDay || null) : null,
        to_day: dateMode === 'range' ? (toDay || null) : null,
      };
      const r =
        kind === 'search'
          ? await facesSearchUpload(f, topK, filter)
          : await facesRecognizeUpload(f, topK, filter);
      const items = (r?.results || []) as Item[];
      setQueryThumb(r?.query_thumb_path ? `${getApiBase()}${r.query_thumb_path}` : null);
      setResults(items);
    } catch (e: any) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function runQualityCheck() {
    setError('');
    setResults([]);
    setQuality(null);
    const f = fileRef.current?.files?.[0];
    if (!f) {
      setError('Select an image file');
      return;
    }
    try {
      setLoading(true);
      const r = await qualityCheckUpload(f);
      setQuality(r);
    } catch (e: any) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      <header>
        <h2 style={{ fontSize: '1.875rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '4px' }}>Search & Recognition</h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9375rem' }}>Verify identity or find similar faces in the database.</p>
      </header>

      <div className="card" style={{ display: 'grid', gap: '24px', maxWidth: '800px', background: 'var(--bg-primary)' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 120px', gap: '20px' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <label style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Query Image</label>
            <div style={{ 
              border: '2px dashed var(--border)', 
              borderRadius: 'var(--radius-md)', 
              padding: '24px', 
              textAlign: 'center',
              background: 'var(--bg-secondary)',
              cursor: 'pointer'
            }} onClick={() => fileRef.current?.click()}>
              <input type="file" ref={fileRef} accept="image/*" style={{ display: 'none' }} />
              <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{fileRef.current?.files?.[0]?.name || 'Select Image...'}</span>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <label style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Top Matches</label>
            <input
              type="number"
              value={topK}
              min={1}
              max={50}
              onChange={(e) => setTopK(Math.max(1, Math.min(50, Number(e.target.value) || 5)))}
              style={{ padding: '12px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}
            />
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <span style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Advanced Filters</span>
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
            <select value={dateMode} onChange={(e) => setDateMode(e.target.value as any)} style={{ padding: '10px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)' }}>
              <option value="all">All Records</option>
              <option value="day">Single Day</option>
              <option value="range">Custom Range</option>
            </select>

            {dateMode === 'day' && (
              <input type="date" value={day} onChange={(e) => setDay(e.target.value)} style={{ padding: '10px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)' }} />
            )}

            {dateMode === 'range' && (
              <>
                <input type="date" value={fromDay} onChange={(e) => setFromDay(e.target.value)} style={{ padding: '10px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)' }} />
                <input type="date" value={toDay} onChange={(e) => setToDay(e.target.value)} style={{ padding: '10px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)' }} />
              </>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', gap: '12px', borderTop: '1px solid var(--border)', paddingTop: '20px' }}>
          <button onClick={() => runSearch('search')} className="primary" style={{ flex: 1, padding: '12px', fontWeight: 700 }}>Search Database</button>
          <button onClick={() => runSearch('recognize')} style={{ flex: 1, padding: '12px', fontWeight: 600, background: 'rgba(167, 139, 250, 0.1)', color: '#7c3aed', border: '1px solid rgba(167, 139, 250, 0.2)' }}>Recognize Face</button>
          <button onClick={runQualityCheck} style={{ flex: 1, padding: '12px', fontWeight: 600 }}>Quality Check</button>
        </div>
      </div>

      {loading && <div style={{ color: 'var(--text-muted)' }}>Processing request...</div>}
      {error && <div style={{ padding: '16px', background: 'rgba(239, 68, 68, 0.1)', color: 'var(--error)', borderRadius: 'var(--radius-md)', border: '1px solid var(--error)' }}>{error}</div>}

      {quality && (
        <section className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h3 style={{ fontSize: '1.125rem', fontWeight: 700 }}>Quality Assessment</h3>
            <span style={{ 
              padding: '4px 16px', 
              borderRadius: '99px', 
              fontSize: '0.75rem', 
              fontWeight: 800, 
              background: String(quality?.total_quality || '').toLowerCase() === 'pass' ? 'var(--success)' : 'var(--error)',
              color: 'white'
            }}>
              {String(quality?.total_quality || (quality?.ok ? 'pass' : 'fail')).toUpperCase()}
            </span>
          </div>
          <pre style={{ margin: 0, padding: '16px', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)', fontSize: '0.8125rem', overflowX: 'auto', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
            {JSON.stringify(quality, null, 2)}
          </pre>
        </section>
      )}

      {(queryThumb || results.length > 0) && (
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(300px, 400px) 1fr', gap: '32px', alignItems: 'flex-start' }}>
          <section className="card" style={{ position: 'sticky', top: '24px' }}>
            <h3 style={{ fontSize: '0.875rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '12px' }}>Query Image</h3>
            <div style={{ background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)', padding: '8px', border: '1px solid var(--border)', minHeight: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {queryThumb ? (
                <img src={queryThumb} alt="query" style={{ maxWidth: '100%', maxHeight: '400px', borderRadius: 'var(--radius-sm)', objectFit: 'contain' }} />
              ) : (
                <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>No preview available</div>
              )}
            </div>
          </section>

          <section>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '20px', color: 'var(--text-primary)' }}>Match Results (Top {topK})</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '20px' }}>
              {results.map((it, i) => (
                <div key={i} className="card" style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <div style={{ height: '160px', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)', overflow: 'hidden' }}>
                    {it.thumb_path ? (
                      <img src={`${getApiBase()}${it.thumb_path}`} alt="match" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                    ) : (
                      <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.75rem' }}>No Image</div>
                    )}
                  </div>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: '1rem', marginBottom: '4px', wordBreak: 'break-all' }}>{it.subject_id || '—'}</div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <div style={{ flex: 1, height: '4px', background: 'var(--bg-secondary)', borderRadius: '2px', overflow: 'hidden' }}>
                        <div style={{ width: `${it.similarity * 100}%`, height: '100%', background: 'var(--primary)' }} />
                      </div>
                      <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--primary)', minWidth: '40px' }}>{(it.similarity * 100).toFixed(1)}%</span>
                    </div>
                    {it.point_id && (
                      <div style={{ fontSize: '0.625rem', color: 'var(--text-muted)', marginTop: '8px', wordBreak: 'break-all' }}>POINT: {it.point_id}</div>
                    )}
                  </div>
                </div>
              ))}
              {results.length === 0 && !loading && (
                <div style={{ gridColumn: '1 / -1', padding: '40px', textAlign: 'center', color: 'var(--text-muted)', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-lg)', border: '2px dashed var(--border)' }}>
                  No matching subjects found.
                </div>
              )}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
