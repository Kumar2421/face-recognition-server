import { useRef, useState } from 'react';
import { facesRecognizeUpload, facesSearchUpload, getApiBase } from '../lib/api';

type Item = { subject_id: string; similarity: number; point_id?: string; image_id?: string; thumb_path?: string };

export default function Search() {
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [topK, setTopK] = useState<number>(5);
  const [results, setResults] = useState<Item[]>([]);
  const [queryThumb, setQueryThumb] = useState<string | null>(null);
  const [error, setError] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);

  async function runSearch(kind: 'search' | 'recognize') {
    setError('');
    setResults([]);
    const f = fileRef.current?.files?.[0];
    if (!f) {
      setError('Select an image file');
      return;
    }
    try {
      setLoading(true);
      const r =
        kind === 'search'
          ? await facesSearchUpload(f, topK)
          : await facesRecognizeUpload(f, topK);
      const items = (r?.results || []) as Item[];
      setQueryThumb(r?.query_thumb_path ? `${getApiBase()}${r.query_thumb_path}` : null);
      setResults(items);
    } catch (e: any) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h2>Search / Recognize</h2>
      <div style={{ display: 'grid', gap: 12, maxWidth: 560 }}>
        <label style={{ display: 'grid', gap: 6 }}>
          <span>Image</span>
          <input type="file" ref={fileRef} accept="image/*" />
        </label>
        <label style={{ display: 'grid', gap: 6 }}>
          <span>Top K</span>
          <input
            type="number"
            value={topK}
            min={1}
            max={50}
            onChange={(e) => setTopK(Math.max(1, Math.min(50, Number(e.target.value) || 5)))}
            style={{ padding: '8px 10px', borderRadius: 8, border: '1px solid #333', background: '#0a0a0a', color: '#e5e5e5', width: 120 }}
          />
        </label>
        <div style={{ display: 'flex', gap: 12 }}>
          <button onClick={() => runSearch('search')} style={{ background: '#22c55e', color: '#111', border: 'none', padding: '8px 12px', borderRadius: 8, cursor: 'pointer', fontWeight: 600 }}>Search</button>
          <button onClick={() => runSearch('recognize')} style={{ background: '#a78bfa', color: '#111', border: 'none', padding: '8px 12px', borderRadius: 8, cursor: 'pointer', fontWeight: 600 }}>Recognize</button>
        </div>
      </div>

      {loading && <div style={{ marginTop: 12 }}>Loading...</div>}
      {error && <div style={{ marginTop: 12, color: '#ef4444' }}>Error: {error}</div>}

      {(queryThumb || results.length > 0) && (
        <div style={{ marginTop: 16 }}>
          <h3>Gallery</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '240px 1fr', gap: 16 }}>
            <div>
              <div style={{ marginBottom: 8, color: '#9ca3af' }}>Query</div>
              <div style={{ border: '1px solid #222', borderRadius: 8, padding: 8, minHeight: 220, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#0f0f0f' }}>
                {queryThumb ? (
                  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                  // @ts-ignore
                  <img src={queryThumb} alt="query" style={{ maxWidth: '100%', maxHeight: 200, borderRadius: 6 }} />
                ) : (
                  <div style={{ color: '#6b7280' }}>No preview</div>
                )}
              </div>
            </div>
            <div>
              <div style={{ marginBottom: 8, color: '#9ca3af' }}>Matches (Top {topK})</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 12 }}>
                {results.map((it, i) => (
                  <div key={i} style={{ border: '1px solid #222', borderRadius: 8, padding: 8, background: '#0f0f0f' }}>
                    {it.thumb_path ? (
                      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                      // @ts-ignore
                      <img src={`${getApiBase()}${it.thumb_path}`} alt="thumb" style={{ width: '100%', height: 120, objectFit: 'cover', borderRadius: 6, marginBottom: 8 }} />
                    ) : (
                      <div style={{ height: 120, background: '#111', borderRadius: 6, marginBottom: 8 }} />
                    )}
                    <div style={{ fontWeight: 700 }}>{it.subject_id || '—'}</div>
                    <div style={{ color: '#9ca3af', fontSize: 12 }}>Sim: {(it.similarity * 100).toFixed(2)}%</div>
                    {it.point_id && <div style={{ color: '#6b7280', fontSize: 12 }}>Point: {it.point_id}</div>}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
