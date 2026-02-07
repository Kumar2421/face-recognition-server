import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { getApiBase, subjectImages, type SubjectImageItem } from '../lib/api';

export default function SubjectDetail() {
  const { id } = useParams<{ id: string }>();
  const subjectId = decodeURIComponent(id || '');
  const [items, setItems] = useState<SubjectImageItem[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [limit, setLimit] = useState<number>(30);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  async function load(c: string | null, l: number) {
    if (!subjectId) return;
    setLoading(true);
    setError(null);
    try {
      const r = await subjectImages(subjectId, { cursor: c || undefined, limit: l });
      setItems(r.items || []);
      setCursor(r.cursor || null);
    } catch (e: any) {
      setError(String(e));
      setItems([]);
      setCursor(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load(null, limit);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [subjectId, limit]);

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <Link to="/subjects" style={{ color: '#a3e635', textDecoration: 'none' }}>← Back</Link>
        <h2 style={{ margin: 0 }}>Subject: {subjectId}</h2>
      </div>

      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 12, marginBottom: 12 }}>
        <label style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <span>Page size</span>
          <input type="number" min={10} max={200} value={limit} onChange={(e) => setLimit(Math.max(10, Math.min(200, Number(e.target.value) || 30)))} style={{ width: 80 }} />
        </label>
        <button onClick={() => load(null, limit)} style={{ background: '#22c55e', color: '#111', border: 'none', padding: '6px 10px', borderRadius: 6, cursor: 'pointer', fontWeight: 600 }}>Refresh</button>
      </div>

      {loading && <div>Loading...</div>}
      {error && <div style={{ color: 'red' }}>Error: {error}</div>}

      {!loading && !error && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 12 }}>
            {items.map((img) => (
              <div key={img.image_id} style={{ border: '1px solid #222', borderRadius: 8, padding: 8, background: '#0f0f0f' }}>
                {img.image_path ? (
                  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                  // @ts-ignore
                  <img src={`${getApiBase()}${img.image_path}`} alt={img.image_id} style={{ width: '100%', height: 140, objectFit: 'cover', borderRadius: 6, marginBottom: 8 }} />
                ) : img.thumb_path ? (
                  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                  // @ts-ignore
                  <img src={`${getApiBase()}${img.thumb_path}`} alt={img.image_id} style={{ width: '100%', height: 140, objectFit: 'cover', borderRadius: 6, marginBottom: 8 }} />
                ) : (
                  <div style={{ height: 140, background: '#111', borderRadius: 6, marginBottom: 8 }} />
                )}
                <div style={{ fontWeight: 700, fontSize: 12 }}>{img.image_id}</div>
                <div style={{ color: '#9ca3af', fontSize: 12 }}>{img.created_at || ''}</div>
                {img.source && <div style={{ color: '#6b7280', fontSize: 12 }}>src: {img.source}</div>}
                {/* Future: add delete image button */}
              </div>
            ))}
            {items.length === 0 && <div>No images.</div>}
          </div>
          <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
            <button disabled={!cursor} onClick={() => load(cursor, limit)} style={{ background: '#0ea5e9', color: '#111', border: 'none', padding: '6px 10px', borderRadius: 6, cursor: 'pointer', opacity: cursor ? 1 : 0.5 }}>Next</button>
          </div>
        </>
      )}
    </div>
  );
}
