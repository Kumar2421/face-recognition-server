import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { getApiBase, getSubject, subjectImages, type DateFilter, type SubjectImageItem, type SubjectItem } from '../lib/api';

export default function SubjectDetail() {
  const { id } = useParams<{ id: string }>();
  const subjectId = decodeURIComponent(id || '');
  const [items, setItems] = useState<SubjectImageItem[]>([]);
  const [subject, setSubject] = useState<SubjectItem | null>(null);
  const [cursor, setCursor] = useState<string | null>(null);
  const [limit, setLimit] = useState<number>(30);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [dateMode, setDateMode] = useState<'all' | 'day' | 'range'>('all');
  const [day, setDay] = useState<string>('');
  const [fromDay, setFromDay] = useState<string>('');
  const [toDay, setToDay] = useState<string>('');

  function currentFilter(): DateFilter {
    return {
      day: dateMode === 'day' ? (day || null) : null,
      from_day: dateMode === 'range' ? (fromDay || null) : null,
      to_day: dateMode === 'range' ? (toDay || null) : null,
    };
  }

  async function load(c: string | null, l: number) {
    if (!subjectId) return;
    setLoading(true);
    setError(null);
    try {
      const r = await subjectImages(subjectId, { cursor: c || undefined, limit: l, ...currentFilter() });
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
    (async () => {
      if (!subjectId) return;
      try {
        const s = await getSubject(subjectId);
        setSubject(s);
      } catch {
        setSubject(null);
      }
    })();
    load(null, limit);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [subjectId, limit, dateMode, day, fromDay, toDay]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      <header>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '12px' }}>
          <Link 
            to="/subjects" 
            style={{ 
              color: 'var(--text-secondary)', 
              textDecoration: 'none',
              fontSize: '0.875rem',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}
          >
            ← Back to Subjects
          </Link>
        </div>
        
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <h2 style={{ fontSize: '2.25rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '8px' }}>{subjectId}</h2>
            {subject && (
              <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                <span style={{ 
                  padding: '4px 12px', 
                  background: 'var(--bg-secondary)', 
                  borderRadius: '99px', 
                  fontSize: '0.75rem', 
                  fontWeight: 700, 
                  color: 'var(--text-secondary)',
                  border: '1px solid var(--border)'
                }}>
                  {typeof subject.embeddings_cap === 'number' ? `${subject.embeddings_count} / ${subject.embeddings_cap} Embeddings` : `${subject.embeddings_count} Embeddings`}
                </span>
                {subject.embeddings_capped && (
                  <span style={{ fontSize: '0.75rem', fontWeight: 800, color: 'var(--warning)', textTransform: 'uppercase' }}>⚠ Capped</span>
                )}
              </div>
            )}
          </div>
          
          <div className="card" style={{ display: 'flex', gap: '16px', alignItems: 'center', padding: '12px 20px', background: 'var(--bg-primary)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-muted)' }}>Show</span>
              <input 
                type="number" 
                min={10} 
                max={200} 
                value={limit} 
                onChange={(e) => setLimit(Math.max(10, Math.min(200, Number(e.target.value) || 30)))} 
                style={{ width: '60px', padding: '6px', borderRadius: '4px', border: '1px solid var(--border)', background: 'var(--bg-secondary)' }} 
              />
            </div>
            
            <select value={dateMode} onChange={(e) => setDateMode(e.target.value as any)} style={{ padding: '6px 12px', borderRadius: '4px', border: '1px solid var(--border)', background: 'var(--bg-secondary)' }}>
              <option value="all">All Dates</option>
              <option value="day">Single Day</option>
              <option value="range">Range</option>
            </select>

            {dateMode === 'day' && (
              <input type="date" value={day} onChange={(e) => setDay(e.target.value)} style={{ padding: '6px', borderRadius: '4px', border: '1px solid var(--border)', background: 'var(--bg-secondary)' }} />
            )}

            {dateMode === 'range' && (
              <div style={{ display: 'flex', gap: '4px' }}>
                <input type="date" value={fromDay} onChange={(e) => setFromDay(e.target.value)} style={{ padding: '6px', borderRadius: '4px', border: '1px solid var(--border)', background: 'var(--bg-secondary)' }} />
                <input type="date" value={toDay} onChange={(e) => setToDay(e.target.value)} style={{ padding: '6px', borderRadius: '4px', border: '1px solid var(--border)', background: 'var(--bg-secondary)' }} />
              </div>
            )}

            <button onClick={() => load(null, limit)} className="primary" style={{ padding: '8px 16px', fontSize: '0.875rem' }}>Refresh</button>
          </div>
        </div>
      </header>

      {error && (
        <div style={{ padding: '16px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--error)', borderRadius: 'var(--radius-md)', color: 'var(--error)', fontWeight: 600 }}>
          {error}
        </div>
      )}

      <div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '20px' }}>
          {items.map((img) => (
            <div key={img.image_id} className="card" style={{ padding: '10px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{ height: '180px', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border)' }}>
                {img.image_path || img.thumb_path ? (
                  <img src={`${getApiBase()}${img.image_path || img.thumb_path}`} alt={img.image_id} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                ) : (
                  <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.75rem' }}>No Image</div>
                )}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <div style={{ fontWeight: 700, fontSize: '0.75rem', color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{img.image_id}</div>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.6875rem', fontWeight: 500 }}>{img.created_at || 'No timestamp'}</div>
                {img.source && (
                  <div style={{ marginTop: '8px', fontSize: '0.625rem', color: 'var(--text-muted)', wordBreak: 'break-all' }}>
                    SOURCE: {img.source}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {items.length === 0 && !loading && (
          <div style={{ textAlign: 'center', padding: '80px', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-lg)', border: '2px dashed var(--border)', color: 'var(--text-muted)' }}>
            No images found for this subject and filter criteria.
          </div>
        )}

        <div style={{ marginTop: '32px', display: 'flex', justifyContent: 'center', gap: '16px' }}>
          <button 
            disabled={!cursor || loading} 
            onClick={() => load(cursor, limit)} 
            className="primary"
            style={{ 
              padding: '12px 40px', 
              fontWeight: 700,
              opacity: !cursor || loading ? 0.5 : 1,
              cursor: !cursor || loading ? 'not-allowed' : 'pointer'
            }}
          >
            {loading ? 'Loading More...' : 'Load More Images'}
          </button>
        </div>
      </div>
    </div>
  );
}
