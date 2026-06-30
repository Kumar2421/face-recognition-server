import { useEffect, useRef, useState } from 'react';
import { facesAddUpload } from '../lib/api';

export default function Enroll() {
  const [subjectId, setSubjectId] = useState('');
  const [status, setStatus] = useState<string>('');
  const [error, setError] = useState<string>('');
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [previews, setPreviews] = useState<Array<{ url: string; name: string; sizeKB: number; w: number; h: number }>>([]);
  const [results, setResults] = useState<Array<{ name: string; hash: string; status: 'enrolled' | 'failed'; reason?: string }>>([]);
  const [loading, setLoading] = useState<boolean>(false);

  function revokePreviews() {
    for (const p of previews) URL.revokeObjectURL(p.url);
  }

  async function fileHash(file: File): Promise<string> {
    const buf = await file.arrayBuffer();
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore
    const digest = await crypto.subtle.digest('SHA-256', buf);
    const arr = Array.from(new Uint8Array(digest));
    return arr.map((b) => b.toString(16).padStart(2, '0')).join('');
  }

  async function onFilesChanged() {
    setResults([]);
    revokePreviews();
    const files = Array.from(fileRef.current?.files || []);
    const pv: Array<{ url: string; name: string; sizeKB: number; w: number; h: number }> = [];
    await Promise.all(
      files.map(async (f) => {
        const url = URL.createObjectURL(f);
        const dims = await new Promise<{ w: number; h: number }>((resolve) => {
          const img = new Image();
          img.onload = () => resolve({ w: img.width, h: img.height });
          img.onerror = () => resolve({ w: 0, h: 0 });
          // eslint-disable-next-line @typescript-eslint/ban-ts-comment
          // @ts-ignore
          img.src = url;
        });
        pv.push({ url, name: f.name, sizeKB: Math.round(f.size / 1024), w: dims.w, h: dims.h });
      })
    );
    setPreviews(pv);
  }

  useEffect(() => {
    return () => revokePreviews();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setStatus('');
    setError('');
    const files = Array.from(fileRef.current?.files || []);
    if (!subjectId.trim()) {
      setError('subject_id is required');
      return;
    }
    if (files.length === 0) {
      setError('Select at least one image');
      return;
    }
    setLoading(true);
    const res: Array<{ name: string; hash: string; status: 'enrolled' | 'failed'; reason?: string }> = [];
    try {
      for (const f of files) {
        try {
          const r = await facesAddUpload(subjectId.trim(), [f]);
          const h = await fileHash(f);
          const ok = (r?.num_embedded || 0) > 0;
          res.push({ name: f.name, hash: h.slice(0, 16), status: ok ? 'enrolled' : 'failed', reason: ok ? undefined : 'no face / low quality' });
        } catch (err: any) {
          const h = await fileHash(f);
          const msg = String(err?.message || err || 'enroll failed');
          res.push({ name: f.name, hash: h.slice(0, 16), status: 'failed', reason: msg.includes('404') ? 'no face / low quality' : 'server error' });
        }
      }
      setResults(res);
      const okCount = res.filter((x) => x.status === 'enrolled').length;
      setStatus(`Processed ${files.length} image(s). Enrolled: ${okCount}, Skipped/Failed: ${files.length - okCount}.`);
      if (fileRef.current) fileRef.current.value = '';
    } catch (e: any) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      <header>
        <h2 style={{ fontSize: '1.875rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '4px' }}>Subject Enrollment</h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9375rem' }}>Upload images to create or extend face recognition profiles.</p>
      </header>

      <form onSubmit={onSubmit} className="card" style={{ display: 'grid', gap: '24px', maxWidth: '600px', background: 'var(--bg-primary)' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <label style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Subject Unique Identifier</label>
          <input
            value={subjectId}
            onChange={(e) => setSubjectId(e.target.value)}
            placeholder="e.g. john_doe_123"
            style={{ padding: '12px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: '1rem' }}
          />
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Use a unique string to identify this person across the system.</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <label style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Face Images</label>
          <div className="dropzone" style={{
            border: '2px dashed var(--border)',
            borderRadius: 'var(--radius-lg)',
            padding: '40px',
            textAlign: 'center',
            background: 'var(--bg-secondary)',
            cursor: 'pointer'
          }} onClick={() => fileRef.current?.click()}>
            <input type="file" ref={fileRef} multiple accept="image/*" onChange={onFilesChanged} style={{ display: 'none' }} />
            <div style={{ fontSize: '2rem', marginBottom: '8px' }}>📸</div>
            <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Click to upload images</div>
            <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginTop: '4px' }}>PNG, JPG or WEBP up to 10MB each</div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '12px', marginTop: '8px' }}>
          <button type="submit" className="primary" style={{ padding: '12px 32px', flex: 1, fontWeight: 700 }}>
            {loading ? 'Enrolling...' : 'Start Enrollment'}
          </button>
          <button 
            type="button" 
            onClick={() => { setSubjectId(''); setStatus(''); setError(''); setResults([]); revokePreviews(); setPreviews([]); if (fileRef.current) fileRef.current.value = ''; }}
            style={{ padding: '12px 24px', fontWeight: 600 }}
          >
            Reset Form
          </button>
        </div>
      </form>

      {status && (
        <div style={{ padding: '16px', background: 'rgba(16, 185, 129, 0.1)', border: '1px solid var(--success)', borderRadius: 'var(--radius-md)', color: 'var(--success)', fontWeight: 600 }}>
          ✓ {status}
        </div>
      )}
      
      {error && (
        <div style={{ padding: '16px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--error)', borderRadius: 'var(--radius-md)', color: 'var(--error)', fontWeight: 600 }}>
          ⚠ Error: {error}
        </div>
      )}

      {previews.length > 0 && (
        <section>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '16px', color: 'var(--text-primary)' }}>Selected Images ({previews.length})</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '20px' }}>
            {previews.map((p, i) => (
              <div key={i} className="card" style={{ padding: '8px' }}>
                <img src={p.url} alt={p.name} style={{ width: '100%', height: '140px', objectFit: 'cover', borderRadius: 'var(--radius-md)', marginBottom: '8px' }} />
                <div style={{ fontSize: '0.8125rem', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.name}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '2px' }}>{p.sizeKB} KB • {p.w}×{p.h}</div>
              </div>
            ))}
          </div>
        </section>
      )}

      {results.length > 0 && (
        <section>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '16px', color: 'var(--text-primary)' }}>Enrollment Results</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '20px' }}>
            {results.map((r, i) => (
              <div key={i} className="card" style={{ 
                borderLeft: `4px solid ${r.status === 'enrolled' ? 'var(--success)' : 'var(--error)'}`,
                display: 'flex',
                flexDirection: 'column',
                gap: '8px'
              }}>
                <div style={{ fontWeight: 700, fontSize: '0.875rem' }}>{r.name}</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ 
                    fontSize: '0.625rem', 
                    fontWeight: 800, 
                    padding: '2px 8px', 
                    borderRadius: '4px', 
                    background: r.status === 'enrolled' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                    color: r.status === 'enrolled' ? 'var(--success)' : 'var(--error)'
                  }}>
                    {r.status.toUpperCase()}
                  </span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ID: {r.hash}</span>
                </div>
                {r.reason && <div style={{ fontSize: '0.75rem', color: 'var(--error)', fontStyle: 'italic' }}>{r.reason}</div>}
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
