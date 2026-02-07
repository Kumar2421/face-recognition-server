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
    <div>
      <h2>Enroll</h2>
      <form onSubmit={onSubmit} style={{ display: 'grid', gap: 12, maxWidth: 720 }}>
        <label style={{ display: 'grid', gap: 6 }}>
          <span>Subject ID</span>
          <input
            value={subjectId}
            onChange={(e) => setSubjectId(e.target.value)}
            placeholder="e.g. john_doe"
            style={{ padding: '8px 10px', borderRadius: 8, border: '1px solid #333', background: '#0a0a0a', color: '#e5e5e5' }}
          />
        </label>
        <label style={{ display: 'grid', gap: 6 }}>
          <span>Images</span>
          <input type="file" ref={fileRef} multiple accept="image/*" onChange={onFilesChanged} />
        </label>
        <div style={{ display: 'flex', gap: 12 }}>
          <button type="submit" style={{ background: '#22c55e', color: '#111', border: 'none', padding: '8px 12px', borderRadius: 8, cursor: 'pointer', fontWeight: 600 }}>Enroll</button>
          <button type="button" onClick={() => { setSubjectId(''); setStatus(''); setError(''); setResults([]); revokePreviews(); setPreviews([]); if (fileRef.current) fileRef.current.value = ''; }} style={{ background: '#374151', color: '#fff', border: 'none', padding: '8px 12px', borderRadius: 8, cursor: 'pointer' }}>Reset</button>
        </div>
      </form>
      {loading && <div style={{ marginTop: 12 }}>Uploading & enrolling...</div>}
      {status && <div style={{ marginTop: 12, color: '#22c55e' }}>{status}</div>}
      {error && <div style={{ marginTop: 12, color: '#ef4444' }}>Error: {error}</div>}

      {previews.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <h3>Preview</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 12 }}>
            {previews.map((p, i) => (
              <div key={i} style={{ border: '1px solid #222', borderRadius: 8, padding: 8, background: '#0f0f0f' }}>
                {/* eslint-disable-next-line @typescript-eslint/ban-ts-comment */}
                {/* @ts-ignore */}
                <img src={p.url} alt={p.name} style={{ width: '100%', height: 120, objectFit: 'cover', borderRadius: 6, marginBottom: 8 }} />
                <div style={{ fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis' }}>{p.name}</div>
                <div style={{ color: '#9ca3af', fontSize: 12 }}>Size: {p.sizeKB} KB</div>
                <div style={{ color: '#9ca3af', fontSize: 12 }}>Resolution: {p.w}×{p.h} {Math.min(p.w, p.h) < 128 ? <span style={{ color: '#f59e0b' }}>(low)</span> : null}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {results.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <h3>Results</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 12 }}>
            {results.map((r, i) => (
              <div key={i} style={{ border: '1px solid #222', borderRadius: 8, padding: 8, background: '#0f0f0f' }}>
                <div style={{ fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis' }}>{r.name}</div>
                <div style={{ color: '#9ca3af', fontSize: 12 }}>Hash: {r.hash}</div>
                <div style={{ marginTop: 6 }}>
                  <span style={{
                    display: 'inline-block',
                    padding: '2px 6px',
                    borderRadius: 6,
                    background: r.status === 'enrolled' ? '#14532d' : '#7f1d1d',
                    color: '#e5e7eb',
                    fontSize: 12,
                    fontWeight: 700,
                  }}>
                    {r.status === 'enrolled' ? 'ENROLLED' : 'FAILED'}
                  </span>
                </div>
                {r.reason && <div style={{ color: '#9ca3af', fontSize: 12, marginTop: 6 }}>Reason: {r.reason}</div>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
