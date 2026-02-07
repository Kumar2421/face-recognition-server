import { useEffect, useState } from 'react';
import { getApiBase } from '../lib/api';

export default function Settings() {
  const [apiBase, setApiBase] = useState<string>('');
  const [saved, setSaved] = useState<string>('');

  useEffect(() => {
    setApiBase(getApiBase());
  }, []);

  function save() {
    try {
      localStorage.setItem('api_base', apiBase.trim());
      setSaved('Saved. Reloading...');
      setTimeout(() => window.location.reload(), 600);
    } catch (e) {
      setSaved(String(e));
    }
  }

  function clearOverride() {
    try {
      localStorage.removeItem('api_base');
      setSaved('Cleared override. Reloading...');
      setTimeout(() => window.location.reload(), 600);
    } catch (e) {
      setSaved(String(e));
    }
  }

  return (
    <div>
      <h2>Settings</h2>
      <div style={{ display: 'grid', gap: 12, maxWidth: 560 }}>
        <label style={{ display: 'grid', gap: 6 }}>
          <span>API Base URL</span>
          <input
            value={apiBase}
            onChange={(e) => setApiBase(e.target.value)}
            placeholder="http://localhost:8001"
            style={{ padding: '8px 10px', borderRadius: 8, border: '1px solid #333', background: '#0a0a0a', color: '#e5e5e5' }}
          />
        </label>
        <div style={{ display: 'flex', gap: 12 }}>
          <button onClick={save} style={{ background: '#22c55e', color: '#111', border: 'none', padding: '8px 12px', borderRadius: 8, cursor: 'pointer', fontWeight: 600 }}>Save</button>
          <button onClick={clearOverride} style={{ background: '#374151', color: '#fff', border: 'none', padding: '8px 12px', borderRadius: 8, cursor: 'pointer' }}>Clear Override</button>
        </div>
        {saved && <div style={{ color: '#a3e635' }}>{saved}</div>}
      </div>
    </div>
  );
}
