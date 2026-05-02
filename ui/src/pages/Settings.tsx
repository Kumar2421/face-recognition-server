import { useEffect, useState } from 'react';
import { getApiBase, getApiKey } from '../lib/api';

export default function Settings() {
  const [apiBase, setApiBase] = useState<string>('');
  const [apiKey, setApiKey] = useState<string>('');
  const [saved, setSaved] = useState<string>('');

  useEffect(() => {
    setApiBase(getApiBase());
    setApiKey(getApiKey());
  }, []);

  function save() {
    try {
      localStorage.setItem('api_base', apiBase.trim());
      localStorage.setItem('api_key', apiKey.trim());
      setSaved('Saved. Reloading...');
      setTimeout(() => window.location.reload(), 600);
    } catch (e) {
      setSaved(String(e));
    }
  }

  function clearOverride() {
    try {
      localStorage.removeItem('api_base');
      localStorage.removeItem('api_key');
      setSaved('Cleared overrides. Reloading...');
      setTimeout(() => window.location.reload(), 600);
    } catch (e) {
      setSaved(String(e));
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      <header>
        <h2 style={{ fontSize: '1.875rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '4px' }}>System Settings</h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9375rem' }}>Configure API endpoints and local dashboard preferences.</p>
      </header>

      <div className="card" style={{ display: 'grid', gap: '24px', maxWidth: '600px', background: 'var(--bg-primary)' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <label style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-secondary)' }}>API Base URL</label>
          <input
            value={apiBase}
            onChange={(e) => setApiBase(e.target.value)}
            placeholder="http://localhost:8001"
            style={{ padding: '12px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: '1rem' }}
          />
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>The URL where the Face Service API is hosted. Usually http://localhost:8001 during development.</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <label style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-secondary)' }}>API Key (x-api-key)</label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="fs_..."
            style={{ padding: '12px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: '1rem' }}
          />
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Required for security enabled environments. Matches FACE_SERVICE_API_KEY.</span>
        </div>

        <div style={{ display: 'flex', gap: '12px', borderTop: '1px solid var(--border)', paddingTop: '20px' }}>
          <button onClick={save} className="primary" style={{ flex: 1, padding: '12px', fontWeight: 700 }}>Save Changes</button>
          <button onClick={clearOverride} style={{ flex: 1, padding: '12px', fontWeight: 600 }}>Clear Override</button>
        </div>

        {saved && (
          <div style={{ 
            padding: '12px', 
            borderRadius: 'var(--radius-md)', 
            background: 'rgba(16, 185, 129, 0.1)', 
            color: 'var(--success)', 
            fontSize: '0.875rem', 
            fontWeight: 600,
            textAlign: 'center'
          }}>
            {saved}
          </div>
        )}
      </div>

      <section className="card" style={{ maxWidth: '600px', border: '1px solid var(--warning)', background: 'rgba(245, 158, 11, 0.05)' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--warning)', marginBottom: '8px' }}>Environment Information</h3>
        <div style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)', display: 'grid', gridTemplateColumns: '120px 1fr', gap: '8px' }}>
          <span>Current Base:</span>
          <code style={{ color: 'var(--text-primary)' }}>{getApiBase()}</code>
          <span>API Key:</span>
          <span>{getApiKey() ? '******** (Set)' : 'Not Set'}</span>
          <span>Storage:</span>
          <span>{localStorage.getItem('api_base') || localStorage.getItem('api_key') ? 'Local Storage (Overridden)' : 'Default (.env)'}</span>
        </div>
      </section>
    </div>
  );
}
