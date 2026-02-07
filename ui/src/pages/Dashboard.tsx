import { useEffect, useState } from 'react';
import { getApiBase, stats, type Stats } from '../lib/api';
import StatCard from '../components/StatCard';

export default function Dashboard() {
  const [data, setData] = useState<Stats | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  async function fetchStats() {
    setLoading(true);
    setErr(null);
    try {
      const s = await stats();
      setData(s);
    } catch (e: any) {
      setErr(String(e));
      setData(null);
    } finally {
      setTimeout(() => {
        setLoading(false);
      }, 5000);
    }
  }

  useEffect(() => {
    fetchStats();
  }, []);

  // read match/no-match counters since last refresh from localStorage
  let uiMatches = 0;
  let uiNoMatches = 0;
  try {
    uiMatches = parseInt(localStorage.getItem('ui_match_count') || '0', 10) || 0;
    uiNoMatches = parseInt(localStorage.getItem('ui_nomatch_count') || '0', 10) || 0;
  } catch { }
  const uiTotal = uiMatches + uiNoMatches;
  const matchRate = uiTotal > 0 ? `${((uiMatches / uiTotal) * 100).toFixed(1)}%` : '—';
  const noMatchRate = uiTotal > 0 ? `${((uiNoMatches / uiTotal) * 100).toFixed(1)}%` : '—';

  return (
    <div>
      <h2>Dashboard</h2>
      <div style={{ marginBottom: 12, color: '#9ca3af' }}>API Base: {getApiBase()}</div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        <button onClick={fetchStats} style={{ background: '#22c55e', color: '#111', border: 'none', padding: '6px 10px', borderRadius: 8, cursor: 'pointer', fontWeight: 600 }}>Refresh</button>
      </div>
      {err && <div style={{ color: '#ef4444', marginBottom: 12 }}>Error: {err}</div>}

      <h3 style={{ marginTop: 8, marginBottom: 8 }}>System Health</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
        <StatCard title="Face Service" value={err ? 'Degraded' : 'OK'} tone={err ? 'warn' : 'good'} hint="Status" />
        <StatCard title="Qdrant" value={loading ? '…' : (data ? (data.qdrant_enabled ? `Connected` : 'Disconnected') : '—')} tone={loading ? 'default' : (data && data.qdrant_enabled ? 'good' : 'bad')} hint={data?.qdrant_collection ? `Collection: ${data.qdrant_collection}` : undefined} />
        <StatCard title="Last Request Time" value={new Date().toLocaleTimeString()} />
      </div>

      <h3 style={{ marginTop: 16, marginBottom: 8 }}>Last 24h Snapshot</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
        <StatCard title="Total Subjects" value={loading ? '…' : (data ? data.subjects_total : '—')} />
        <StatCard title="Total Embeddings" value={loading ? '…' : (data ? data.embeddings_total : '—')} />
        <StatCard title="Enrollments (24h)" value={loading ? '…' : (data ? data.last_24h_enrolls : '—')} />
        <StatCard title="Searches (24h)" value={loading ? '…' : (data ? data.last_24h_searches : '—')} />
      </div>

      <h3 style={{ marginTop: 16, marginBottom: 8 }}>Since Last Refresh</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
        <StatCard title="Match Rate" value={matchRate} hint="from Recognize results" />
        <StatCard title="No-match Rate" value={noMatchRate} hint="from Recognize results" />
      </div>
    </div>
  );
}
