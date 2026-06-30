import { useEffect, useMemo, useState } from 'react';
import { getApiBase, getBranches, recognitionFeedbackStats, stats, type FeedbackStatsResponse, type Stats } from '../lib/api';
import StatCard from '../components/StatCard';

export default function Dashboard() {
  const [data, setData] = useState<Stats | null>(null);
  const [fb, setFb] = useState<FeedbackStatsResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [employeeStats, setEmployeeStats] = useState<{ total: number; byBranch: Record<string, number> }>({ total: 0, byBranch: {} });

  async function fetchStats() {
    setLoading(true);
    setErr(null);
    try {
      const [s, f, b] = await Promise.all([
        stats(),
        recognitionFeedbackStats({ 
          since_ts: Math.floor(Date.now() / 1000) - 24 * 3600, 
          until_ts: Math.floor(Date.now() / 1000) 
        }),
        getBranches()
      ]);
      
      setData(s);
      setFb(f);

      const byBranch: Record<string, number> = {};
      let totalEmps = 0;
      for (const branch of b.branches || []) {
        const count = branch.subject_count || 0;
        byBranch[branch.name || branch.branch_id] = count;
        totalEmps += count;
      }
      setEmployeeStats({ total: totalEmps, byBranch });

    } catch (e: any) {
      setErr(String(e));
      setData(null);
      setFb(null);
    } finally {
      setLoading(false);
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

  const labeledPct = fb && fb.total > 0 ? `${((fb.labeled / fb.total) * 100).toFixed(1)}%` : '—';
  const fpRate = fb && fb.fp_rate_match != null ? `${(fb.fp_rate_match * 100).toFixed(2)}%` : '—';

  const sortedBranches = useMemo(() => {
    return Object.entries(employeeStats.byBranch).sort((a, b) => b[1] - a[1]);
  }, [employeeStats.byBranch]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h2 style={{ fontSize: '1.875rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '4px' }}>Dashboard Overview</h2>
          <div style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>API Base: <code style={{ background: 'var(--bg-secondary)', padding: '2px 4px', borderRadius: 4 }}>{getApiBase()}</code></div>
        </div>
        <button onClick={fetchStats} className="primary" style={{ height: '40px' }}>
          {loading ? <><span className="spinner" />Refreshing...</> : 'Refresh Stats'}
        </button>
      </header>

      {err && (
        <div style={{ padding: '16px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--error)', borderRadius: 'var(--radius-md)', color: 'var(--error)', fontWeight: 500 }}>
          {err}
        </div>
      )}

      <section>
        <h3 style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          System Health
          <span style={{ fontSize: '0.75rem', padding: '2px 8px', borderRadius: '99px', background: err ? 'var(--error)' : 'var(--success)', color: 'white' }}>
            {err ? 'DEGRADED' : 'OPERATIONAL'}
          </span>
        </h3>
        <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))' }}>
          <StatCard title="Face Service" value={err ? 'Offline' : 'Online'} tone={err ? 'bad' : 'good'} hint="Core API Status" />
          <StatCard title="Qdrant Vector DB" value={loading ? 'Checking...' : (data ? (data.qdrant_enabled ? `Connected` : 'Disconnected') : 'N/A')} tone={loading ? 'default' : (data && data.qdrant_enabled ? 'good' : 'bad')} hint={data?.qdrant_collection ? `Collection: ${data.qdrant_collection}` : 'Vector storage connection'} />
          <StatCard title="System Clock" value={new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} hint="Local server time" />
        </div>
      </section>

      <section>
        <h3 style={{ marginBottom: '16px' }}>Database Snapshot</h3>
        <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
          <StatCard title="Total Subjects" value={loading ? '…' : (data ? data.subjects_total.toLocaleString() : '—')} />
          <StatCard title="Total Employees" value={loading ? '…' : employeeStats.total.toLocaleString()} tone="good" hint="Subjects starting with 'employee-'" />
          <StatCard title="Total Embeddings" value={loading ? '…' : (data ? data.embeddings_total.toLocaleString() : '—')} />
          <StatCard title="Enrollments (24h)" value={loading ? '…' : (data ? data.last_24h_enrolls : '—')} />
          <StatCard title="Searches (24h)" value={loading ? '…' : (data ? data.last_24h_searches : '—')} />
        </div>
      </section>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '32px' }}>
        <section>
          <h3 style={{ marginBottom: '16px' }}>Branch-wise Employee Count</h3>
          <div className="card" style={{ padding: '0', overflow: 'hidden' }}>
            {loading ? (
              <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>Calculating...</div>
            ) : sortedBranches.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                {sortedBranches.map(([b, count], idx: number) => (
                  <div key={b} style={{ display: 'flex', justifyContent: 'space-between', padding: '12px 20px', background: idx % 2 === 0 ? 'transparent' : 'var(--bg-secondary)', borderBottom: idx === sortedBranches.length - 1 ? 'none' : '1px solid var(--border)' }}>
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{b}</span>
                    <span style={{ fontWeight: 800, color: 'var(--primary)' }}>{count}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>No branch data available</div>
            )}
          </div>
        </section>

        <section>
          <h3 style={{ marginBottom: '16px' }}>Performance Metrics (Local)</h3>
          <div className="grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <StatCard title="Match Rate" value={matchRate} hint="Since refresh" tone={matchRate !== '—' && parseFloat(matchRate) > 80 ? 'good' : 'default'} />
            <StatCard title="No-match Rate" value={noMatchRate} hint="Since refresh" />
          </div>
        </section>
      </div>

      <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '32px' }}>
        <section>
          <h3 style={{ marginBottom: '16px' }}>Ground Truth Quality (24h)</h3>
          <div className="grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <StatCard title="Labeled Coverage" value={loading ? '…' : labeledPct} hint={fb ? `${fb.labeled}/${fb.total} verified` : undefined} tone={labeledPct !== '—' && parseFloat(labeledPct) > 50 ? 'good' : 'warn'} />
            <StatCard title="False Positive Rate" value={loading ? '…' : fpRate} hint="Incorrect match rate" tone={fpRate !== '—' && parseFloat(fpRate) < 1 ? 'good' : 'bad'} />
          </div>
        </section>

        <section>
          <h3 style={{ marginBottom: '16px' }}>Verification Counts (24h)</h3>
          <div className="grid" style={{ gridTemplateColumns: 'repeat(2, 1fr)' }}>
            <StatCard title="True Positives" value={loading ? '…' : (fb ? fb.tp : '0')} />
            <StatCard title="False Positives" value={loading ? '…' : (fb ? fb.fp : '0')} tone={fb && fb.fp > 0 ? 'bad' : 'default'} />
          </div>
        </section>
      </div>
    </div>
  );
}
