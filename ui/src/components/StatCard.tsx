import React from 'react';

export default function StatCard({ title, value, tone = 'default', hint }: { title: string; value: React.ReactNode; tone?: 'default' | 'good' | 'warn' | 'bad'; hint?: string }) {
  const getColors = () => {
    switch (tone) {
      case 'good': return { bg: 'rgba(16, 185, 129, 0.05)', border: 'rgba(16, 185, 129, 0.2)', text: 'var(--success)' };
      case 'warn': return { bg: 'rgba(245, 158, 11, 0.05)', border: 'rgba(245, 158, 11, 0.2)', text: 'var(--warning)' };
      case 'bad': return { bg: 'rgba(239, 68, 68, 0.05)', border: 'rgba(239, 68, 68, 0.2)', text: 'var(--error)' };
      default: return { bg: 'var(--bg-primary)', border: 'var(--border)', text: 'var(--text-primary)' };
    }
  };

  const colors = getColors();

  return (
    <div className="card" style={{ 
      background: colors.bg, 
      borderColor: colors.border,
      display: 'flex',
      flexDirection: 'column',
      gap: 4
    }}>
      <div style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', fontWeight: 500 }}>{title}</div>
      <div style={{ fontSize: '1.5rem', fontWeight: 700, color: colors.text === 'var(--text-primary)' ? 'var(--text-primary)' : colors.text }}>
        {value}
      </div>
      {hint && <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginTop: 4 }}>{hint}</div>}
    </div>
  );
}
