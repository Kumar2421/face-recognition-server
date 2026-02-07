import React from 'react';

export default function StatCard({ title, value, tone = 'default', hint }: { title: string; value: React.ReactNode; tone?: 'default' | 'good' | 'warn' | 'bad'; hint?: string }) {
  const border = tone === 'good' ? '#14532d' : tone === 'warn' ? '#78350f' : tone === 'bad' ? '#7f1d1d' : '#222';
  const bg = tone === 'good' ? '#052e16' : tone === 'warn' ? '#111827' : tone === 'bad' ? '#0b0f1a' : '#0f0f0f';
  return (
    <div style={{ border: `1px solid ${border}`, borderRadius: 10, padding: 12, background: bg }}>
      <div style={{ color: '#9ca3af', fontSize: 12 }}>{title}</div>
      <div style={{ fontSize: 24, fontWeight: 700 }}>{value}</div>
      {hint && <div style={{ color: '#6b7280', fontSize: 12, marginTop: 6 }}>{hint}</div>}
    </div>
  );
}
