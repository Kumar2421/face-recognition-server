import type { CSSProperties } from 'react';
import { Link, NavLink, Outlet } from 'react-router-dom';

const nav = [
  { to: '/', label: 'Dashboard' },
  { to: '/enroll', label: 'Enroll' },
  { to: '/search', label: 'Search' },
  { to: '/recognition', label: 'Recognition' },
  { to: '/events', label: 'Events' },
  { to: '/employees', label: 'Employees' },
  { to: '/rejections', label: 'Rejections' },
  { to: '/subjects', label: 'Subjects' },
  { to: '/settings', label: 'Settings' },
];

export default function AppShell() {
  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: 'var(--bg-secondary)' }}>
      <aside style={{ 
        width: 260, 
        borderRight: '1px solid var(--border)', 
        padding: '24px 16px',
        background: 'var(--bg-primary)',
        display: 'flex',
        flexDirection: 'column',
        gap: 32
      }}>
        <Link to="/" style={{ textDecoration: 'none', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 12, padding: '0 8px' }}>
          <img src="/src/assets/logo.png" alt="Logo" style={{ width: 32, height: 32, objectFit: 'contain' }} />
          <h1 style={{ fontSize: '1.25rem', fontWeight: 700, letterSpacing: '-0.025em' }}>Face Service</h1>
        </Link>
        <nav style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {nav.map(item => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              style={({ isActive }: { isActive: boolean }): CSSProperties => ({
                padding: '10px 14px',
                borderRadius: 'var(--radius-md)',
                color: isActive ? 'var(--primary)' : 'var(--text-secondary)',
                background: isActive ? 'rgba(37, 99, 235, 0.1)' : 'transparent',
                textDecoration: 'none',
                fontWeight: isActive ? 600 : 500,
                fontSize: '0.9375rem',
                transition: 'all 0.2s',
                display: 'flex',
                alignItems: 'center',
                gap: 10
              })}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main style={{ flex: 1, padding: '40px', overflowY: 'auto' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <Outlet />
        </div>
      </main>
    </div>
  );
}
