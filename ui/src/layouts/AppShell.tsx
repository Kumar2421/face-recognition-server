import type { CSSProperties } from 'react';
import { Link, NavLink, Outlet } from 'react-router-dom';

const nav = [
  { to: '/', label: 'Dashboard' },
  { to: '/enroll', label: 'Enroll' },
  { to: '/search', label: 'Search' },
  { to: '/recognition', label: 'Recognition' },
  { to: '/rejections', label: 'Rejections' },
  { to: '/subjects', label: 'Subjects' },
  { to: '/settings', label: 'Settings' },
];

export default function AppShell() {
  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: '#0a0a0a', color: '#e5e5e5' }}>
      <aside style={{ width: 240, borderRight: '1px solid #222', padding: 16 }}>
        <Link to="/" style={{ textDecoration: 'none', color: '#fff' }}>
          <h1 style={{ margin: '0 0 16px 0' }}>Face Service</h1>
        </Link>
        <nav style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {nav.map(item => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              style={({ isActive }: { isActive: boolean }): CSSProperties => ({
                padding: '8px 12px',
                borderRadius: 8,
                color: isActive ? '#111' : '#e5e5e5',
                background: isActive ? '#a3e635' : 'transparent',
                textDecoration: 'none',
                fontWeight: 600
              })}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main style={{ flex: 1, padding: 24 }}>
        <Outlet />
      </main>
    </div>
  );
}
