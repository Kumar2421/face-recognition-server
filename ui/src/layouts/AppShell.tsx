import { Link, NavLink, Outlet, useLocation } from 'react-router-dom';
import logo from '../assets/logo.png';

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
  const location = useLocation();
  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <Link to="/" style={{ textDecoration: 'none', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 12, padding: '0 8px' }}>
          <img src={logo} alt="Logo" style={{ width: 32, height: 32, objectFit: 'contain' }} />
          <h1 style={{ fontSize: '1.25rem', fontWeight: 700, letterSpacing: '-0.025em' }}>Face Service</h1>
        </Link>
        <nav style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {nav.map(item => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="app-main">
        <div key={location.pathname} className="app-main-inner route-fade">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
