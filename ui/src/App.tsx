import { Routes, Route } from 'react-router-dom';
import AppShell from './layouts/AppShell';
import Dashboard from './pages/Dashboard';
import Enroll from './pages/Enroll';
import Search from './pages/Search';
import Subjects from './pages/Subjects';
import SubjectDetail from './pages/SubjectDetail';
import Settings from './pages/Settings';
import Rejections from './pages/Rejections';
import Recognition from './pages/Recognition';

function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Dashboard />} />
        <Route path="/enroll" element={<Enroll />} />
        <Route path="/search" element={<Search />} />
        <Route path="/recognition" element={<Recognition />} />
        <Route path="/rejections" element={<Rejections />} />
        <Route path="/subjects" element={<Subjects />} />
        <Route path="/subjects/:id" element={<SubjectDetail />} />
        <Route path="/settings" element={<Settings />} />
      </Route>
    </Routes>
  );
}

export default App
