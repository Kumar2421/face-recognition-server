# Face Recognition Server — Repo Review & Optimization Backlog

> Review date: 2026-05-29 · Reviewer: code review pass · Scope: full repo, with a UI deep dive.
> This document is **findings only** — no source code was changed to produce it.

## Snapshot

- **Backend:** single FastAPI monolith `app.py` = **4,209 lines / 159 KB**, 41 routes, with a *partially* extracted `src/` package (`models`, `services`, `core`, `utils`) — `src/routes/` exists but is **empty**.
- **Frontend:** Vite + React 18 + react-router 6, 10 pages, **4,490 lines** of TSX. Largest: `Recognition.tsx` (788), `Employees.tsx` (647), `Search.tsx` (542).
- **Repo:** 378 tracked files, but ~39,000 `.jpg` on disk; thousands of employee face images are committed.

---

## 🔴 Critical — Security & Privacy (fix before anything else)

1. **Secret key leaked into logs.** `app.py:85`:
   ```python
   logger.warning(f"AUTH FAIL: '{header_value}' != '{expected}'")
   ```
   This prints the **real expected API key** in plaintext on every failed auth. Anyone with log access gets the key. Remove `expected` (and ideally the supplied value) from the log line.

2. **Insecure default API key.** `app.py:78` falls back to `"your-secret-key"` if `FACE_SERVICE_API_KEY` is unset. A missing env var silently makes the service accessible with a publicly-known key. Fail closed instead (refuse to start, or return 503).

3. **Non-constant-time key comparison.** `app.py:84` uses `!=`. Use `hmac.compare_digest` to avoid timing leaks.

4. **`.env` is committed to git** (`git ls-files` confirms it tracked). Even though it currently holds only `VITE_API_BASE`/`CORS_ALLOW_ORIGINS`, it must be in `.gitignore` and removed from history before any real secret lands there.

5. **Biometric PII committed to the repo.** `downloaded_images/` (thousands of `employee-emerald-*` face JPGs), plus `employees-temp ... .csv` and `Result - 1 ... .csv` are tracked. Faces are sensitive personal data — likely a compliance violation and clone bloat. Purge from history (`git filter-repo`) and gitignore.

6. **CORS `*` + `allow_credentials=True`** (`app.py:944-945`). This combination is invalid per spec and a security smell — when origins resolve to `["*"]`, credentials should be off, or origins should be explicit.

---

## 🟠 High — Architecture & Maintainability

7. **The monolith.** A 4,200-line `app.py` mixing routing, auth, model inference, Qdrant access, event storage, metrics, and image handling is hard to test/review. The `src/` scaffold shows refactoring intent — finish it: move the 41 routes into `src/routes/` (e.g. `faces.py`, `events.py`, `branches.py`, `search.py`) using `APIRouter`, and have `app.py` just wire them up.

8. **Local imports scattered in handlers.** `from src.services.events_store import SearchEvent` appears inside handlers at lines 1981, 2047, 2101, 3338. Hoist to module top — repeated runtime imports add overhead and hide dependencies.

9. **No tests anywhere.** No `tests/`, no `*.test.tsx`. For a service making identity decisions, even smoke tests on the auth dependency, similarity thresholds, and event-store queries would catch regressions. The scattered `benchmark.py`, `stress_benchmark.py`, `cross_check.py`, `scratch/` look like ad-hoc scripts, not a suite.

10. **Repo hygiene.** `__pycache__/` is committed (incl. `app.cpython-310.pyc` showing as modified in git status). `scratch/`, `testfolder/`, `input/`, `facefolder/`, `benchmark_report_*.json`, `poller_state.json` all look like working artifacts. Add to `.gitignore`.

---

## UI Deep Dive (`ui/`)

Stack is fine (React 18, Vite 6, RR6). Concrete issues:

11. **Tailwind is installed but never used.** `tailwindcss@4.1.18`, `autoprefixer`, `postcss` are in `package.json`, yet there's **no `tailwind.config`, no `postcss.config`, and no `@tailwind` import** in `index.css`. Instead there are **582 inline `style={{…}}` objects** across pages. Either adopt Tailwind (and delete the inline-style sprawl) or drop the three dead dependencies.

12. **Inline-style sprawl → no reuse, extra re-renders.** Recognition/Employees repeat the same select/input/card style object literals dozens of times. Every render reallocates these objects. Extract shared primitives (`<Card>`, `<Select>`, `<TextInput>`, status pill) or move to CSS classes (`.card`/`.primary` already exist in `index.css`).

13. **Broken asset path.** `AppShell.tsx:29` uses `src="/src/assets/logo.png"`. Works in dev but **breaks in production build** — Vite won't serve `/src/...` from `dist`. Import the asset (`import logo from '../assets/logo.png'`) so it gets hashed/bundled.

14. **Effect-driven over-fetching in `Recognition.tsx`.** Six `useEffect`s; changing `camera` or a date triggers `load()`, `loadStats()`, and `loadUniqueCount()` — three network calls, sometimes duplicated because an initial `load(true)` runs on mount *and* the filter effect fires on mount too. Consolidate into a single keyed fetch effect (or adopt TanStack Query) to dedupe and cache.

15. **Client-side filtering of server-paginated data is misleading.** `filtered` re-filters `items` by decision/camera/subject (`Recognition.tsx:284`) *after* the server already filtered them, and the grouping/duration logic runs only over the loaded page. With "Load More" pagination, durations and "unique counts" computed client-side won't reflect the full dataset — easy to show wrong numbers.

16. **Redundant cache-busting.** `q.set('_cb', Date.now())` (`api.ts:302`) plus `cache:'no-store'` plus `Cache-Control` headers is belt-and-suspenders; pick one. The `_cb` param also defeats any future CDN/HTTP caching.

17. **API key stored in `localStorage`** (`Settings.tsx:17`, `api.ts:11`) is readable by any XSS. Acceptable for an internal admin tool, but worth flagging given the sensitivity.

18. **Weak typing.** 53 uses of `any`/`as any` across the UI; `meta?: any` everywhere in `api.ts`. Tightening these would catch shape mismatches at compile time.

19. **No error boundary / inconsistent empty & loading states.** Errors surface as raw `String(e)` text dumps. A shared error/empty component would improve UX consistency across the 10 pages.

20. **`console.error` left in production paths** (Subjects, Recognition×2, Rejections, SearchEvents×2). Fine for debugging, noisy in prod.

---

## 🟡 Backend Optimization Opportunities

21. **Per-route auth repetition.** `dependencies=[Depends(get_api_key)]` is repeated on ~all routes — consider a router-level dependency or global auth middleware (with an allowlist for `/health`, `/metrics`) instead.

22. **`recognitionCameras({ limit: 50000 })`** (`Recognition.tsx:259`) pulls up to 50k camera rows to populate a dropdown. Add a dedicated distinct-cameras endpoint server-side (`SELECT DISTINCT`) and cap it.

23. **Index the events store.** `src/services/events_store.py` (671 lines) — verify SQLite queries backing `/v1/events/recognition`, `/stats`, `/feedback_stats` are indexed on `ts`, `camera`, `subject_id`, `decision`. The UI filters heavily on these; without indexes, stats endpoints degrade as events grow.

24. **Static file serving.** `/data/thumbs`, `/data/images`, `/data/events` served by FastAPI/uvicorn — fine for low traffic, but front with nginx/CDN for image-heavy dashboards (the Recognition table loads 2 images per row × up to 100 rows).

25. **`/api` prefix rewrite middleware** (`app.py:951`) mutates `request.scope["path"]` on every request — cheap, but a routing concern that belongs in the reverse proxy.

26. **Dependency reproducibility.** `requirements.txt` pins versions (good) but there's no hash-locked file; `onnxruntime-gpu`, `insightface`, `tflite-runtime` are heavy/fragile — consider `pip-tools`/`uv` lock for reproducible GPU builds.

---

## Suggested Priority Order

| When | Items |
| --- | --- |
| **Today** | Remove secret from log line (#1); fail-closed on missing API key (#2); untrack `.env` (#4) |
| **This week** | Purge biometric images + CSVs from git history (#5); `.gitignore` artifacts + `__pycache__` (#10); fix logo asset path (#13) |
| **Next** | Finish `src/routes` extraction (#7); fix CORS combo (#6); add DB indexes (#23); dedupe Recognition fetches (#14) |
| **Cleanup** | Decide Tailwind vs inline styles (#11/#12); tighten `any` types (#18); add a minimal test suite (#9) |

---

## Findings Index

| # | Severity | Area | Summary |
| --- | --- | --- | --- |
| 1 | 🔴 Critical | Backend/auth | Expected API key logged in plaintext |
| 2 | 🔴 Critical | Backend/auth | Insecure default API key fallback |
| 3 | 🔴 Critical | Backend/auth | Non-constant-time key comparison |
| 4 | 🔴 Critical | Repo | `.env` committed to git |
| 5 | 🔴 Critical | Privacy | Biometric face images + CSV PII committed |
| 6 | 🔴 Critical | Backend/CORS | `*` origins + credentials |
| 7 | 🟠 High | Architecture | 4.2k-line monolith; finish `src/` refactor |
| 8 | 🟠 High | Architecture | Imports inside handlers |
| 9 | 🟠 High | Testing | No test suite |
| 10 | 🟠 High | Repo | Build artifacts/scratch committed |
| 11 | UI | Frontend/build | Tailwind installed but unused |
| 12 | UI | Frontend/perf | 582 inline styles, no reuse |
| 13 | UI | Frontend/build | Logo path breaks in prod build |
| 14 | UI | Frontend/data | Over-fetching via effects |
| 15 | UI | Frontend/correctness | Client-side filtering of paginated data |
| 16 | UI | Frontend | Redundant cache-busting |
| 17 | UI | Frontend/security | API key in localStorage |
| 18 | UI | Frontend/types | 53 `any` usages |
| 19 | UI | Frontend/UX | No error boundary, inconsistent states |
| 20 | UI | Frontend | `console.error` in prod paths |
| 21 | 🟡 Opt | Backend | Per-route auth repetition |
| 22 | 🟡 Opt | Backend | 50k-row camera dropdown fetch |
| 23 | 🟡 Opt | Backend/db | Index events store columns |
| 24 | 🟡 Opt | Backend | Static files via uvicorn |
| 25 | 🟡 Opt | Backend | `/api` prefix rewrite in app |
| 26 | 🟡 Opt | Backend/deps | No hash-locked deps |
