# UI Guide — Changing the Dashboard & Extending with Existing APIs

> How the `ui/` dashboard is wired, how to safely change it, the full API surface it
> can call, and concrete features we can add **using endpoints that already exist** on
> the backend. Pairs with `docs/REPO_REVIEW.md`.

---

## 1. How the UI is structured

```
ui/
├── index.html              # Vite entry
├── vite.config.ts          # dev server + allowedHosts (face.service.tools.thefusionapps.com)
├── package.json            # React 18, react-router-dom 6, Vite 6
└── src/
    ├── main.tsx            # mounts <BrowserRouter><App/></BrowserRouter>
    ├── App.tsx             # all routes (see table below)
    ├── layouts/AppShell.tsx# sidebar nav + <Outlet/>
    ├── lib/
    │   ├── api.ts          # ALL backend calls live here (typed wrappers)
    │   └── format.ts       # formatting helpers
    ├── components/StatCard.tsx
    ├── index.css           # CSS variables (theme) + .card/.primary/.grid classes
    └── pages/              # one file per route
        ├── Dashboard.tsx   Enroll.tsx     Search.tsx
        ├── Recognition.tsx Employees.tsx  Rejections.tsx
        ├── Subjects.tsx    SubjectDetail.tsx
        ├── SearchEvents.tsx Settings.tsx
```

### Routes (`App.tsx` → `AppShell` nav)

| Path | Page | Purpose |
|---|---|---|
| `/` | Dashboard | Top-level stats |
| `/enroll` | Enroll | Add a subject's faces |
| `/search` | Search | Upload an image, find matches |
| `/recognition` | Recognition | Recognition event history + feedback |
| `/events` | SearchEvents | Search history |
| `/employees` | Employees | Employee subjects |
| `/rejections` | Rejections | Quality-rejected events |
| `/subjects` | Subjects | Subject list |
| `/subjects/:id` | SubjectDetail | Single subject |
| `/settings` | Settings | API base URL + API key |

---

## 2. Conventions you must follow when changing the UI

These match the *current* code so new work stays consistent. (See `REPO_REVIEW.md` #11–#13
for the tech-debt caveats — Tailwind is installed but unused, styling is inline.)

- **All HTTP goes through `src/lib/api.ts`.** Never `fetch()` directly in a page. Add a typed
  wrapper function + its response `type` there, then import it into the page.
- **Auth + base URL are automatic.** `apiGet/apiPostJson/apiPostForm/apiDelete` already inject
  `x-api-key` (from `localStorage` or `VITE_API_KEY`) and prefix `getApiBase()`. Just pass the path.
- **Images** are served by the API: build URLs as `` `${getApiBase()}${thumb_path}` `` (paths come
  back relative, e.g. `/thumbs/...`, `/images/...`).
- **Styling** today is inline `style={{…}}` using CSS variables from `index.css`
  (`var(--primary)`, `var(--bg-secondary)`, `var(--radius-md)`, etc.) plus the `.card`,
  `.primary`, `.grid` classes. Reuse those tokens so the theme stays coherent.
- **Dates**: the UI displays in IST (`Asia/Kolkata`) via `Intl.DateTimeFormat`; date filters use
  `en-CA` (`YYYY-MM-DD`) strings. Copy the helpers in `Recognition.tsx` rather than re-inventing.
- **Pagination** uses opaque `cursor` values returned by the API + a "Load More" button. Don't
  assume offset paging.

### Add a new API call
In `src/lib/api.ts`:
```ts
export type FooResponse = { items: string[] };
export async function getFoo(id: string): Promise<FooResponse> {
  return apiGet(`/v1/foo/${encodeURIComponent(id)}`);
}
```

### Add a new page
1. Create `src/pages/Foo.tsx` (copy a small page like `Settings.tsx` as a skeleton).
2. Register the route in `App.tsx`: `<Route path="/foo" element={<Foo />} />`.
3. Add a nav entry in `layouts/AppShell.tsx`'s `nav` array.

### Run / build
```bash
cd ui
npm install
npm run dev      # http://localhost:5173, talks to VITE_API_BASE
npm run build    # tsc -b && vite build → dist/
```

---

## 3. Full API surface

### Already wrapped in `api.ts` (usable now)

| Function | Endpoint |
|---|---|
| `health` | `GET /health` |
| `stats` | `GET /v1/stats` |
| `facesSubjects` | `GET /v1/faces/subjects` |
| `subjects` / `getSubject` / `subjectImages` | `GET /v1/subjects…` |
| `deleteSubject` | `DELETE /v1/faces/subjects/{id}` |
| `getBranches` | `GET /v1/branches` |
| `facesAddUpload` | `POST /v1/faces/add_upload` |
| `compareFacesUpload` | `POST /v1/face/compare_upload` |
| `facesSearchUpload` | `POST /v1/faces/search_upload` |
| `facesRecognizeUpload` | `POST /v1/faces/recognize_upload` |
| `qualityCheckUpload` | `POST /v1/quality/check_upload` |
| `crossMatch` | `GET /v1/faces/cross_match/{id}` |
| `crossCheckVisitorsVsEmployees` | `GET /v1/cross_check/visitors_vs_employees` |
| `recognitionEvents` / `recognitionStats` / `recognitionCameras` | `GET /v1/events/recognition…` |
| `setRecognitionEventFeedback` / `recognitionFeedbackStats` | recognition feedback |
| `searchEvents` / `searchEventsStats` | `GET /v1/search_history…` |

### Backend endpoints that exist but are **NOT yet surfaced in the UI**

These are ready to use — they just need an `api.ts` wrapper + UI.

| Endpoint | What it enables in the UI |
|---|---|
| `GET /v1/groups`, `POST /v1/groups`, `DELETE /v1/groups/{id}` | **Groups management page** — none exists today |
| `POST /v1/branches`, `DELETE /v1/branches/{id}` | **Create/delete branches** (UI only reads them) |
| `GET /v1/events/recognition/{event_id}` | **Event detail view** (deep-link a single event) |
| `POST /v1/faces/privacy_extract` | **Privacy/redaction tool** (extract a face crop) |
| `POST /v1/events/recognition/forward` | **Forward/replay an event** to downstream consumers |
| `GET /v1/search_history/asset/image/{id}`, `/thumb/{id}` | Stable per-event image URLs for history |
| `GET /debug/providers` | **System/health panel** — show GPU/ONNX execution providers |
| `GET /metrics` | Prometheus metrics → a small **ops widget** or Grafana link |
| `POST /v1/face/compare`, `/v1/face/search`, `/v1/faces/search`, `/v1/faces/recognize` | JSON (non-upload) variants — useful for re-querying by embedding/URL without re-uploading |

---

## 4. Features we can add (using existing APIs)

Grouped by effort. All of these need **no new backend work**.

### Quick wins
- **Groups page** (`/groups`): list/create/delete via `/v1/groups`. Today there's zero UI for groups.
- **Branch admin**: add create + delete buttons on a branches view (`POST`/`DELETE /v1/branches`).
- **System/Health panel** on Dashboard: surface `GET /health` + `GET /debug/providers`
  (GPU provider, model load status) so operators can see the backend is healthy.
- **Event permalink / detail modal** using `GET /v1/events/recognition/{event_id}` — currently the
  Recognition modal only shows already-loaded rows; a deep-linkable `/events/:id` would help support.
- **Feedback dashboard**: `recognitionFeedbackStats` is wrapped but under-used — a chart of
  TP/FP/FN and `fp_rate_match` over time would make the labeling loop actionable.

### Medium
- **Live recognition feed**: poll `recognitionEvents` (it already has a `_cb` cache-buster) on an
  interval, or add auto-refresh, to show an operations "now" view.
- **CSV export** for Recognition / Search history (client-side from the already-fetched rows).
- **Privacy-extract tool page**: upload an image → `POST /v1/faces/privacy_extract` → show the
  cropped/redacted face. Useful for data-handling workflows.
- **Cross-check (visitors vs employees) page**: `crossCheckVisitorsVsEmployees` is wrapped but has
  no dedicated page — a screen with camera/date filters would expose this security feature.
- **Subject merge/relabel UX**: combine `subjects` + `deleteSubject` + `facesAddUpload` to let an
  operator move images between subject IDs.

### Larger / UX overhauls
- **Server-side correctness for grouping & counts** (see `REPO_REVIEW.md` #15): move the duration /
  unique-visit grouping out of the client so numbers reflect the full dataset, not just the page.
- **Adopt a data layer** (TanStack Query) to dedupe the multiple effects firing per filter change
  (`REPO_REVIEW.md` #14) and get caching/retries for free.
- **Design-system pass**: extract shared `<Card> <Select> <TextInput> <StatusPill>` and either
  commit to Tailwind (currently installed-but-unused) or move inline styles to classes
  (`REPO_REVIEW.md` #11–#12).
- **Global filter bar / saved views**: camera + date range are re-implemented per page; a shared
  filter context would unify Recognition, SearchEvents, Rejections, Cross-check.
- **Dark mode**: the theme is already CSS-variable based in `index.css` — add a second variable set
  and a toggle.

---

## 5. Gotchas to watch

- **Logo path breaks in production** — `AppShell.tsx` uses `/src/assets/logo.png`; import the asset
  instead before shipping (`REPO_REVIEW.md` #13).
- **API key lives in `localStorage`** (Settings page) — readable by XSS; fine for an internal tool,
  but don't expand its blast radius.
- **`allowedHosts` / HMR host** in `vite.config.ts` are hardcoded to
  `face.service.tools.thefusionapps.com`; update for other environments.
- **Client-side filtering over paginated data** can show misleading totals (see #15) — prefer the
  server `*/stats` endpoints for headline numbers.
