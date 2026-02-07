from __future__ import annotations


def ui_html() -> str:
    return """<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Face Service</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }
      .row { display: flex; gap: 24px; flex-wrap: wrap; }
      .card { border: 1px solid #ddd; border-radius: 10px; padding: 16px; width: 420px; }
      input, button { font-size: 14px; }
      input[type=text], input[type=number] { width: 100%; padding: 8px; margin: 6px 0 10px 0; }
      input[type=file] { width: 100%; margin: 6px 0 10px 0; }
      button { padding: 8px 12px; cursor: pointer; }
      pre { background: #0b1020; color: #e6edf3; padding: 12px; border-radius: 10px; overflow: auto; }
      h2 { margin: 0 0 12px 0; }
    </style>
  </head>
  <body>
    <h1>Face Service</h1>
    <div class=\"row\">
      <div class=\"card\">
        <h2>Add / Enroll (multi-image)</h2>
        <label>Subject ID</label>
        <input id=\"add_subject\" type=\"text\" placeholder=\"e.g. alice\" />
        <label>Images (select multiple)</label>
        <input id=\"add_files\" type=\"file\" accept=\"image/*\" multiple />
        <button onclick=\"addFaces()\">Add</button>
        <pre id=\"add_out\"></pre>
      </div>

      <div class=\"card\">
        <h2>Search (top-K)</h2>
        <label>Top K</label>
        <input id=\"search_k\" type=\"number\" value=\"5\" min=\"1\" max=\"50\" />
        <label>Query Image</label>
        <input id=\"search_file\" type=\"file\" accept=\"image/*\" />
        <button onclick=\"searchFaces()\">Search</button>
        <pre id=\"search_out\"></pre>
      </div>

      <div class=\"card\">
        <h2>Recognize (best match)</h2>
        <label>Top K</label>
        <input id=\"rec_k\" type=\"number\" value=\"5\" min=\"1\" max=\"50\" />
        <label>Min similarity (optional)</label>
        <input id=\"rec_min\" type=\"number\" placeholder=\"leave blank to use server default\" step=\"0.01\" />
        <label>Query Image</label>
        <input id=\"rec_file\" type=\"file\" accept=\"image/*\" />
        <button onclick=\"recognizeFace()\">Recognize</button>
        <pre id=\"rec_out\"></pre>
      </div>

      <div class=\"card\">
        <h2>Frigate Search (/v1/face/search)</h2>
        <label>Query Image</label>
        <input id=\"frigate_file\" type=\"file\" accept=\"image/*\" />
        <button onclick=\"frigateSearch()\">Search</button>
        <pre id=\"frigate_out\"></pre>
      </div>
    </div>

    <script>
      async function addFaces() {
        const subject = document.getElementById('add_subject').value.trim();
        const files = document.getElementById('add_files').files;
        const out = document.getElementById('add_out');
        out.textContent = 'Working...';
        if (!subject) { out.textContent = 'subject_id required'; return; }
        if (!files || files.length === 0) { out.textContent = 'select at least one image'; return; }
        const fd = new FormData();
        fd.append('subject_id', subject);
        for (const f of files) { fd.append('files', f, f.name); }
        const resp = await fetch('/v1/faces/add_upload', { method: 'POST', body: fd });
        try {
          const data = await resp.json();
          out.textContent = JSON.stringify(data, null, 2);
        } catch {
          out.textContent = await resp.text();
        }
      }

      async function searchFaces() {
        const k = parseInt(document.getElementById('search_k').value || '5', 10);
        const file = document.getElementById('search_file').files[0];
        const out = document.getElementById('search_out');
        out.textContent = 'Working...';
        if (!file) { out.textContent = 'select an image'; return; }
        const fd = new FormData();
        fd.append('top_k', String(k));
        fd.append('file', file, file.name);
        const resp = await fetch('/v1/faces/search_upload', { method: 'POST', body: fd });
        try {
          const data = await resp.json();
          out.textContent = JSON.stringify(data, null, 2);
        } catch {
          out.textContent = await resp.text();
        }
      }

      async function recognizeFace() {
        const k = parseInt(document.getElementById('rec_k').value || '5', 10);
        const minStr = document.getElementById('rec_min').value;
        const file = document.getElementById('rec_file').files[0];
        const out = document.getElementById('rec_out');
        out.textContent = 'Working...';
        if (!file) { out.textContent = 'select an image'; return; }
        const fd = new FormData();
        fd.append('top_k', String(k));
        if (minStr !== null && String(minStr).trim() !== '') fd.append('min_similarity', String(parseFloat(minStr)));
        fd.append('file', file, file.name);
        const resp = await fetch('/v1/faces/recognize_upload', { method: 'POST', body: fd });
        try {
          const data = await resp.json();
          out.textContent = JSON.stringify(data, null, 2);
        } catch {
          out.textContent = await resp.text();
        }
      }

      async function frigateSearch() {
        const file = document.getElementById('frigate_file').files[0];
        const out = document.getElementById('frigate_out');
        out.textContent = 'Working...';
        if (!file) { out.textContent = 'select an image'; return; }
        const fd = new FormData();
        fd.append('file', file, file.name);
        const resp = await fetch('/v1/face/search_upload', { method: 'POST', body: fd });
        try {
          const data = await resp.json();
          out.textContent = JSON.stringify(data, null, 2);
        } catch {
          out.textContent = await resp.text();
        }
      }
    </script>
  </body>
</html>"""
