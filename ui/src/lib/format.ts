export function fmtPercent(x: number | null | undefined): string {
  if (x == null || isNaN(Number(x))) return '—';
  return `${(Number(x) * 100).toFixed(2)}%`;
}

export function shortId(id: string | null | undefined, len: number = 8): string {
  const s = (id || '').trim();
  if (!s) return '—';
  return s.length <= len ? s : `${s.slice(0, Math.max(4, Math.floor(len/2)))}…${s.slice(-Math.max(4, Math.ceil(len/2)))}`;
}

export function fmtDate(s?: string | null): string {
  if (!s) return '—';
  const d = new Date(s);
  if (isNaN(d.getTime())) return s;
  return d.toLocaleString();
}
