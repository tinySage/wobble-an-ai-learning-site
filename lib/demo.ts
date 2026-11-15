/** Seeded demo data for the home-page plot. The real engine arrives in stage 2. */
function mulberry32(a: number) { return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function normal(r: () => number) { return Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r()); }
function polyfit(xs: number[], ys: number[], d: number): number[] {
  const n = d + 1, A = Array.from({ length: n }, () => new Array(n + 1).fill(0));
  for (let i = 0; i < n; i++) { for (let j = 0; j < n; j++) A[i][j] = xs.reduce((s, x) => s + x ** (i + j), 0); A[i][n] = xs.reduce((s, x, k) => s + ys[k] * x ** i, 0); }
  for (let c = 0; c < n; c++) { let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r; [A[c], A[p]] = [A[p], A[c]];
    for (let r = 0; r < n; r++) if (r !== c) { const f = A[r][c] / A[c][c]; for (let k = c; k <= n; k++) A[r][k] -= f * A[c][k]; } }
  return A.map((row, i) => row[n] / row[i]);
}
export const evalPoly = (c: number[], x: number) => c.reduce((s, v, i) => s + v * x ** i, 0);
export function demo() {
  const r = mulberry32(7), f = (x: number) => Math.sin(2 * Math.PI * x) * 0.9;
  const xs = Array.from({ length: 14 }, () => 0.04 + r() * 0.92).sort((a, b) => a - b);
  const ys = xs.map((x) => f(x) + normal(r) * 0.22);
  const c = polyfit(xs, ys, 3);
  const curve = Array.from({ length: 100 }, (_, i) => { const x = i / 99; return { x, y: evalPoly(c, x) }; });
  return { xs, ys, curve, fit: (x: number) => evalPoly(c, x) };
}
