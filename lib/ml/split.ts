/** Small seeded generator so every lesson shows the same data for every visitor. */
export function rng(seed: number) {
  let a = seed | 0;
  return () => { a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export function shuffled(n: number, seed: number): number[] {
  const r = rng(seed), a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
export function trainTestSplit(n: number, testFraction: number, seed: number) {
  const idx = shuffled(n, seed), nt = Math.max(1, Math.round(n * testFraction));
  return { test: idx.slice(0, nt).sort((a, b) => a - b), train: idx.slice(nt).sort((a, b) => a - b) };
}
/** k folds as in scikit-learn's KFold: the first n % k folds get one extra row. Shuffle only when a seed is given. */
export function kFold(n: number, k: number, seed?: number): { train: number[]; test: number[] }[] {
  const idx = seed === undefined ? Array.from({ length: n }, (_, i) => i) : shuffled(n, seed);
  const out = []; let start = 0;
  for (let f = 0; f < k; f++) {
    const size = Math.floor(n / k) + (f < n % k ? 1 : 0);
    const test = idx.slice(start, start + size);
    out.push({ test: test.slice().sort((a, b) => a - b), train: [...idx.slice(0, start), ...idx.slice(start + size)].sort((a, b) => a - b) });
    start += size;
  }
  return out;
}
/** Time-ordered split. Training always ends before testing begins. */
export const timeSplit = (n: number, trainFraction: number) => {
  const c = Math.floor(n * trainFraction);
  return { train: Array.from({ length: c }, (_, i) => i), test: Array.from({ length: n - c }, (_, i) => c + i) };
};
/** Walk-forward windows: fit on [0, t), test on [t, t + step). */
export function walkForward(n: number, minTrain: number, step: number) {
  const out = [];
  for (let t = minTrain; t < n; t += step) out.push({ train: Array.from({ length: t }, (_, i) => i), test: Array.from({ length: Math.min(step, n - t) }, (_, i) => t + i) });
  return out;
}
