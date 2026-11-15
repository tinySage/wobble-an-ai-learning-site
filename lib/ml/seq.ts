import { rng, shuffled } from "./split.ts";
import { fitLinear, predict } from "./regression.ts";

const gauss = (r: () => number) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());

// ---------- Time series ----------

/** A series with a slow trend, a yearly-style cycle of period 12 and a little autocorrelated noise. */
export function makeSeries(n = 144, seed = 3, noise = 0.35): number[] {
  const r = rng(seed); let e = 0;
  return Array.from({ length: n }, (_, t) => { e = 0.5 * e + noise * gauss(r); return 0.04 * t + 1.4 * Math.sin((2 * Math.PI * t) / 12) + e; });
}

/** Same splits as scikit-learn's TimeSeriesSplit: equal test blocks at the end, each trained on everything before it (minus an optional gap). */
export function tsSplit(n: number, nSplits: number, gap = 0): { train: number[]; test: number[] }[] {
  const size = Math.floor(n / (nSplits + 1)), out: { train: number[]; test: number[] }[] = [];
  for (let s = n - nSplits * size; s < n; s += size) {
    const end = s - gap;
    out.push({ train: Array.from({ length: Math.max(0, end) }, (_, i) => i), test: Array.from({ length: Math.min(size, n - s) }, (_, i) => s + i) });
  }
  return out;
}

/** Row t uses y[t-1], y[t-2], y[t-12] and the time index. With peek it also gets the average of the neighbours y[t-1] and y[t+1], which needs the future. */
export function seriesRows(y: number[], peek: boolean) {
  const rows: number[][] = [], target: number[] = [], at: number[] = [];
  for (let t = 12; t < y.length - 1; t++) {
    const r = [y[t - 1], y[t - 2], y[t - 12], t / 100]; if (peek) r.push((y[t - 1] + y[t + 1]) / 2);
    rows.push(r); target.push(y[t]); at.push(t);
  }
  return { rows, target, at };
}

export type CvMode = "shuffled" | "walk";
/** Mean squared error of a ridge forecaster under shuffled K-fold or walk-forward testing. */
export function scoreSeries(y: number[], mode: CvMode, peek: boolean, folds = 5, seed = 1, lambda = 1e-3): number {
  const { rows, target } = seriesRows(y, peek), n = rows.length;
  let splits: { train: number[]; test: number[] }[];
  if (mode === "walk") splits = tsSplit(n, folds);
  else { const idx = shuffled(n, seed), sz = Math.floor(n / folds); splits = Array.from({ length: folds }, (_, f) => { const test = idx.slice(f * sz, f === folds - 1 ? n : (f + 1) * sz), set = new Set(test); return { test, train: idx.filter((i) => !set.has(i)) }; }); }
  let se = 0, cnt = 0;
  for (const s of splits) {
    const m = fitLinear(s.train.map((i) => rows[i]), s.train.map((i) => target[i]), lambda), p = predict(m, s.test.map((i) => rows[i]));
    p.forEach((v, k) => { se += (v - target[s.test[k]]) ** 2; cnt++; });
  }
  return se / cnt;
}

// ---------- Hidden Markov models (discrete observations) ----------

export interface HMM { pi: number[]; A: number[][]; B: number[][] }

/** Scaled forward pass. loglik is the log probability of the observations. */
export function forwardHMM(m: HMM, obs: number[]) {
  const N = m.pi.length, T = obs.length, alpha: number[][] = [], c: number[] = [];
  for (let t = 0; t < T; t++) {
    const a = Array.from({ length: N }, (_, j) => (t === 0 ? m.pi[j] : alpha[t - 1].reduce((s, v, i) => s + v * m.A[i][j], 0)) * m.B[j][obs[t]]);
    const z = a.reduce((s, v) => s + v, 0) || 1e-300; c.push(z); alpha.push(a.map((v) => v / z));
  }
  return { alpha, c, loglik: c.reduce((s, v) => s + Math.log(v), 0) };
}

export function backwardHMM(m: HMM, obs: number[], c: number[]) {
  const N = m.pi.length, T = obs.length, beta: number[][] = Array.from({ length: T }, () => new Array(N).fill(1));
  for (let t = T - 2; t >= 0; t--) for (let i = 0; i < N; i++) beta[t][i] = m.A[i].reduce((s, a, j) => s + a * m.B[j][obs[t + 1]] * beta[t + 1][j], 0) / c[t + 1];
  return beta;
}

/** P(state at t | all observations). */
export function posteriorHMM(m: HMM, obs: number[]): number[][] {
  const f = forwardHMM(m, obs), b = backwardHMM(m, obs, f.c);
  return f.alpha.map((a, t) => { const g = a.map((v, i) => v * b[t][i]), z = g.reduce((s, v) => s + v, 0); return g.map((v) => v / z); });
}

/** Most probable single state path, in log space. */
export function viterbi(m: HMM, obs: number[]) {
  const N = m.pi.length, T = obs.length, L = (x: number) => Math.log(Math.max(x, 1e-300));
  const d: number[][] = [], back: number[][] = [];
  for (let t = 0; t < T; t++) {
    const row: number[] = [], bp: number[] = [];
    for (let j = 0; j < N; j++) {
      if (t === 0) { row.push(L(m.pi[j]) + L(m.B[j][obs[0]])); bp.push(0); continue; }
      let best = -Infinity, arg = 0; for (let i = 0; i < N; i++) { const v = d[t - 1][i] + L(m.A[i][j]); if (v > best) { best = v; arg = i; } }
      row.push(best + L(m.B[j][obs[t]])); bp.push(arg);
    }
    d.push(row); back.push(bp);
  }
  let s = d[T - 1].indexOf(Math.max(...d[T - 1])); const logp = d[T - 1][s], path = new Array(T).fill(0);
  for (let t = T - 1; t >= 0; t--) { path[t] = s; s = back[t][s]; }
  return { path, logp };
}

/** Baum-Welch (EM). Returns the final model and the log-likelihood after every pass. */
export function baumWelch(m0: HMM, obs: number[], iters = 30) {
  let m: HMM = { pi: [...m0.pi], A: m0.A.map((r) => [...r]), B: m0.B.map((r) => [...r]) };
  const N = m.pi.length, M = m.B[0].length, T = obs.length, hist: number[] = [];
  for (let it = 0; it < iters; it++) {
    const f = forwardHMM(m, obs), b = backwardHMM(m, obs, f.c); hist.push(f.loglik);
    const gamma = f.alpha.map((a, t) => a.map((v, i) => v * b[t][i]));
    const xi = Array.from({ length: N }, () => new Array(N).fill(0));
    for (let t = 0; t < T - 1; t++) for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) xi[i][j] += (f.alpha[t][i] * m.A[i][j] * m.B[j][obs[t + 1]] * b[t + 1][j]) / f.c[t + 1];
    const A = xi.map((row) => { const z = row.reduce((s, v) => s + v, 0) || 1e-300; return row.map((v) => v / z); });
    const B = Array.from({ length: N }, (_, i) => { const row = new Array(M).fill(1e-12); for (let t = 0; t < T; t++) row[obs[t]] += gamma[t][i]; const z = row.reduce((s, v) => s + v, 0); return row.map((v) => v / z); });
    m = { pi: gamma[0].slice(), A, B };
  }
  hist.push(forwardHMM(m, obs).loglik);
  return { model: m, hist };
}

export function sampleHMM(m: HMM, n: number, seed = 1) {
  const r = rng(seed), pick = (p: number[]) => { let u = r(), k = 0; while (k < p.length - 1 && u > p[k]) { u -= p[k]; k++; } return k; };
  const states: number[] = [], obs: number[] = []; let s = pick(m.pi);
  for (let t = 0; t < n; t++) { states.push(s); obs.push(pick(m.B[s])); s = pick(m.A[s]); }
  return { states, obs };
}

/** Exact likelihood by listing every state path. Only for tiny inputs; used to check the forward pass. */
export function bruteLikelihood(m: HMM, obs: number[]): number {
  const N = m.pi.length, T = obs.length; let tot = 0;
  const walk = (t: number, prev: number, p: number) => { if (t === T) { tot += p; return; } for (let j = 0; j < N; j++) walk(t + 1, j, p * (t === 0 ? m.pi[j] : m.A[prev][j]) * m.B[j][obs[t]]); };
  walk(0, -1, 1); return tot;
}
