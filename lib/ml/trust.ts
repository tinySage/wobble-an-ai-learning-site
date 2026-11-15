import { rng } from "./split.ts";

const gauss = (r: () => number) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
type F = (x: number[]) => number;

// ---------- Explaining a model ----------

/** Exact Shapley values for one row by trying every subset of features. A missing feature is filled in from each background row in turn (the interventional choice). */
export function shapleyExact(f: F, x: number[], bg: number[][]): { phi: number[]; base: number; value: number } {
  const d = x.length, v = new Array(1 << d).fill(0);
  for (let m = 0; m < 1 << d; m++) { let s = 0; for (const b of bg) s += f(x.map((xi, i) => ((m >> i) & 1 ? xi : b[i]))); v[m] = s / bg.length; }
  const fact = [1]; for (let i = 1; i <= d; i++) fact.push(fact[i - 1] * i);
  const phi = x.map((_, i) => { let tot = 0; for (let m = 0; m < 1 << d; m++) { if ((m >> i) & 1) continue; const k = m.toString(2).split("1").length - 1; tot += ((fact[k] * fact[d - k - 1]) / fact[d]) * (v[m | (1 << i)] - v[m]); } return tot; });
  return { phi, base: v[0], value: v[(1 << d) - 1] };
}

/** How much the error rises when one feature is shuffled. Large means the model leans on it. */
export function permutationImportance(f: F, X: number[][], y: number[], seed = 1, reps = 8): number[] {
  const r = rng(seed), mse = (rows: number[][]) => rows.reduce((s, q, i) => s + (f(q) - y[i]) ** 2, 0) / rows.length, base = mse(X), d = X[0].length;
  return Array.from({ length: d }, (_, j) => {
    let tot = 0;
    for (let k = 0; k < reps; k++) { const col = X.map((q) => q[j]); for (let i = col.length - 1; i > 0; i--) { const t = Math.floor(r() * (i + 1)); [col[i], col[t]] = [col[t], col[i]]; } tot += mse(X.map((q, i) => q.map((v, c) => (c === j ? col[i] : v)))) - base; }
    return tot / reps;
  });
}

// ---------- Fairness ----------
export interface FairData { s: number[]; y: number[]; g: number[] }
/** Two groups with different base rates. The score is equally informative in both: given the true outcome it is Normal(+1 or -1, 1) in each group. */
export function makeFairness(n = 4000, base: [number, number] = [0.5, 0.3], seed = 3): FairData {
  const r = rng(seed), s: number[] = [], y: number[] = [], g: number[] = [];
  for (let i = 0; i < n; i++) { const gi = i % 2, yi = r() < base[gi] ? 1 : 0; g.push(gi); y.push(yi); s.push((yi ? 1 : -1) + gauss(r)); }
  return { s, y, g };
}

export interface GroupStat { n: number; base: number; selection: number; tpr: number; fpr: number; ppv: number }
export function groupStats(d: FairData, thr: [number, number]): GroupStat[] {
  return [0, 1].map((gi) => {
    let n = 0, pos = 0, sel = 0, tp = 0, fp = 0, neg = 0;
    d.s.forEach((s, i) => { if (d.g[i] !== gi) return; n++; const hit = s >= thr[gi]; if (d.y[i]) { pos++; if (hit) tp++; } else { neg++; if (hit) fp++; } if (hit) sel++; });
    return { n, base: pos / n, selection: sel / n, tpr: pos ? tp / pos : 0, fpr: neg ? fp / neg : 0, ppv: sel ? tp / sel : 0 };
  });
}

/** Threshold that gives one group a target selection rate, true positive rate or false positive rate. */
export function thresholdFor(d: FairData, gi: number, kind: "selection" | "tpr" | "fpr", target: number): number {
  const pool = d.s.filter((_, i) => d.g[i] === gi && (kind === "selection" || (kind === "tpr" ? d.y[i] === 1 : d.y[i] === 0))).sort((a, b) => b - a);
  return pool[Math.min(pool.length - 1, Math.max(0, Math.round(target * pool.length) - 1))];
}

/** PPV from rates and base rate: TPR p / (TPR p + FPR (1 - p)). With equal error rates, different base rates force different PPV. */
export const ppvFromRates = (tpr: number, fpr: number, p: number) => (tpr * p) / (tpr * p + fpr * (1 - p));
