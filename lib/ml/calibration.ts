/** Mean squared gap between predicted probability and the 0/1 outcome. Lower is better. */
export const brier = (y: number[], p: number[]) => y.reduce((s, t, i) => s + (p[i] - t) ** 2, 0) / y.length;

/** Reliability diagram points: equal-width bins on [0, 1], empty bins dropped (scikit-learn's calibration_curve, strategy="uniform"). */
export function reliability(y: number[], p: number[], bins = 10) {
  const out: { mean: number; frac: number; n: number }[] = [];
  for (let b = 0; b < bins; b++) {
    const idx = p.map((v, i) => i).filter((i) => (b === bins - 1 ? p[i] >= b / bins && p[i] <= 1 : p[i] >= b / bins && p[i] < (b + 1) / bins));
    if (idx.length) out.push({ mean: idx.reduce((s, i) => s + p[i], 0) / idx.length, frac: idx.reduce((s, i) => s + y[i], 0) / idx.length, n: idx.length });
  }
  return out;
}

/** Expected calibration error: bin gaps weighted by how many predictions fall in each bin. */
export function ece(y: number[], p: number[], bins = 10) {
  return reliability(y, p, bins).reduce((s, b) => s + (b.n / y.length) * Math.abs(b.mean - b.frac), 0);
}

/** Average precision, as scikit-learn defines it: sum over thresholds of (recall step) x precision. */
export function averagePrecision(y: number[], score: number[]) {
  const th = [...new Set(score)].sort((a, b) => b - a), np = y.filter((t) => t === 1).length;
  let prevR = 0, ap = 0;
  for (const t of th) {
    let tp = 0, fp = 0; y.forEach((v, i) => { if (score[i] >= t) v === 1 ? tp++ : fp++; });
    const r = tp / np; ap += (r - prevR) * (tp / (tp + fp)); prevR = r;
  }
  return ap;
}
