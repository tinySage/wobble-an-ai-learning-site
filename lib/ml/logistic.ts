import { solveSPD, type Matrix } from "./linalg.ts";

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

/** L2 logistic regression by Newton steps. Objective: sum of log loss + (1 / 2C) ||w||^2, intercept not penalised (scikit-learn's convention). */
export function fitLogistic(X: Matrix, y: number[], C = 1, iters = 100, tol = 1e-12) {
  const n = X.length, p = X[0].length, lam = 1 / C, w = new Array(p + 1).fill(0); // w[p] is the intercept
  const z = (r: number[]) => r.reduce((s, v, j) => s + v * w[j], 0) + w[p];
  for (let it = 0; it < iters; it++) {
    const g = new Array(p + 1).fill(0), H: Matrix = Array.from({ length: p + 1 }, () => new Array(p + 1).fill(0));
    for (let i = 0; i < n; i++) {
      const pr = sigmoid(z(X[i])), e = pr - y[i], s = pr * (1 - pr), row = [...X[i], 1];
      for (let a = 0; a <= p; a++) { g[a] += e * row[a]; for (let b = 0; b <= p; b++) H[a][b] += s * row[a] * row[b]; }
    }
    for (let j = 0; j < p; j++) { g[j] += lam * w[j]; H[j][j] += lam; }
    H[p][p] += 1e-12;
    const step = solveSPD(H, g);
    let delta = 0; for (let a = 0; a <= p; a++) { w[a] -= step[a]; delta = Math.max(delta, Math.abs(step[a])); }
    if (delta < tol) break;
  }
  return { coef: w.slice(0, p), intercept: w[p], proba: (M: Matrix) => M.map((r) => sigmoid(z(r))) };
}
