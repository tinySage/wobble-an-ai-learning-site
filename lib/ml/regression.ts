import { dot, lstsq, mean, type Matrix } from "./linalg.ts";

export interface LinearModel { coef: number[]; intercept: number }

/** Columns x^1 .. x^degree (no bias column), the same layout as scikit-learn's PolynomialFeatures. */
export const polyFeatures = (x: number[], degree: number): Matrix => x.map((v) => Array.from({ length: degree }, (_, k) => v ** (k + 1)));

export const predict = (m: LinearModel, X: Matrix): number[] => X.map((r) => dot(m.coef, r) + m.intercept);

/** Ordinary or ridge least squares. Objective: ||y - Xw - b||^2 + lambda ||w||^2, intercept not penalised. */
export function fitLinear(X: Matrix, y: number[], lambda = 0): LinearModel {
  const n = X.length, p = X[0].length;
  const xm = Array.from({ length: p }, (_, j) => mean(X.map((r) => r[j]))), ym = mean(y);
  const Xc = X.map((r) => r.map((v, j) => v - xm[j])), yc = y.map((v) => v - ym);
  let A = Xc, b = yc;
  if (lambda > 0) {
    const s = Math.sqrt(lambda);
    A = [...Xc, ...Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => (i === j ? s : 0)))];
    b = [...yc, ...new Array(p).fill(0)];
  }
  const coef = lstsq(A, b);
  return { coef, intercept: ym - dot(coef, xm) };
}

/** Polynomial fit. x is mapped to [-1, 1] inside so high degrees stay numerically usable. */
export function fitPolynomial(x: number[], y: number[], degree: number, lambda = 0) {
  const lo = Math.min(...x), hi = Math.max(...x), mid = (lo + hi) / 2, half = (hi - lo) / 2 || 1;
  const feat = (v: number[]) => polyFeatures(v.map((t) => (t - mid) / half), degree);
  const model = fitLinear(feat(x), y, lambda);
  return { model, predict: (v: number[]) => predict(model, feat(v)) };
}

/** Lasso by coordinate descent. Objective: (1/2n)||y - Xw - b||^2 + alpha ||w||_1, as in scikit-learn. */
export function fitLasso(X: Matrix, y: number[], alpha: number, maxIter = 100000, tol = 1e-13): LinearModel {
  const n = X.length, p = X[0].length;
  const xm = Array.from({ length: p }, (_, j) => mean(X.map((r) => r[j]))), ym = mean(y);
  const Xc = X.map((r) => r.map((v, j) => v - xm[j])), r = y.map((v) => v - ym);
  const z = Array.from({ length: p }, (_, j) => Xc.reduce((s, row) => s + row[j] ** 2, 0) / n);
  const w = new Array(p).fill(0);
  for (let it = 0; it < maxIter; it++) {
    let delta = 0;
    for (let j = 0; j < p; j++) {
      if (z[j] === 0) continue;
      let rho = 0; for (let i = 0; i < n; i++) rho += Xc[i][j] * (r[i] + Xc[i][j] * w[j]);
      rho /= n;
      const nw = Math.sign(rho) * Math.max(Math.abs(rho) - alpha, 0) / z[j];
      const d = nw - w[j];
      if (d !== 0) { for (let i = 0; i < n; i++) r[i] -= Xc[i][j] * d; w[j] = nw; delta = Math.max(delta, Math.abs(d)); }
    }
    if (delta < tol) break;
  }
  return { coef: w, intercept: ym - dot(w, xm) };
}

export const mse = (a: number[], b: number[]) => a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length;
