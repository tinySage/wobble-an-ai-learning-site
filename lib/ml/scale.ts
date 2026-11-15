import { mean, type Matrix } from "./linalg.ts";

/** Standardise columns to mean 0 and population standard deviation 1 (scikit-learn's StandardScaler). */
export function standardise(X: Matrix) {
  const p = X[0].length, n = X.length;
  const mu = Array.from({ length: p }, (_, j) => mean(X.map((r) => r[j])));
  const sd = Array.from({ length: p }, (_, j) => Math.sqrt(X.reduce((s, r) => s + (r[j] - mu[j]) ** 2, 0) / n) || 1);
  return { mean: mu, std: sd, transform: (M: Matrix) => M.map((r) => r.map((v, j) => (v - mu[j]) / sd[j])) };
}
