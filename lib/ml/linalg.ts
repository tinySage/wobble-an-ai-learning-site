export type Matrix = number[][];

/** Least squares by Householder QR. Stable for the ill-conditioned polynomial designs the lessons use. */
export function lstsq(A: Matrix, b: number[]): number[] {
  const m = A.length, n = A[0].length;
  const R = A.map((r) => r.slice()), y = b.slice();
  for (let k = 0; k < n; k++) {
    let norm = 0;
    for (let i = k; i < m; i++) norm += R[i][k] ** 2;
    norm = Math.sqrt(norm);
    if (norm === 0) continue;
    const alpha = R[k][k] > 0 ? -norm : norm;
    const v = new Array(m - k).fill(0).map((_, i) => R[k + i][k]);
    v[0] -= alpha;
    const vn = Math.sqrt(v.reduce((s, t) => s + t * t, 0));
    if (vn === 0) continue;
    for (let i = 0; i < v.length; i++) v[i] /= vn;
    for (let j = k; j < n; j++) {
      let d = 0; for (let i = 0; i < v.length; i++) d += v[i] * R[k + i][j];
      for (let i = 0; i < v.length; i++) R[k + i][j] -= 2 * v[i] * d;
    }
    let d = 0; for (let i = 0; i < v.length; i++) d += v[i] * y[k + i];
    for (let i = 0; i < v.length; i++) y[k + i] -= 2 * v[i] * d;
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let j = i + 1; j < n; j++) s -= R[i][j] * x[j];
    x[i] = Math.abs(R[i][i]) < 1e-12 ? 0 : s / R[i][i];
  }
  return x;
}

/** Solve a symmetric positive definite system by Cholesky. */
export function solveSPD(A: Matrix, b: number[]): number[] {
  const n = A.length, L: Matrix = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = A[i][j]; for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
    if (i === j) { if (s <= 0) throw new Error("matrix is not positive definite"); L[i][i] = Math.sqrt(s); } else L[i][j] = s / L[j][j];
  }
  const z = new Array(n).fill(0);
  for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * z[k]; z[i] = s / L[i][i]; }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) { let s = z[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; }
  return x;
}

export const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
export const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0);
