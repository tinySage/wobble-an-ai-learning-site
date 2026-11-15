export interface Step { w: number; b: number; loss: number }

/** Plain gradient descent for y = w x + b with mean squared error. Returns every step so a lesson can animate it. */
export function gradientDescent(x: number[], y: number[], lr: number, steps: number): Step[] {
  const n = x.length; let w = 0, b = 0; const out: Step[] = [];
  for (let s = 0; s < steps; s++) {
    let loss = 0, gw = 0, gb = 0;
    for (let i = 0; i < n; i++) { const e = w * x[i] + b - y[i]; loss += e * e; gw += e * x[i]; gb += e; }
    out.push({ w, b, loss: loss / n });
    w -= lr * 2 * gw / n; b -= lr * 2 * gb / n;
  }
  return out;
}

/** A run has diverged when the loss grows over the last stretch or stops being finite. */
export function diverged(traj: Step[]): boolean {
  const last = traj[traj.length - 1];
  if (!Number.isFinite(last.loss)) return true;
  const k = Math.min(5, traj.length - 1);
  return k > 0 && last.loss > traj[traj.length - 1 - k].loss * 1.5;
}

/** Gradient descent for many features with an intercept. Losses only, so long runs stay cheap. */
export function gradientDescentMulti(X: number[][], y: number[], lr: number, steps: number): { losses: number[]; w: number[] } {
  const n = X.length, p = X[0].length, w = new Array(p + 1).fill(0), losses: number[] = [];
  for (let s = 0; s < steps; s++) {
    const g = new Array(p + 1).fill(0); let loss = 0;
    for (let i = 0; i < n; i++) {
      let e = w[p]; for (let j = 0; j < p; j++) e += w[j] * X[i][j]; e -= y[i];
      loss += e * e; for (let j = 0; j < p; j++) g[j] += e * X[i][j]; g[p] += e;
    }
    losses.push(loss / n);
    for (let j = 0; j <= p; j++) w[j] -= lr * 2 * g[j] / n;
    if (!Number.isFinite(loss)) break;
  }
  return { losses, w };
}

/** Largest curvature of the mean-squared-error surface. Gradient descent is stable only below lr = 2 / lambdaMax. */
export function lambdaMax(X: number[][]): number {
  const n = X.length, p = X[0].length + 1, Z = X.map((r) => [...r, 1]);
  const A = Array.from({ length: p }, (_, a) => Array.from({ length: p }, (_, b) => (2 / n) * Z.reduce((s, r) => s + r[a] * r[b], 0)));
  // Jacobi rotations: exact for the small symmetric matrices used here, unlike power iteration when eigenvalues are close.
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0; for (let i = 0; i < p; i++) for (let j = i + 1; j < p; j++) off += A[i][j] ** 2;
    if (off < 1e-24) break;
    for (let i = 0; i < p; i++) for (let j = i + 1; j < p; j++) {
      if (Math.abs(A[i][j]) < 1e-300) continue;
      const th = (A[j][j] - A[i][i]) / (2 * A[i][j]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), sn = t * c;
      for (let k = 0; k < p; k++) { const x = A[k][i], y = A[k][j]; A[k][i] = c * x - sn * y; A[k][j] = sn * x + c * y; }
      for (let k = 0; k < p; k++) { const x = A[i][k], y = A[j][k]; A[i][k] = c * x - sn * y; A[j][k] = sn * x + c * y; }
    }
  }
  return Math.max(...A.map((r, i) => r[i]));
}

/** First step at which the loss is within `tol` (relative) of the final optimum, or null if it never gets there. */
export function stepsToConverge(losses: number[], optimum: number, tol = 0.01): number | null {
  const i = losses.findIndex((l) => l <= optimum * (1 + tol) + 1e-12);
  return i < 0 ? null : i;
}
