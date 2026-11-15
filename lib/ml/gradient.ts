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
  const n = X.length, p = X[0].length + 1;
  const Z = X.map((r) => [...r, 1]), H = Array.from({ length: p }, (_, a) => Array.from({ length: p }, (_, b) => (2 / n) * Z.reduce((s, r) => s + r[a] * r[b], 0)));
  let v = new Array(p).fill(1 / Math.sqrt(p)), lam = 0;
  for (let it = 0; it < 2000; it++) {
    const u = H.map((row) => row.reduce((s, h, j) => s + h * v[j], 0)), nu = Math.sqrt(u.reduce((s, t) => s + t * t, 0));
    if (nu === 0) return 0;
    v = u.map((t) => t / nu); if (Math.abs(nu - lam) < 1e-13 * nu) { lam = nu; break; } lam = nu;
  }
  return lam;
}

/** First step at which the loss is within `tol` (relative) of the final optimum, or null if it never gets there. */
export function stepsToConverge(losses: number[], optimum: number, tol = 0.01): number | null {
  const i = losses.findIndex((l) => l <= optimum * (1 + tol) + 1e-12);
  return i < 0 ? null : i;
}
