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
