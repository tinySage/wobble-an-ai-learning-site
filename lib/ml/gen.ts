import { rng } from "./split.ts";

const gauss = (r: () => number) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());

// ---------- Autoencoders ----------
export type AEAct = "tanh" | "linear";
export interface AE { sizes: number[]; acts: AEAct[]; W: number[][][]; b: number[][] }

export function makeAE(sizes: number[], acts: AEAct[], seed = 1): AE {
  const r = rng(seed);
  return { sizes, acts, W: sizes.slice(1).map((n, l) => Array.from({ length: n }, () => Array.from({ length: sizes[l] }, () => gauss(r) * Math.sqrt(1 / sizes[l])))), b: sizes.slice(1).map((n) => new Array(n).fill(0)) };
}
const act = (a: AEAct, z: number) => (a === "tanh" ? Math.tanh(z) : z);

/** All layer outputs for one row, input first. */
export function aeForward(ae: AE, x: number[]): number[][] {
  const A = [x];
  ae.W.forEach((W, l) => A.push(W.map((row, j) => act(ae.acts[l], row.reduce((s, w, i) => s + w * A[l][i], ae.b[l][j])))));
  return A;
}
export const reconstruct = (ae: AE, x: number[]) => aeForward(ae, x)[ae.W.length];
/** The code: the output of the narrowest layer. */
export const encode = (ae: AE, x: number[]) => { const A = aeForward(ae, x), i = ae.sizes.indexOf(Math.min(...ae.sizes.slice(1, -1))); return A[i]; };
export const reconError = (ae: AE, X: number[][]) => X.reduce((s, x) => { const o = reconstruct(ae, x); return s + o.reduce((t, v, i) => t + (v - x[i]) ** 2, 0); }, 0) / X.length;

/** Full-batch Adam on squared reconstruction error. noise > 0 feeds corrupted rows but asks for the clean ones back (denoising). */
export function trainAE(X: number[][], sizes: number[], acts: AEAct[], o: { epochs?: number; lr?: number; seed?: number; noise?: number } = {}) {
  const { epochs = 600, lr = 0.02, seed = 1, noise = 0 } = o, ae = makeAE(sizes, acts, seed), r = rng(seed + 100), L = ae.W.length, n = X.length, loss: number[] = [];
  const mW = ae.W.map((W) => W.map((row) => row.map(() => 0))), vW = mW.map((W) => W.map((row) => row.map(() => 0))), mb = ae.b.map((b) => b.map(() => 0)), vb = mb.map((b) => b.map(() => 0));
  for (let e = 1; e <= epochs; e++) {
    const gW = ae.W.map((W) => W.map((row) => row.map(() => 0))), gb = ae.b.map((b) => b.map(() => 0)); let tot = 0;
    for (const x of X) {
      const inp = noise > 0 ? x.map((v) => v + noise * gauss(r)) : x, A = aeForward(ae, inp);
      let d = A[L].map((v, i) => (2 * (v - x[i])) / n); tot += A[L].reduce((s, v, i) => s + (v - x[i]) ** 2, 0) / n;
      for (let l = L - 1; l >= 0; l--) {
        const dz = d.map((v, j) => (ae.acts[l] === "tanh" ? v * (1 - A[l + 1][j] ** 2) : v));
        dz.forEach((dj, j) => { gb[l][j] += dj; A[l].forEach((a, i) => { gW[l][j][i] += dj * a; }); });
        if (l > 0) d = A[l].map((_, i) => dz.reduce((s, dj, j) => s + dj * ae.W[l][j][i], 0));
      }
    }
    loss.push(tot);
    const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** e, c2 = 1 - b2 ** e;
    for (let l = 0; l < L; l++) {
      ae.W[l].forEach((row, j) => row.forEach((_, i) => { const g = gW[l][j][i]; mW[l][j][i] = b1 * mW[l][j][i] + (1 - b1) * g; vW[l][j][i] = b2 * vW[l][j][i] + (1 - b2) * g * g; ae.W[l][j][i] -= (lr * (mW[l][j][i] / c1)) / (Math.sqrt(vW[l][j][i] / c2) + 1e-8); }));
      ae.b[l].forEach((_, j) => { const g = gb[l][j]; mb[l][j] = b1 * mb[l][j] + (1 - b1) * g; vb[l][j] = b2 * vb[l][j] + (1 - b2) * g * g; ae.b[l][j] -= (lr * (mb[l][j] / c1)) / (Math.sqrt(vb[l][j] / c2) + 1e-8); });
    }
  }
  return { ae, loss };
}

/** A noisy half circle: one number (the angle) describes each point, but not in a straight line. */
export function makeArc(n = 120, noise = 0.03, seed = 2): number[][] {
  const r = rng(seed); return Array.from({ length: n }, () => { const t = r() * Math.PI; return [Math.cos(t) + noise * gauss(r), Math.sin(t) + noise * gauss(r)]; });
}

// ---------- Diffusion ----------
export interface Schedule { betas: number[]; alphas: number[]; abar: number[] }
/** Noise schedule. strength scales how much noise is added in total (1 is the usual amount). */
export function schedule(T: number, kind: "linear" | "cosine" = "linear", strength = 1): Schedule {
  let betas: number[];
  if (kind === "linear") { const k = 1000 / T; betas = Array.from({ length: T }, (_, t) => Math.min(0.999, strength * k * (1e-4 + (0.02 - 1e-4) * (t / Math.max(1, T - 1))))); }
  else { const s = 0.008, f = (t: number) => Math.cos(((t / T + s) / (1 + s)) * Math.PI / 2) ** 2; betas = Array.from({ length: T }, (_, t) => Math.min(0.999, strength * (1 - f(t + 1) / f(t)))); }
  const alphas = betas.map((b) => 1 - b), abar: number[] = []; alphas.forEach((a, t) => abar.push((t ? abar[t - 1] : 1) * a));
  return { betas, alphas, abar };
}
/** x_t straight from x_0 in one step: sqrt(abar) x0 + sqrt(1 - abar) noise. t counts from 1. */
export const qSample = (x0: number, t: number, s: Schedule, eps: number) => Math.sqrt(s.abar[t - 1]) * x0 + Math.sqrt(1 - s.abar[t - 1]) * eps;

export interface Mix { w: number[]; mu: number[]; sd: number[] }
export function sampleMix(m: Mix, n: number, seed: number): number[] {
  const r = rng(seed); return Array.from({ length: n }, () => { let u = r(), k = 0; while (k < m.w.length - 1 && u > m.w[k]) { u -= m.w[k]; k++; } return m.mu[k] + m.sd[k] * gauss(r); });
}
/** Exact score (gradient of log density) of the noised mixture at noise level abar. */
export function mixScore(m: Mix, x: number, abar: number): number {
  const mu = m.mu.map((v) => Math.sqrt(abar) * v), vr = m.sd.map((s) => abar * s * s + 1 - abar);
  const lw = m.w.map((w, k) => Math.log(w) - 0.5 * Math.log(vr[k]) - ((x - mu[k]) ** 2) / (2 * vr[k])), mx = Math.max(...lw), e = lw.map((v) => Math.exp(v - mx)), z = e.reduce((a, b) => a + b, 0);
  return e.reduce((s, v, k) => s + (v / z) * (-(x - mu[k]) / vr[k]), 0);
}
/** Ancestral sampling: start from standard noise and walk back with the exact score standing in for a trained network. */
export function reverseSample(m: Mix, s: Schedule, n: number, seed: number, keepPaths = 0) {
  const r = rng(seed), T = s.betas.length, out: number[] = [], paths: number[][] = [];
  for (let i = 0; i < n; i++) {
    let x = gauss(r); const p = [x];
    for (let t = T; t >= 1; t--) {
      const eps = -Math.sqrt(1 - s.abar[t - 1]) * mixScore(m, x, s.abar[t - 1]);
      x = (x - (s.betas[t - 1] / Math.sqrt(1 - s.abar[t - 1])) * eps) / Math.sqrt(s.alphas[t - 1]) + (t > 1 ? Math.sqrt(s.betas[t - 1]) * gauss(r) : 0);
      if (i < keepPaths) p.push(x);
    }
    out.push(x); if (i < keepPaths) paths.push(p);
  }
  return { samples: out, paths };
}
