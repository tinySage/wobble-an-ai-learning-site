import { rng } from "./split.ts";
export type Act = "relu" | "tanh" | "sigmoid" | "leaky" | "linear";
export const ACT: Record<Act, { f: (z: number) => number; d: (z: number) => number }> = {
  relu: { f: (z) => (z > 0 ? z : 0), d: (z) => (z > 0 ? 1 : 0) },
  leaky: { f: (z) => (z > 0 ? z : 0.01 * z), d: (z) => (z > 0 ? 1 : 0.01) },
  tanh: { f: Math.tanh, d: (z) => 1 - Math.tanh(z) ** 2 },
  sigmoid: { f: (z) => 1 / (1 + Math.exp(-z)), d: (z) => { const s = 1 / (1 + Math.exp(-z)); return s * (1 - s); } },
  linear: { f: (z) => z, d: () => 1 },
};
const sigmoid = ACT.sigmoid.f;
const gauss = (r: () => number) => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());

export interface Net { sizes: number[]; act: Act; residual: boolean; W: Float64Array[]; b: Float64Array[] }
export type Init = "auto" | "he" | "xavier" | number;
/** Dense network with a sigmoid output. W[l] is row-major (out x in). */
export function makeNet(sizes: number[], o: { act?: Act; init?: Init; seed?: number; residual?: boolean } = {}): Net {
  const r = rng(o.seed ?? 1), act = o.act ?? "tanh", init = o.init ?? "auto", W: Float64Array[] = [], b: Float64Array[] = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    const fin = sizes[l], fout = sizes[l + 1], heLike = act === "relu" || act === "leaky";
    const sd = typeof init === "number" ? init : init === "he" || (init === "auto" && heLike) ? Math.sqrt(2 / fin) : Math.sqrt(2 / (fin + fout));
    W.push(Float64Array.from({ length: fin * fout }, () => gauss(r) * sd)); b.push(new Float64Array(fout));
  }
  return { sizes, act, residual: !!o.residual, W, b };
}
export const cloneNet = (n: Net): Net => ({ ...n, W: n.W.map((w) => Float64Array.from(w)), b: n.b.map((v) => Float64Array.from(v)) });
const skip = (n: Net, l: number) => n.residual && l > 0 && n.sizes[l] === n.sizes[l + 1];

interface Cache { A: Float64Array[]; Z: Float64Array[]; M: (Float64Array | null)[]; p: number }
export function forward(n: Net, x: number[], drop = 0, r?: () => number): Cache {
  const L = n.sizes.length - 1, A: Float64Array[] = [Float64Array.from(x)], Z: Float64Array[] = [], M: (Float64Array | null)[] = [], f = ACT[n.act].f;
  for (let l = 0; l < L; l++) {
    const fin = n.sizes[l], fout = n.sizes[l + 1], z = new Float64Array(fout), a = A[l];
    for (let j = 0; j < fout; j++) { let s = n.b[l][j]; for (let i = 0; i < fin; i++) s += n.W[l][j * fin + i] * a[i]; z[j] = s; }
    Z.push(z);
    if (l === L - 1) break;
    const h = new Float64Array(fout); for (let j = 0; j < fout; j++) h[j] = f(z[j]) + (skip(n, l) ? a[j] : 0);
    if (drop > 0 && r) { const m = new Float64Array(fout); for (let j = 0; j < fout; j++) { m[j] = r() < drop ? 0 : 1 / (1 - drop); h[j] *= m[j]; } M.push(m); } else M.push(null);
    A.push(h);
  }
  return { A, Z, M, p: sigmoid(Z[L - 1][0]) };
}
export const predictProba = (n: Net, x: number[]) => forward(n, x).p;
export const netAccuracy = (n: Net, X: number[][], y: number[]) => X.filter((x, i) => (predictProba(n, x) >= 0.5 ? 1 : 0) === y[i]).length / X.length;
export type Bug = "none" | "no-act-deriv" | "sign";
export interface Grads { W: Float64Array[]; b: Float64Array[]; loss: number }
/** Mean binary cross-entropy (+ L2 on weights) and its gradient by backpropagation. */
export function lossGrad(n: Net, X: number[][], y: number[], o: { l2?: number; drop?: number; rand?: () => number; bug?: Bug } = {}): Grads {
  const L = n.sizes.length - 1, gW = n.W.map((w) => new Float64Array(w.length)), gb = n.b.map((v) => new Float64Array(v.length)), d = ACT[n.act].d, l2 = o.l2 ?? 0; let loss = 0;
  for (let s = 0; s < X.length; s++) {
    const c = forward(n, X[s], o.drop ?? 0, o.rand), p = Math.min(Math.max(c.p, 1e-12), 1 - 1e-12);
    loss += -(y[s] * Math.log(p) + (1 - y[s]) * Math.log(1 - p));
    let dz: Float64Array = Float64Array.of(c.p - y[s]), extra: Float64Array | null = null;
    for (let l = L - 1; l >= 0; l--) {
      const fin = n.sizes[l], fout = n.sizes[l + 1], a = c.A[l], da = new Float64Array(fin);
      for (let j = 0; j < fout; j++) { gb[l][j] += dz[j]; for (let i = 0; i < fin; i++) { gW[l][j * fin + i] += dz[j] * a[i]; da[i] += n.W[l][j * fin + i] * dz[j]; } }
      if (extra) for (let i = 0; i < fin; i++) da[i] += extra[i];
      if (l === 0) break;
      const dh = Float64Array.from(da), m = c.M[l - 1]; if (m) for (let j = 0; j < fin; j++) dh[j] *= m[j];
      extra = skip(n, l - 1) ? Float64Array.from(dh) : null;
      dz = new Float64Array(fin); for (let j = 0; j < fin; j++) dz[j] = o.bug === "no-act-deriv" ? dh[j] : dh[j] * d(c.Z[l - 1][j]);
    }
  }
  const N = X.length; let reg = 0;
  for (let l = 0; l < L; l++) { for (let k = 0; k < gW[l].length; k++) { reg += 0.5 * l2 * n.W[l][k] ** 2; gW[l][k] = gW[l][k] / N + l2 * n.W[l][k]; } for (let k = 0; k < gb[l].length; k++) gb[l][k] = (o.bug === "sign" ? -1 : 1) * gb[l][k] / N; }
  return { W: gW, b: gb, loss: loss / N + reg };
}
/** Largest relative difference between backprop and central finite differences. */
export function gradCheck(n: Net, X: number[][], y: number[], o: { l2?: number; bug?: Bug; count?: number } = {}): number {
  const g = lossGrad(n, X, y, o), h = 1e-5; let worst = 0;
  const all: [Float64Array, Float64Array][] = [...n.W.map((w, l) => [w, g.W[l]] as [Float64Array, Float64Array]), ...n.b.map((w, l) => [w, g.b[l]] as [Float64Array, Float64Array])];
  for (const [arr, gr] of all) for (let k = 0; k < Math.min(arr.length, o.count ?? 6); k++) {
    const v = arr[k]; arr[k] = v + h; const up = lossGrad(n, X, y, { l2: o.l2 }).loss; arr[k] = v - h; const dn = lossGrad(n, X, y, { l2: o.l2 }).loss; arr[k] = v;
    const num = (up - dn) / (2 * h); worst = Math.max(worst, Math.abs(num - gr[k]) / Math.max(1e-8, Math.abs(num) + Math.abs(gr[k])));
  }
  return worst;
}
export function lossOnly(n: Net, X: number[][], y: number[], l2 = 0): number {
  let loss = 0; for (let s = 0; s < X.length; s++) { const p = Math.min(Math.max(forward(n, X[s]).p, 1e-12), 1 - 1e-12); loss += -(y[s] * Math.log(p) + (1 - y[s]) * Math.log(1 - p)); }
  let reg = 0; if (l2) for (const w of n.W) for (const v of w) reg += 0.5 * l2 * v * v; return loss / X.length + reg;
}
export const centred = (X: number[][]) => X.map((p) => p.map((v) => 2 * v - 1));
export function layerGradNorms(n: Net, X: number[][], y: number[]): number[] { const g = lossGrad(n, X, y); return g.W.map((w) => Math.sqrt(w.reduce((s, v) => s + v * v, 0))); }
export function deadUnits(n: Net, X: number[][]): number { let dead = 0, tot = 0; const caches = X.map((x) => forward(n, x)); for (let l = 0; l < n.sizes.length - 2; l++) for (let j = 0; j < n.sizes[l + 1]; j++) { tot++; if (caches.every((c) => c.Z[l][j] <= 0)) dead++; } return tot ? dead / tot : 0; }

export type Method = "sgd" | "momentum" | "rmsprop" | "adam";
/** Optimiser over a list of flat parameter arrays. Same update rules as PyTorch's SGD(momentum), RMSprop and Adam. */
export class Opt {
  t = 0; m: Float64Array[] = []; v: Float64Array[] = [];
  method: Method; lr: number; o: { mu?: number; b1?: number; b2?: number; rho?: number; eps?: number };
  constructor(method: Method, lr: number, o: { mu?: number; b1?: number; b2?: number; rho?: number; eps?: number } = {}) { this.method = method; this.lr = lr; this.o = o; }
  step(params: Float64Array[], grads: Float64Array[]) {
    const { mu = 0.9, b1 = 0.9, b2 = 0.999, rho = 0.99, eps = 1e-8 } = this.o; this.t++;
    if (!this.m.length) { this.m = params.map((p) => new Float64Array(p.length)); this.v = params.map((p) => new Float64Array(p.length)); }
    params.forEach((p, a) => { for (let k = 0; k < p.length; k++) {
      const g = grads[a][k];
      if (this.method === "sgd") p[k] -= this.lr * g;
      else if (this.method === "momentum") { this.m[a][k] = mu * this.m[a][k] + g; p[k] -= this.lr * this.m[a][k]; }
      else if (this.method === "rmsprop") { this.v[a][k] = rho * this.v[a][k] + (1 - rho) * g * g; p[k] -= (this.lr * g) / (Math.sqrt(this.v[a][k]) + eps); }
      else { this.m[a][k] = b1 * this.m[a][k] + (1 - b1) * g; this.v[a][k] = b2 * this.v[a][k] + (1 - b2) * g * g; p[k] -= (this.lr * (this.m[a][k] / (1 - b1 ** this.t))) / (Math.sqrt(this.v[a][k] / (1 - b2 ** this.t)) + eps); }
    } });
  }
}
export function optimisePath(grad: (p: number[]) => number[], start: number[], method: Method, lr: number, steps: number): number[][] {
  const p = [Float64Array.from(start)], opt = new Opt(method, lr), path = [[...start]];
  for (let i = 0; i < steps; i++) { opt.step(p, [Float64Array.from(grad(Array.from(p[0])))]); path.push(Array.from(p[0])); if (!p[0].every(Number.isFinite)) break; }
  return path;
}
export interface TrainCfg { every?: number; epochs: number; lr: number; method?: Method; l2?: number; drop?: number; seed?: number; bug?: Bug; batch?: number; test?: { X: number[][]; y: number[] } }
export function trainNet(n: Net, X: number[][], y: number[], c: TrainCfg) {
  const opt = new Opt(c.method ?? "adam", c.lr), r = rng((c.seed ?? 1) * 31 + 7), hist = { loss: [] as number[], acc: [] as number[], testLoss: [] as number[], testAcc: [] as number[] }, params = [...n.W, ...n.b];
  const idx = X.map((_, i) => i), bs = c.batch ?? X.length;
  for (let e = 0; e < c.epochs; e++) {
    for (let i = idx.length - 1; i > 0 && bs < X.length; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    for (let s = 0; s < X.length; s += bs) { const ii = idx.slice(s, s + bs), g = lossGrad(n, ii.map((i) => X[i]), ii.map((i) => y[i]), { l2: c.l2, drop: c.drop, rand: r, bug: c.bug }); opt.step(params, [...g.W, ...g.b]); }
    if ((e + 1) % (c.every ?? 1) !== 0 && e !== c.epochs - 1) continue;
    hist.loss.push(lossOnly(n, X, y, c.l2)); hist.acc.push(netAccuracy(n, X, y));
    if (c.test) { hist.testLoss.push(lossOnly(n, c.test.X, c.test.y)); hist.testAcc.push(netAccuracy(n, c.test.X, c.test.y)); }
  }
  return hist;
}

/** Perceptron rule on 0/1 labels: update only when y*(w.x+b) <= 0. */
export function fitPerceptron(X: number[][], y01: number[], epochs: number) {
  const w = new Array(X[0].length).fill(0); let b = 0; const mistakes: number[] = [], hist: { w: number[]; b: number }[] = [{ w: [...w], b }];
  for (let e = 0; e < epochs; e++) { let m = 0; X.forEach((x, i) => { const t = y01[i] * 2 - 1; if (t * (w.reduce((s, wi, j) => s + wi * x[j], 0) + b) <= 0) { x.forEach((v, j) => (w[j] += t * v)); b += t; m++; } }); mistakes.push(m); hist.push({ w: [...w], b }); }
  return { w, b, mistakes, hist };
}
/** Cross-correlation (what CNN layers call convolution) with zero padding and stride. */
export function conv2d(img: number[][], k: number[][], stride = 1, pad = 0): number[][] {
  const H = img.length, W = img[0].length, kh = k.length, kw = k[0].length, P = Array.from({ length: H + 2 * pad }, (_, r) => Array.from({ length: W + 2 * pad }, (_, c) => img[r - pad]?.[c - pad] ?? 0));
  const oh = Math.floor((H + 2 * pad - kh) / stride) + 1, ow = Math.floor((W + 2 * pad - kw) / stride) + 1;
  return Array.from({ length: Math.max(oh, 0) }, (_, r) => Array.from({ length: Math.max(ow, 0) }, (_, c) => { let s = 0; for (let a = 0; a < kh; a++) for (let b = 0; b < kw; b++) s += P[r * stride + a][c * stride + b] * k[a][b]; return s; }));
}
export const convOutSize = (n: number, k: number, pad: number, stride: number) => Math.floor((n + 2 * pad - k) / stride) + 1;
export const maxPool2d = (m: number[][], size = 2) => Array.from({ length: Math.floor(m.length / size) }, (_, r) => Array.from({ length: Math.floor(m[0].length / size) }, (_, c) => { let v = -Infinity; for (let a = 0; a < size; a++) for (let b = 0; b < size; b++) v = Math.max(v, m[r * size + a][c * size + b]); return v; }));
