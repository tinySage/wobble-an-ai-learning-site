import { rng } from "./split.ts";
export type Pts = number[][];
const d2 = (a: number[], b: number[]) => a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0);
const gauss = (r: () => number) => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());

export type Shape = "blobs" | "stretched" | "moons" | "circles" | "uneven";
/** Unlabelled 2D clouds in roughly [0,1]^2, plus the true groups for scoring. */
export function makeCloud(shape: Shape, n = 150, seed = 3): { X: Pts; truth: number[] } {
  const r = rng(seed * 977 + shape.length), X: Pts = [], truth: number[] = [];
  for (let i = 0; i < n; i++) {
    let x = 0, y = 0, g: number = i % 3;
    if (shape === "blobs") { const c = [[0.22, 0.28], [0.74, 0.3], [0.5, 0.78]][g]; x = c[0] + gauss(r) * 0.07; y = c[1] + gauss(r) * 0.07; }
    else if (shape === "uneven") { g = i % 5 === 0 ? 2 : i % 5 < 3 ? 0 : 1; const c = [[0.25, 0.3, 0.05], [0.7, 0.35, 0.12], [0.5, 0.82, 0.03]][g]; x = c[0] + gauss(r) * c[2]; y = c[1] + gauss(r) * c[2]; }
    else if (shape === "stretched") { const c = [[0.2, 0.25], [0.5, 0.5], [0.8, 0.75]][g]; const u = gauss(r) * 0.17, v = gauss(r) * 0.03; x = c[0] + u * 0.8 - v * 0.6 + 0.03; y = c[1] - u * 0.6 - v * 0.8 + 0.03 + g * 0; }
    else if (shape === "moons") { g = i % 2; const t = r() * Math.PI; x = g ? 0.7 - Math.cos(t) * 0.3 : 0.3 + Math.cos(t) * 0.3; y = g ? 0.58 - Math.sin(t) * 0.22 : 0.42 + Math.sin(t) * 0.22; x += gauss(r) * 0.025; y += gauss(r) * 0.025; }
    else { g = i % 2; const t = r() * 2 * Math.PI, rad = g ? 0.36 : 0.14; x = 0.5 + Math.cos(t) * rad + gauss(r) * 0.015; y = 0.5 + Math.sin(t) * rad + gauss(r) * 0.015; }
    X.push([x, y]); truth.push(g);
  }
  return { X, truth };
}
/** d-dimensional Gaussian clusters with chosen spreads: for t-SNE and PCA demos. */
export function makeHighD(n = 120, dim = 8, spreads = [0.3, 0.9, 0.6], seed = 5): { X: Pts; truth: number[] } {
  const r = rng(seed), k = spreads.length, cen = Array.from({ length: k }, () => Array.from({ length: dim }, () => gauss(r) * 3));
  const truth = Array.from({ length: n }, (_, i) => i % k);
  return { X: truth.map((g) => cen[g].map((c) => c + gauss(r) * spreads[g])), truth };
}

/** Symmetric eigen decomposition by cyclic Jacobi. Values sorted high to low. */
export function eigSym(A: number[][]): { values: number[]; vectors: number[][] } {
  const n = A.length, a = A.map((r) => [...r]), v: number[][] = A.map((_, i) => A.map((__, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const kp = a[k][p], kq = a[k][q]; a[k][p] = c * kp - s * kq; a[k][q] = s * kp + c * kq; }
      for (let k = 0; k < n; k++) { const pk = a[p][k], qk = a[q][k]; a[p][k] = c * pk - s * qk; a[q][k] = s * pk + c * qk; }
      for (let k = 0; k < n; k++) { const kp = v[k][p], kq = v[k][q]; v[k][p] = c * kp - s * kq; v[k][q] = s * kp + c * kq; }
    }
  }
  const idx = a.map((_, i) => i).sort((i, j) => a[j][j] - a[i][i]);
  return { values: idx.map((i) => a[i][i]), vectors: idx.map((i) => v.map((row) => row[i])) };
}

export function fitPCA(X: Pts) {
  const n = X.length, d = X[0].length, mu = Array.from({ length: d }, (_, j) => X.reduce((s, r) => s + r[j], 0) / n);
  const Z = X.map((r) => r.map((v, j) => v - mu[j])), cov = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => Z.reduce((s, r) => s + r[i] * r[j], 0) / (n - 1)));
  const { values, vectors } = eigSym(cov), tot = values.reduce((s, v) => s + Math.max(v, 0), 0);
  return { mean: mu, values, components: vectors, ratio: values.map((v) => Math.max(v, 0) / tot),
    project: (q: number[], k: number) => vectors.slice(0, k).map((c) => c.reduce((s, w, j) => s + w * (q[j] - mu[j]), 0)),
    reconstruct: (q: number[], k: number) => { const z = vectors.slice(0, k).map((c) => c.reduce((s, w, j) => s + w * (q[j] - mu[j]), 0)); return mu.map((m, j) => m + z.reduce((s, zi, i) => s + zi * vectors[i][j], 0)); } };
}
export const standardiseCols = (X: Pts) => { const d = X[0].length, n = X.length, m = Array.from({ length: d }, (_, j) => X.reduce((s, r) => s + r[j], 0) / n), sd = m.map((mj, j) => Math.sqrt(X.reduce((s, r) => s + (r[j] - mj) ** 2, 0) / n) || 1); return X.map((r) => r.map((v, j) => (v - m[j]) / sd[j])); };

export function kmeansPP(X: Pts, k: number, seed: number): Pts {
  const r = rng(seed), C = [X[Math.floor(r() * X.length)]];
  while (C.length < k) { const w = X.map((p) => Math.min(...C.map((c) => d2(p, c)))), tot = w.reduce((s, v) => s + v, 0); let t = r() * tot, i = 0; while (i < w.length - 1 && (t -= w[i]) > 0) i++; C.push(X[i]); }
  return C.map((c) => [...c]);
}
export function kmeans(X: Pts, k: number, o: { seed?: number; init?: Pts; maxIter?: number; random?: boolean } = {}) {
  const d = X[0].length; let C = o.init ? o.init.map((c) => [...c]) : o.random ? (() => { const r = rng(o.seed ?? 1); return Array.from({ length: k }, () => [...X[Math.floor(r() * X.length)]]); })() : kmeansPP(X, k, o.seed ?? 1);
  let labels: number[] = [], iters = 0; const max = o.maxIter ?? 300;
  const assign = () => X.map((p) => { let b = 0, bd = Infinity; C.forEach((c, j) => { const v = d2(p, c); if (v < bd) { bd = v; b = j; } }); return b; });
  for (; iters < max; iters++) {
    const nl = assign(); if (nl.every((v, i) => v === labels[i])) { labels = nl; break; } labels = nl;
    const S = Array.from({ length: k }, () => new Array(d).fill(0)), N = new Array(k).fill(0);
    X.forEach((p, i) => { N[labels[i]]++; p.forEach((v, j) => (S[labels[i]][j] += v)); });
    C = C.map((c, j) => (N[j] ? S[j].map((v) => v / N[j]) : c));
  }
  labels = assign();
  return { centers: C, labels, inertia: X.reduce((s, p, i) => s + d2(p, C[labels[i]]), 0), iterations: iters };
}
export function silhouette(X: Pts, labels: number[]): number {
  const ids = [...new Set(labels)].filter((l) => l >= 0); if (ids.length < 2) return 0;
  const keep = X.map((_, i) => i).filter((i) => labels[i] >= 0); let s = 0;
  for (const i of keep) {
    const own = labels[i], sum: Record<number, number> = {}, cnt: Record<number, number> = {};
    for (const j of keep) if (j !== i) { sum[labels[j]] = (sum[labels[j]] ?? 0) + Math.sqrt(d2(X[i], X[j])); cnt[labels[j]] = (cnt[labels[j]] ?? 0) + 1; }
    if (!cnt[own]) continue;
    const a = sum[own] / cnt[own], b = Math.min(...ids.filter((l) => l !== own && cnt[l]).map((l) => sum[l] / cnt[l]));
    s += (b - a) / Math.max(a, b);
  }
  return s / keep.length;
}
export function dbscan(X: Pts, eps: number, minSamples: number): number[] {
  const n = X.length, nb = X.map((p) => X.map((q, j) => (d2(p, q) <= eps * eps ? j : -1)).filter((j) => j >= 0)), lab = new Array(n).fill(-1), seen = new Array(n).fill(false); let c = 0;
  for (let i = 0; i < n; i++) {
    if (seen[i] || nb[i].length < minSamples) continue;
    const stack = [i]; seen[i] = true; lab[i] = c;
    while (stack.length) { const p = stack.pop()!; for (const q of nb[p]) { if (lab[q] < 0) lab[q] = c; if (!seen[q] && nb[q].length >= minSamples) { seen[q] = true; stack.push(q); } } }
    c++;
  }
  return lab;
}
export type Linkage = "single" | "complete" | "average" | "ward";
export function agglomerative(X: Pts, k: number, linkage: Linkage): number[] {
  const n = X.length, D = X.map((p) => X.map((q) => (linkage === "ward" ? d2(p, q) : Math.sqrt(d2(p, q))))), size = new Array(n).fill(1), alive = new Array(n).fill(true), grp = X.map((_, i) => [i]);
  let m = n;
  while (m > k) {
    let bi = -1, bj = -1, bd = Infinity;
    for (let i = 0; i < n; i++) if (alive[i]) for (let j = i + 1; j < n; j++) if (alive[j] && D[i][j] < bd) { bd = D[i][j]; bi = i; bj = j; }
    for (let x = 0; x < n; x++) if (alive[x] && x !== bi && x !== bj) {
      const a = D[bi][x], b = D[bj][x], ni = size[bi], nj = size[bj], nx = size[x];
      const v = linkage === "single" ? Math.min(a, b) : linkage === "complete" ? Math.max(a, b) : linkage === "average" ? (ni * a + nj * b) / (ni + nj) : ((ni + nx) * a + (nj + nx) * b - nx * bd) / (ni + nj + nx);
      D[bi][x] = D[x][bi] = v;
    }
    size[bi] += size[bj]; grp[bi] = grp[bi].concat(grp[bj]); alive[bj] = false; m--;
  }
  const lab = new Array(n).fill(0); let c = 0; for (let i = 0; i < n; i++) if (alive[i]) { grp[i].forEach((p) => (lab[p] = c)); c++; }
  return lab;
}

export type Cov = "full" | "diag" | "spherical";
export interface GmmInit { weights: number[]; means: Pts; covs: number[][][] }
function logDens(x: number[], mu: number[], S: number[][], inv: number[][], logdet: number) { const d = x.length, df = x.map((v, i) => v - mu[i]); let q = 0; for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) q += df[i] * inv[i][j] * df[j]; return -0.5 * (d * Math.log(2 * Math.PI) + logdet + q); }
function inv2(S: number[][]) { const d = S.length; if (d === 1) return { inv: [[1 / S[0][0]]], logdet: Math.log(S[0][0]) }; const [[a, b], [c, e]] = S, det = a * e - b * c; return { inv: [[e / det, -b / det], [-c / det, a / det]], logdet: Math.log(det) }; }
/** EM for a Gaussian mixture in 2D (diag/spherical/full), same recipe as scikit-learn with reg_covar = 1e-6. */
export function fitGMM(X: Pts, k: number, o: { cov?: Cov; seed?: number; init?: GmmInit; maxIter?: number; tol?: number } = {}) {
  const n = X.length, d = X[0].length, cov = o.cov ?? "full", reg = 1e-6, maxIter = o.maxIter ?? 100, tol = o.tol ?? 1e-3;
  const shape = (S: number[][]) => cov === "full" ? S : cov === "diag" ? S.map((r, i) => r.map((_, j) => (i === j ? S[i][i] : 0))) : S.map((r, i) => r.map((_, j) => (i === j ? S.reduce((s, row, t) => s + row[t], 0) / d : 0)));
  let w: number[], mu: Pts, S: number[][][];
  const mstep = (R: number[][]) => {
    const Nk = Array.from({ length: k }, (_, j) => R.reduce((s, r) => s + r[j], 0) + 10 * Number.EPSILON);
    w = Nk.map((v) => v / n); mu = Nk.map((nk, j) => X.reduce((s, p, i) => s.map((v, t) => v + R[i][j] * p[t]), new Array(d).fill(0)).map((v) => v / nk));
    S = Nk.map((nk, j) => { const C = Array.from({ length: d }, (_, a) => Array.from({ length: d }, (_, b) => X.reduce((s, p, i) => s + R[i][j] * (p[a] - mu[j][a]) * (p[b] - mu[j][b]), 0) / nk)); for (let a = 0; a < d; a++) C[a][a] += reg; return shape(C); });
  };
  let R: number[][];
  if (o.init) { w = o.init.weights; mu = o.init.means; S = o.init.covs.map(shape); R = []; }
  else { const km = kmeans(X, k, { seed: o.seed ?? 1 }); R = X.map((_, i) => Array.from({ length: k }, (_, j) => (km.labels[i] === j ? 1 : 0))); mstep(R); }
  const estep = () => {
    const pre = S!.map((c) => inv2(c)); let ll = 0;
    R = X.map((p) => { const lp = mu!.map((m, j) => Math.log(w![j]) + logDens(p, m, S![j], pre[j].inv, pre[j].logdet)), mx = Math.max(...lp), s = lp.reduce((a, v) => a + Math.exp(v - mx), 0), ls = mx + Math.log(s); ll += ls; return lp.map((v) => Math.exp(v - ls)); });
    return ll / n;
  };
  let prev = -Infinity, cur = estep(), it = 0;
  for (; it < maxIter; it++) { mstep(R); prev = cur; cur = estep(); if (Math.abs(cur - prev) < tol) break; }
  const p = cov === "full" ? k * d * (d + 1) / 2 + k * d + k - 1 : cov === "diag" ? 2 * k * d + k - 1 : k + k * d + k - 1, total = cur * n;
  return { weights: w!, means: mu!, covs: S!, resp: R, loglik: total, perSample: cur, bic: -2 * total + p * Math.log(n), aic: -2 * total + 2 * p, iterations: it,
    labels: R.map((r) => r.indexOf(Math.max(...r))), logProb: (q: number[]) => { const pre = S!.map((c) => inv2(c)), lp = mu!.map((m, j) => Math.log(w![j]) + logDens(q, m, S![j], pre[j].inv, pre[j].logdet)), mx = Math.max(...lp); return mx + Math.log(lp.reduce((a, v) => a + Math.exp(v - mx), 0)); } };
}

const cn = (m: number) => (m <= 1 ? 0 : 2 * (Math.log(m - 1) + 0.5772156649) - (2 * (m - 1)) / m);
/** Isolation forest (Liu et al.): returns a scorer where closer to 1 means more anomalous. */
export function fitIsolation(X: Pts, o: { trees?: number; sample?: number; seed?: number } = {}) {
  const r = rng(o.seed ?? 1), T = o.trees ?? 100, psi = Math.min(o.sample ?? 256, X.length), lim = Math.ceil(Math.log2(psi)), d = X[0].length;
  type N = { f: number; t: number; l: N; r: N } | { size: number };
  const build = (P: Pts, depth: number): N => {
    if (depth >= lim || P.length <= 1) return { size: P.length };
    const f = Math.floor(r() * d), vals = P.map((p) => p[f]), lo = Math.min(...vals), hi = Math.max(...vals); if (lo === hi) return { size: P.length };
    const t = lo + r() * (hi - lo); return { f, t, l: build(P.filter((p) => p[f] < t), depth + 1), r: build(P.filter((p) => p[f] >= t), depth + 1) };
  };
  const trees = Array.from({ length: T }, () => { const s = [...X.keys()]; for (let i = s.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [s[i], s[j]] = [s[j], s[i]]; } return build(s.slice(0, psi).map((i) => X[i]), 0); });
  const path = (q: number[], t: N, depth = 0): number => ("size" in t ? depth + cn(t.size) : path(q, q[t.f] < t.t ? t.l : t.r, depth + 1));
  return { score: (q: number[]) => 2 ** (-(trees.reduce((s, t) => s + path(q, t), 0) / T) / cn(psi)) };
}

/** Joint affinities P for t-SNE, built exactly as scikit-learn does. */
export function tsneAffinities(X: Pts, perplexity: number): number[][] {
  const n = X.length, D = X.map((p) => X.map((q) => d2(p, q))), target = Math.log(perplexity), P = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    let lo = -Infinity, hi = Infinity, beta = 1, row = new Array(n).fill(0);
    for (let it = 0; it < 100; it++) {
      let s = 0; for (let j = 0; j < n; j++) { row[j] = j === i ? 0 : Math.exp(-D[i][j] * beta); s += row[j]; }
      s = Math.max(s, 1e-300); let H = Math.log(s); for (let j = 0; j < n; j++) H += (beta * D[i][j] * row[j]) / s; for (let j = 0; j < n; j++) row[j] /= s;
      if (Math.abs(H - target) < 1e-5) break;
      if (H > target) { lo = beta; beta = hi === Infinity ? beta * 2 : (beta + hi) / 2; } else { hi = beta; beta = lo === -Infinity ? beta / 2 : (beta + lo) / 2; }
    }
    P[i] = row;
  }
  return P.map((r, i) => r.map((_, j) => (P[i][j] + P[j][i]) / (2 * n)));
}
export function tsne(X: Pts, o: { perplexity?: number; iters?: number; seed?: number } = {}): Pts {
  const n = X.length, perp = Math.min(o.perplexity ?? 30, (n - 1) / 3), iters = o.iters ?? 400, P = tsneAffinities(X, perp), r = rng(o.seed ?? 1);
  const pc = fitPCA(X), Y: Pts = X.map((p) => pc.project(p, 2).map((v) => v)); const s = Math.sqrt(Y.reduce((a, y) => a + y[0] ** 2, 0) / n) || 1; Y.forEach((y) => { y[0] = (y[0] / s) * 1e-2 + (r() - 0.5) * 1e-4; y[1] = (y[1] / s) * 1e-2 + (r() - 0.5) * 1e-4; });
  const V = Y.map(() => [0, 0]), G = Y.map(() => [1, 1]), lr = Math.max(n / 12 / 4, 50), exagEnd = Math.floor(iters / 4);
  for (let it = 0; it < iters; it++) {
    const ex = it < exagEnd ? 12 : 1, mom = it < exagEnd ? 0.5 : 0.8; let Z = 0; const Qn = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const q = 1 / (1 + d2(Y[i], Y[j])); Qn[i][j] = Qn[j][i] = q; Z += 2 * q; }
    const grad = Y.map((yi, i) => { const g = [0, 0]; for (let j = 0; j < n; j++) if (j !== i) { const m = (ex * P[i][j] - Qn[i][j] / Z) * Qn[i][j]; g[0] += m * (yi[0] - Y[j][0]); g[1] += m * (yi[1] - Y[j][1]); } return [4 * g[0], 4 * g[1]]; });
    for (let i = 0; i < n; i++) for (let t = 0; t < 2; t++) { const gr = grad[i][t]; G[i][t] = V[i][t] * gr < 0 ? G[i][t] + 0.2 : Math.max(G[i][t] * 0.8, 0.01); V[i][t] = mom * V[i][t] - lr * G[i][t] * gr; Y[i][t] += V[i][t]; }
  }
  return Y;
}
export const knnPreserved = (A: Pts, B: Pts, k = 10) => { const nn = (S: Pts, i: number) => S.map((q, j) => [d2(S[i], q), j]).filter(([, j]) => j !== i).sort((a, b) => a[0] - b[0]).slice(0, k).map(([, j]) => j); let s = 0; for (let i = 0; i < A.length; i++) { const a = new Set(nn(A, i)); s += nn(B, i).filter((j) => a.has(j)).length / k; } return s / A.length; };

/** Adjusted Rand index between two labelings (1 = same grouping, about 0 = no better than chance). Noise (-1) counts as its own group. */
export function adjustedRand(a: number[], b: number[]): number {
  const n = a.length, tab = new Map<string, number>(), ra = new Map<number, number>(), rb = new Map<number, number>();
  a.forEach((x, i) => { const k = x + "|" + b[i]; tab.set(k, (tab.get(k) ?? 0) + 1); ra.set(x, (ra.get(x) ?? 0) + 1); rb.set(b[i], (rb.get(b[i]) ?? 0) + 1); });
  const c2 = (v: number) => (v * (v - 1)) / 2, s = [...tab.values()].reduce((t, v) => t + c2(v), 0), sa = [...ra.values()].reduce((t, v) => t + c2(v), 0), sb = [...rb.values()].reduce((t, v) => t + c2(v), 0), e = (sa * sb) / c2(n), m = (sa + sb) / 2;
  return m === e ? 1 : (s - e) / (m - e);
}
