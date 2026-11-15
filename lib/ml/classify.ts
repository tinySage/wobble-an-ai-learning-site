import { rng, shuffled } from "./split.ts";
type M = number[][];

/* ---------- k-nearest neighbours ---------- */
export function knnProba(X: M, y: number[], q: number[], k: number): number {
  const d = X.map((p, i) => [(p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2, i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let s = 0; for (let i = 0; i < k; i++) s += y[d[i][1]]; return s / k;
}

/* ---------- Gaussian naive Bayes (matches scikit-learn's GaussianNB with var_smoothing 1e-9) ---------- */
export function fitGaussianNB(X: M, y: number[], cols?: number[]) {
  const p = X[0].length, use = cols ?? Array.from({ length: p }, (_, j) => j), cls = [0, 1], eps = 1e-9 * Math.max(...use.map((j) => { const m = X.reduce((a, r) => a + r[j], 0) / X.length; return X.reduce((a, r) => a + (r[j] - m) ** 2, 0) / X.length; }));
  const st = cls.map((c) => { const rows = X.filter((_, i) => y[i] === c), n = rows.length; return { prior: n / X.length, mu: use.map((j) => rows.reduce((a, r) => a + r[j], 0) / n), v: use.map((j) => { const m = rows.reduce((a, r) => a + r[j], 0) / n; return rows.reduce((a, r) => a + (r[j] - m) ** 2, 0) / n + eps; }) }; });
  return { proba(q: number[]) {
    const ll = st.map((s) => Math.log(s.prior) + use.reduce((a, j, t) => a - 0.5 * Math.log(2 * Math.PI * s.v[t]) - ((q[j] - s.mu[t]) ** 2) / (2 * s.v[t]), 0)), m = Math.max(...ll), e = ll.map((l) => Math.exp(l - m));
    return e[1] / (e[0] + e[1]);
  } };
}

/* ---------- classification tree (CART, Gini) ---------- */
export interface Node { leaf: boolean; p: number; n: number; f?: number; t?: number; l?: Node; r?: Node }
const gini = (n1: number, n: number) => { const p = n1 / n; return 2 * p * (1 - p); };
export function fitTree(X: M, y: number[], opt: { maxDepth?: number; minSplit?: number; features?: number; rand?: () => number } = {}): Node {
  const { maxDepth = 3, minSplit = 2, features, rand } = opt;
  const build = (idx: number[], depth: number): Node => {
    const n1 = idx.reduce((a, i) => a + y[i], 0), node: Node = { leaf: true, p: n1 / idx.length, n: idx.length };
    if (depth >= maxDepth || idx.length < minSplit || n1 === 0 || n1 === idx.length) return node;
    let cols = Array.from({ length: X[0].length }, (_, j) => j);
    if (features && rand) cols = cols.map((c) => [rand(), c]).sort((a, b) => a[0] - b[0]).slice(0, features).map((t) => t[1]);
    let best = { score: gini(n1, idx.length) * idx.length - 1e-12, f: -1, t: 0 };
    for (const f of cols) {
      const s = idx.slice().sort((a, b) => X[a][f] - X[b][f]); let ln = 0, l1 = 0;
      for (let i = 0; i < s.length - 1; i++) {
        ln++; l1 += y[s[i]]; if (X[s[i]][f] === X[s[i + 1]][f]) continue;
        const rn = s.length - ln, sc = gini(l1, ln) * ln + gini(n1 - l1, rn) * rn;
        if (sc < best.score) best = { score: sc, f, t: (X[s[i]][f] + X[s[i + 1]][f]) / 2 };
      }
    }
    if (best.f < 0) return node;
    return { leaf: false, p: node.p, n: node.n, f: best.f, t: best.t, l: build(idx.filter((i) => X[i][best.f] <= best.t), depth + 1), r: build(idx.filter((i) => X[i][best.f] > best.t), depth + 1) };
  };
  return build(Array.from({ length: X.length }, (_, i) => i), 0);
}
export function treeProba(t: Node, q: number[]): number { let n = t; while (!n.leaf) n = q[n.f!] <= n.t! ? n.l! : n.r!; return n.p; }
export const treeDepth = (t: Node): number => (t.leaf ? 0 : 1 + Math.max(treeDepth(t.l!), treeDepth(t.r!)));

/* ---------- random forest ---------- */
export function fitForest(X: M, y: number[], opt: { trees?: number; maxDepth?: number; seed?: number } = {}) {
  const { trees = 25, maxDepth = 6, seed = 1 } = opt, r = rng(seed), n = X.length;
  const ts = Array.from({ length: trees }, () => { const idx = Array.from({ length: n }, () => Math.floor(r() * n)); return fitTree(idx.map((i) => X[i]), idx.map((i) => y[i]), { maxDepth, features: 1, rand: r }); });
  return { trees: ts, proba: (q: number[]) => ts.reduce((a, t) => a + treeProba(t, q), 0) / ts.length };
}

/* ---------- gradient boosting for binary log loss (scikit-learn's recipe: MSE trees on residuals, Newton leaf values) ---------- */
interface RNode { leaf: boolean; v: number; f?: number; t?: number; l?: RNode; r?: RNode }
function fitRegTree(X: M, res: number[], hess: number[], depth: number): RNode {
  const build = (idx: number[], d: number): RNode => {
    const sr = idx.reduce((a, i) => a + res[i], 0), sh = idx.reduce((a, i) => a + hess[i], 0), leaf: RNode = { leaf: true, v: sh > 1e-12 ? sr / sh : 0 };
    if (d >= depth || idx.length < 2) return leaf;
    let best = { gain: 1e-12, f: -1, t: 0 };
    for (let f = 0; f < X[0].length; f++) {
      const s = idx.slice().sort((a, b) => X[a][f] - X[b][f]); let ls = 0;
      for (let i = 0; i < s.length - 1; i++) {
        ls += res[s[i]]; if (X[s[i]][f] === X[s[i + 1]][f]) continue;
        const ln = i + 1, rn = s.length - ln, g = (ls * ls) / ln + ((sr - ls) ** 2) / rn - (sr * sr) / s.length;
        if (g > best.gain) best = { gain: g, f, t: (X[s[i]][f] + X[s[i + 1]][f]) / 2 };
      }
    }
    if (best.f < 0) return leaf;
    return { leaf: false, v: leaf.v, f: best.f, t: best.t, l: build(idx.filter((i) => X[i][best.f] <= best.t), d + 1), r: build(idx.filter((i) => X[i][best.f] > best.t), d + 1) };
  };
  return build(Array.from({ length: X.length }, (_, i) => i), 0);
}
const regPredict = (t: RNode, q: number[]): number => { let n = t; while (!n.leaf) n = q[n.f!] <= n.t! ? n.l! : n.r!; return n.v; };
export function fitBoost(X: M, y: number[], opt: { rounds?: number; lr?: number; depth?: number } = {}) {
  const { rounds = 20, lr = 0.3, depth = 2 } = opt, p0 = y.reduce((a, b) => a + b, 0) / y.length, f0 = Math.log(p0 / (1 - p0)), F = new Array(y.length).fill(f0), ts: RNode[] = [];
  for (let m = 0; m < rounds; m++) {
    const p = F.map((v) => 1 / (1 + Math.exp(-v))), res = y.map((t, i) => t - p[i]), hess = p.map((v) => v * (1 - v)), t = fitRegTree(X, res, hess, depth);
    ts.push(t); X.forEach((row, i) => { F[i] += lr * regPredict(t, row); });
  }
  const raw = (q: number[], upTo = ts.length) => ts.slice(0, upTo).reduce((a, t) => a + lr * regPredict(t, q), f0);
  return { raw, proba: (q: number[], upTo = ts.length) => 1 / (1 + Math.exp(-raw(q, upTo))) };
}

/* ---------- soft-margin SVM by simplified SMO ---------- */
export function fitSVM(X: M, y01: number[], opt: { C?: number; gamma?: number | null; seed?: number; passes?: number } = {}) {
  const { C = 1, gamma = 1, seed = 1, passes = 12 } = opt, y = y01.map((t) => (t ? 1 : -1)), n = X.length, r = rng(seed);
  const k = (a: number[], b: number[]) => (gamma === null ? a[0] * b[0] + a[1] * b[1] : Math.exp(-gamma * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)));
  const K = X.map((a) => X.map((b) => k(a, b))), al = new Array(n).fill(0); let b = 0, still = 0, iter = 0;
  const f = (i: number) => { let s = b; for (let j = 0; j < n; j++) if (al[j]) s += al[j] * y[j] * K[j][i]; return s; };
  while (still < passes && iter++ < 400) {
    let changed = 0;
    for (let i = 0; i < n; i++) {
      const Ei = f(i) - y[i];
      if ((y[i] * Ei < -1e-3 && al[i] < C) || (y[i] * Ei > 1e-3 && al[i] > 0)) {
        let j = Math.floor(r() * (n - 1)); if (j >= i) j++;
        const Ej = f(j) - y[j], ai = al[i], aj = al[j];
        const L = y[i] === y[j] ? Math.max(0, ai + aj - C) : Math.max(0, aj - ai), Hh = y[i] === y[j] ? Math.min(C, ai + aj) : Math.min(C, C + aj - ai);
        if (L === Hh) continue;
        const eta = 2 * K[i][j] - K[i][i] - K[j][j]; if (eta >= 0) continue;
        let ajn = Math.max(L, Math.min(Hh, aj - (y[j] * (Ei - Ej)) / eta)); if (Math.abs(ajn - aj) < 1e-5) continue;
        const ain = ai + y[i] * y[j] * (aj - ajn), b1 = b - Ei - y[i] * (ain - ai) * K[i][i] - y[j] * (ajn - aj) * K[i][j], b2 = b - Ej - y[i] * (ain - ai) * K[i][j] - y[j] * (ajn - aj) * K[j][j];
        al[i] = ain; al[j] = ajn; b = ain > 0 && ain < C ? b1 : ajn > 0 && ajn < C ? b2 : (b1 + b2) / 2; changed++;
      }
    }
    still = changed ? 0 : still + 1;
  }
  const sv = al.map((a, i) => (a > 1e-6 ? i : -1)).filter((i) => i >= 0);
  return { sv, decision: (q: number[]) => sv.reduce((s, j) => s + al[j] * y[j] * k(X[j], q), b) };
}
