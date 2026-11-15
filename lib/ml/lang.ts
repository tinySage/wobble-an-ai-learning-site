import { rng } from "./split.ts";
import { eigSym } from "./unsup.ts";

const gauss = (r: () => number) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());

// ---------- Embeddings: count co-occurrences, take PPMI, factorise ----------

const ANIMALS: [string, string][] = [["cat", "fish"], ["dog", "bone"], ["horse", "hay"], ["cow", "grass"]];
const CAPITALS: [string, string][] = [["paris", "france"], ["rome", "italy"], ["madrid", "spain"], ["berlin", "germany"]];
const SIZES: [string, string][] = [["king", "queen"], ["man", "woman"], ["boy", "girl"], ["prince", "princess"]];

/** A small made-up corpus with clear structure: animals eat their food, capitals sit in countries, male and female pairs. */
export function makeCorpus(sentences = 400, seed = 2): string[][] {
  const r = rng(seed), out: string[][] = [], P = <T,>(a: T[]) => a[Math.floor(r() * a.length)];
  for (let i = 0; i < sentences; i++) {
    const k = r();
    if (k < 0.35) { const [a, f] = P(ANIMALS); out.push(["the", a, "eats", f]); }
    else if (k < 0.65) { const [c, n] = P(CAPITALS); out.push(["the", "city", c, "is", "in", n]); }
    else { const [m, f] = P(SIZES), fem = r() < 0.5; out.push(fem ? ["the", f, "is", "a", "she"] : ["the", m, "is", "a", "he"]); }
  }
  return out;
}

export const vocabOf = (corpus: string[][]) => [...new Set(corpus.flat())].sort();

/** Symmetric co-occurrence counts within a window. */
export function cooccur(corpus: string[][], vocab: string[], window = 2): number[][] {
  const ix = new Map(vocab.map((w, i) => [w, i])), C = vocab.map(() => new Array(vocab.length).fill(0));
  for (const s of corpus) for (let i = 0; i < s.length; i++) for (let j = Math.max(0, i - window); j <= Math.min(s.length - 1, i + window); j++) if (j !== i) C[ix.get(s[i])!][ix.get(s[j])!] += 1;
  return C;
}

/** Positive pointwise mutual information. Context counts are raised to ctxPower (0.75 in word2vec) to soften the pull of very frequent words. */
export function ppmi(C: number[][], ctxPower = 1): number[][] {
  const n = C.length, tot = C.reduce((s, r) => s + r.reduce((a, b) => a + b, 0), 0), rowS = C.map((r) => r.reduce((a, b) => a + b, 0));
  const colP = C[0].map((_, j) => C.reduce((s, r) => s + r[j], 0) ** ctxPower), colZ = colP.reduce((a, b) => a + b, 0);
  return C.map((row, i) => row.map((v, j) => (v > 0 ? Math.max(0, Math.log((v / tot) / ((rowS[i] / tot) * (colP[j] / colZ)))) : 0)));
}

/** Rank-k factorisation of a symmetric matrix by its largest eigenvalues (by size). Returns word vectors and the full spectrum. */
export function embedSym(M: number[][], k: number) {
  const { values, vectors } = eigSym(M), order = values.map((v, i) => i).sort((a, b) => Math.abs(values[b]) - Math.abs(values[a]));
  const keep = order.slice(0, k);
  return { vecs: M.map((_, w) => keep.map((i) => vectors[i][w] * Math.sqrt(Math.abs(values[i])))), spectrum: order.map((i) => values[i]), keep: keep.map((i) => ({ value: values[i], vector: vectors[i] })) };
}

export const cosine = (a: number[], b: number[]) => { const d = a.reduce((s, v, i) => s + v * b[i], 0), na = Math.hypot(...a), nb = Math.hypot(...b); return na && nb ? d / (na * nb) : 0; };

export function nearest(vecs: number[][], vocab: string[], word: string, n = 4, skip: string[] = []) {
  const q = vecs[vocab.indexOf(word)];
  return vocab.map((w, i) => ({ w, s: cosine(q, vecs[i]) })).filter((x) => x.w !== word && !skip.includes(x.w)).sort((a, b) => b.s - a.s).slice(0, n);
}

/** a is to b as c is to ?  (vector arithmetic b - a + c, nearest word that is not one of the three). */
export function analogy(vecs: number[][], vocab: string[], a: string, b: string, c: string, n = 3) {
  const [va, vb, vc] = [a, b, c].map((w) => vecs[vocab.indexOf(w)]), t = vb.map((v, i) => v - va[i] + vc[i]);
  return vocab.map((w, i) => ({ w, s: cosine(t, vecs[i]) })).filter((x) => ![a, b, c].includes(x.w)).sort((x, y) => y.s - x.s).slice(0, n);
}

// ---------- Attention ----------

export function softmax(z: number[]): number[] { const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); }
export const entropy = (p: number[]) => -p.reduce((s, v) => s + (v > 0 ? v * Math.log(v) : 0), 0);

/** softmax(Q K^T / sqrt(dk)) V. scale=false drops the division. causal hides later positions. */
export function attention(Q: number[][], K: number[][], V: number[][], opts: { scale?: boolean; causal?: boolean } = {}) {
  const { scale = true, causal = false } = opts, dk = K[0].length;
  const scores = Q.map((q) => K.map((k) => q.reduce((s, v, i) => s + v * k[i], 0) / (scale ? Math.sqrt(dk) : 1)));
  const weights = scores.map((row, i) => softmax(causal ? row.map((v, j) => (j > i ? -Infinity : v)) : row));
  const out = weights.map((w) => V[0].map((_, c) => w.reduce((s, wj, j) => s + wj * V[j][c], 0)));
  return { scores, weights, out };
}

export const randMatrix = (rows: number, cols: number, seed: number) => { const r = rng(seed); return Array.from({ length: rows }, () => Array.from({ length: cols }, () => gauss(r))); };

/** How peaked attention is for random vectors of dimension d: the average largest weight and entropy. */
export function attentionSpread(d: number, n = 10, scale = true, seed = 5, trials = 40) {
  let mx = 0, en = 0;
  for (let t = 0; t < trials; t++) { const q = randMatrix(1, d, seed + 2 * t), K = randMatrix(n, d, seed + 2 * t + 1), w = attention(q, K, K, { scale }).weights[0]; mx += Math.max(...w); en += entropy(w); }
  return { maxWeight: mx / trials, entropy: en / trials, uniform: Math.log(n) };
}
