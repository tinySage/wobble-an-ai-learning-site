import { rng } from "./split.ts";

const gauss = (r: () => number) => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
/** Gamma(shape >= 1, 1) by Marsaglia and Tsang. */
function gamma(shape: number, r: () => number): number {
  const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) { let x: number, v: number; do { x = gauss(r); v = 1 + c * x; } while (v <= 0); v = v * v * v; const u = r(); if (Math.log(u + 1e-300) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v; }
}
export const betaSample = (a: number, b: number, r: () => number) => { const x = gamma(a, r), y = gamma(b, r); return x / (x + y); };

// ---------- Bandits ----------
export type Policy = "greedy" | "epsilon" | "ucb" | "thompson";

/** Bernoulli bandit. Returns cumulative regret against always pulling the best arm, plus how often each arm was pulled. */
export function runBandit(means: number[], policy: Policy, steps: number, seed: number, opts: { eps?: number; c?: number } = {}) {
  const { eps = 0.1, c = 2 } = opts, r = rng(seed), K = means.length, n = new Array(K).fill(0), s = new Array(K).fill(0), best = Math.max(...means);
  const regret: number[] = []; let reg = 0;
  const argmax = (v: number[]) => { const m = Math.max(...v), ties = v.map((x, i) => (x === m ? i : -1)).filter((i) => i >= 0); return ties[Math.floor(r() * ties.length)]; };
  for (let t = 1; t <= steps; t++) {
    let a: number;
    if (policy === "thompson") a = argmax(means.map((_, i) => betaSample(1 + s[i], 1 + n[i] - s[i], r)));
    else if (policy === "ucb") a = n.findIndex((v) => v === 0) >= 0 ? n.findIndex((v) => v === 0) : argmax(means.map((_, i) => s[i] / n[i] + c * Math.sqrt(Math.log(t) / n[i])));
    else if (policy === "epsilon" && r() < eps) a = Math.floor(r() * K);
    else a = argmax(means.map((_, i) => (n[i] ? s[i] / n[i] : 0)));
    const win = r() < means[a] ? 1 : 0; n[a]++; s[a] += win; reg += best - means[a]; regret.push(reg);
  }
  return { regret, counts: n };
}

// ---------- Cliff walking (Sutton and Barto, example 6.6) ----------
export const CW = 12, CH = 4, START = (CH - 1) * CW, GOAL = (CH - 1) * CW + CW - 1;
export const isCliff = (s: number) => Math.floor(s / CW) === CH - 1 && s % CW > 0 && s % CW < CW - 1;
const DR = [-1, 0, 1, 0], DC = [0, 1, 0, -1];
/** One move. Falling off the cliff costs 100 and sends the walker back to the start. Every other step costs 1. */
export function cliffStep(s: number, a: number): { s2: number; r: number; done: boolean } {
  const r0 = Math.floor(s / CW), c0 = s % CW, r1 = Math.min(CH - 1, Math.max(0, r0 + DR[a])), c1 = Math.min(CW - 1, Math.max(0, c0 + DC[a])), s2 = r1 * CW + c1;
  if (isCliff(s2)) return { s2: START, r: -100, done: false };
  return { s2, r: -1, done: s2 === GOAL };
}

export type Learner = "q" | "sarsa";
export function learnCliff(kind: Learner, episodes: number, seed: number, opts: { eps?: number; alpha?: number; gamma?: number } = {}) {
  const { eps = 0.1, alpha = 0.5, gamma = 1 } = opts, r = rng(seed), Q = Array.from({ length: CW * CH }, () => new Array(4).fill(0)), returns: number[] = [];
  const pick = (s: number) => { if (r() < eps) return Math.floor(r() * 4); const m = Math.max(...Q[s]), t = Q[s].map((v, i) => (v === m ? i : -1)).filter((i) => i >= 0); return t[Math.floor(r() * t.length)]; };
  for (let e = 0; e < episodes; e++) {
    let s = START, a = pick(s), G = 0;
    for (let k = 0; k < 2000; k++) {
      const { s2, r: rew, done } = cliffStep(s, a), a2 = pick(s2); G += rew;
      const tgt = done ? 0 : kind === "q" ? Math.max(...Q[s2]) : Q[s2][a2];
      Q[s][a] += alpha * (rew + gamma * tgt - Q[s][a]); s = s2; a = a2; if (done) break;
    }
    returns.push(G);
  }
  return { Q, returns };
}

/** Follow the greedy policy from the start. Stops at the goal or after a cap. */
export function greedyPath(Q: number[][], cap = 60): number[] {
  const path = [START]; let s = START;
  for (let k = 0; k < cap; k++) { const a = Q[s].indexOf(Math.max(...Q[s])), { s2, done } = cliffStep(s, a); path.push(s2); if (done) break; s = s2; }
  return path;
}

/** Exact optimal action values by value iteration, for checking the learners. */
export function valueIterationCliff(iters = 200) {
  const Q = Array.from({ length: CW * CH }, () => new Array(4).fill(0));
  for (let it = 0; it < iters; it++) for (let s = 0; s < CW * CH; s++) { if (s === GOAL || isCliff(s)) continue; for (let a = 0; a < 4; a++) { const { s2, r, done } = cliffStep(s, a); Q[s][a] = r + (done ? 0 : Math.max(...Q[s2])); } }
  return Q;
}
