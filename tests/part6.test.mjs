import test from "node:test";
import assert from "node:assert/strict";
import { makeSeries, tsSplit, scoreSeries, forwardHMM, backwardHMM, posteriorHMM, viterbi, baumWelch, sampleHMM, bruteLikelihood } from "../lib/ml/seq.ts";
import { runBandit, learnCliff, greedyPath, valueIterationCliff, START, GOAL, CW } from "../lib/ml/rl.ts";
import { trainAE, makeArc, reconError, schedule, qSample, sampleMix, reverseSample, mixScore } from "../lib/ml/gen.ts";
import { shapleyExact, permutationImportance, makeFairness, groupStats, thresholdFor, ppvFromRates } from "../lib/ml/trust.ts";
import { makeCorpus, vocabOf, cooccur, ppmi, embedSym, cosine, nearest, analogy, softmax, attention, attentionSpread, randMatrix } from "../lib/ml/lang.ts";

const near = (a, b, tol, msg = "") => assert.ok(Math.abs(a - b) <= tol, `${msg} got ${a}, want ${b}`);

test("tsSplit matches scikit-learn TimeSeriesSplit (n=10, 3 splits)", () => {
  const s = tsSplit(10, 3);
  assert.deepEqual(s.map((x) => x.test), [[4, 5], [6, 7], [8, 9]]);
  assert.deepEqual(s.map((x) => x.train.length), [4, 6, 8]);
  assert.deepEqual(tsSplit(10, 3, 1).map((x) => x.train.length), [3, 5, 7]);
});
test("random folds flatter a series that peeks, and walk-forward is honest", () => {
  const y = makeSeries(144, 3);
  const honest = scoreSeries(y, "walk", false), shuffledPeek = scoreSeries(y, "shuffled", true);
  assert.ok(shuffledPeek < honest * 0.5, `peeking shuffled ${shuffledPeek} vs honest ${honest}`);
});

const hmm = { pi: [0.6, 0.4], A: [[0.7, 0.3], [0.4, 0.6]], B: [[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]] };
test("forward algorithm equals listing every path", () => {
  const obs = [0, 2, 1, 2, 0];
  near(Math.exp(forwardHMM(hmm, obs).loglik), bruteLikelihood(hmm, obs), 1e-12);
});
test("posteriors sum to one and viterbi path is the best path", () => {
  const obs = [0, 2, 1, 2, 0, 0, 2], post = posteriorHMM(hmm, obs);
  post.forEach((p) => near(p[0] + p[1], 1, 1e-12));
  const v = viterbi(hmm, obs);
  // best path by brute force
  let best = -Infinity, bp = null;
  const walk = (t, path, lp) => { if (t === obs.length) { if (lp > best) { best = lp; bp = [...path]; } return; } for (let j = 0; j < 2; j++) walk(t + 1, [...path, j], lp + Math.log(t === 0 ? hmm.pi[j] : hmm.A[path[t - 1]][j]) + Math.log(hmm.B[j][obs[t]])); };
  walk(0, [], 0);
  assert.deepEqual(v.path, bp); near(v.logp, best, 1e-9);
});
test("Baum-Welch never lowers the likelihood and beats a poor start", () => {
  const { obs } = sampleHMM(hmm, 400, 7), start = { pi: [0.5, 0.5], A: [[0.6, 0.4], [0.5, 0.5]], B: [[0.4, 0.3, 0.3], [0.3, 0.3, 0.4]] };
  const { hist } = baumWelch(start, obs, 40);
  for (let i = 1; i < hist.length; i++) assert.ok(hist[i] >= hist[i - 1] - 1e-9, `drop at ${i}`);
  assert.ok(hist.at(-1) > hist[0] + 5);
});
test("Baum-Welch from identical states cannot tell them apart", () => {
  const { obs } = sampleHMM(hmm, 300, 9), flat = { pi: [0.5, 0.5], A: [[0.5, 0.5], [0.5, 0.5]], B: [[0.3, 0.3, 0.4], [0.3, 0.3, 0.4]] };
  const { model } = baumWelch(flat, obs, 30);
  model.B[0].forEach((v, k) => near(v, model.B[1][k], 1e-9, "B rows stay equal"));
});

test("PPMI embeddings put similar words near each other", () => {
  const c = makeCorpus(400, 2), vocab = vocabOf(c), M = ppmi(cooccur(c, vocab, 2)), { vecs } = embedSym(M, 8);
  const n = nearest(vecs, vocab, "paris", 3).map((x) => x.w), n2 = nearest(vecs, vocab, "rome", 3).map((x) => x.w);
  assert.ok(["madrid", "berlin", "rome"].some((w) => n.includes(w)), n.join());
  assert.ok(["madrid", "berlin", "paris"].some((w) => n2.includes(w)), n2.join());
});
test("low-rank error equals the dropped eigenvalues (Eckart-Young)", () => {
  const c = makeCorpus(300, 2), vocab = vocabOf(c), M = ppmi(cooccur(c, vocab, 2));
  const k = 5, { keep, spectrum } = embedSym(M, k);
  const R = M.map((row, i) => row.map((_, j) => keep.reduce((s, e) => s + e.value * e.vector[i] * e.vector[j], 0)));
  let err = 0; M.forEach((row, i) => row.forEach((v, j) => { err += (v - R[i][j]) ** 2; }));
  near(Math.sqrt(err), Math.sqrt(spectrum.slice(k).reduce((s, v) => s + v * v, 0)), 1e-6);
});
test("softmax attention rows sum to one; identical keys give equal weights; causal mask hides the future", () => {
  near(softmax([1, 2, 3]).reduce((a, b) => a + b, 0), 1, 1e-12);
  const Q = [[1, 0], [0, 1], [1, 1]], K = [[1, 1], [1, 1], [1, 1]], V = [[1, 0], [0, 1], [1, 1]];
  attention(Q, K, V).weights.forEach((w) => w.forEach((v) => near(v, 1 / 3, 1e-12)));
  const c = attention(Q, [[1, 0], [0, 1], [1, 1]], V, { causal: true }).weights;
  assert.equal(c[0][1], 0); assert.equal(c[0][2], 0); assert.equal(c[1][2], 0);
});
test("attention matches a hand calculation", () => {
  const r = attention([[1, 0]], [[1, 0], [0, 1]], [[10, 0], [0, 10]], { scale: false });
  const w0 = Math.exp(1) / (Math.exp(1) + 1);
  near(r.weights[0][0], w0, 1e-12); near(r.out[0][0], 10 * w0, 1e-12);
});
test("dot products grow like sqrt(d); without scaling attention saturates, with it it does not", () => {
  const Q = randMatrix(2000, 64, 1), K = randMatrix(2000, 64, 2), dots = Q.map((q, i) => q.reduce((s, v, j) => s + v * K[i][j], 0)), sd = Math.sqrt(dots.reduce((s, v) => s + v * v, 0) / dots.length);
  near(sd, 8, 0.5);
  const raw = attentionSpread(512, 10, false), sc = attentionSpread(512, 10, true);
  assert.ok(raw.maxWeight > 0.95, `raw ${raw.maxWeight}`); assert.ok(sc.maxWeight < 0.6, `scaled ${sc.maxWeight}`);
});

test("bandits: a policy that never explores can lock onto a poor arm; UCB and Thompson keep regret low", () => {
  const means = [0.3, 0.5, 0.7], T = 1500;
  const reg = (p, seeds) => seeds.map((sd) => runBandit(means, p, T, sd).regret.at(-1));
  const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length, seeds = [1, 2, 3, 4, 5, 6, 7, 8];
  const g = avg(reg("greedy", seeds)), u = avg(reg("ucb", seeds)), t = avg(reg("thompson", seeds));
  assert.ok(g > 2 * t, `greedy ${g} vs thompson ${t}`); assert.ok(u < g, `ucb ${u} vs greedy ${g}`);
});
test("Q-learning on the cliff finds the shortest path; SARSA takes a safer, longer one; value iteration gives the same best return", () => {
  const V = valueIterationCliff(), best = Math.max(...V[START]);
  assert.equal(best, -13);
  const q = learnCliff("q", 1500, 3), sa = learnCliff("sarsa", 1500, 3);
  const pq = greedyPath(q.Q), ps = greedyPath(sa.Q);
  assert.equal(pq.at(-1), GOAL); assert.equal(ps.at(-1), GOAL);
  assert.equal(pq.length - 1, 13, `q path ${pq.length - 1}`);
  assert.ok(ps.length - 1 > 13, `sarsa path ${ps.length - 1}`);
  const nearCliff = (p) => p.filter((s) => Math.floor(s / CW) === 2).length;
  assert.ok(nearCliff(pq) > nearCliff(ps));
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  assert.ok(mean(sa.returns.slice(-300)) > mean(q.returns.slice(-300)), "while exploring, SARSA earns more");
});

test("a linear autoencoder with a one-number code reaches the PCA error", () => {
  const X = makeArc(150, 0.02, 2).map(([a, b]) => [3 * a, 0.4 * b]);
  const mean = [0, 1].map((j) => X.reduce((s, r) => s + r[j], 0) / X.length);
  const Xc = X.map((r) => r.map((x, j) => x - mean[j]));
  const cov = [[0, 0], [0, 0]]; Xc.forEach((r) => { for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) cov[i][j] += (r[i] * r[j]) / X.length; });
  const tr = cov[0][0] + cov[1][1], det = cov[0][0] * cov[1][1] - cov[0][1] ** 2, small = tr / 2 - Math.sqrt((tr / 2) ** 2 - det);
  const { ae } = trainAE(Xc, [2, 1, 2], ["linear", "linear"], { epochs: 1500, lr: 0.03, seed: 1 });
  assert.ok(reconError(ae, Xc) < small * 1.15 + 1e-6, `ae ${reconError(ae, Xc)} vs pca ${small}`);
});
test("a nonlinear autoencoder beats the best straight line on a curved arc", () => {
  const X = makeArc(120, 0.02, 2), mean = [0, 1].map((j) => X.reduce((s, r) => s + r[j], 0) / X.length), Xc = X.map((r) => r.map((x, j) => x - mean[j]));
  const cov = [[0, 0], [0, 0]]; Xc.forEach((r) => { for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) cov[i][j] += (r[i] * r[j]) / X.length; });
  const tr = cov[0][0] + cov[1][1], det = cov[0][0] * cov[1][1] - cov[0][1] ** 2, small = tr / 2 - Math.sqrt((tr / 2) ** 2 - det);
  const { ae } = trainAE(Xc, [2, 10, 1, 10, 2], ["tanh", "linear", "tanh", "linear"], { epochs: 900, lr: 0.02, seed: 1 });
  assert.ok(reconError(ae, Xc) < small * 0.5, `nonlinear ${reconError(ae, Xc)} vs line ${small}`);
});

test("diffusion: one-step noising matches running the chain", () => {
  const s = schedule(100, "linear"), t = 60, n = 20000, r = sampleMix({ w: [1], mu: [0], sd: [1] }, n * t, 5);
  const xs = Array.from({ length: n }, (_, i) => { let x = 2; for (let k = 0; k < t; k++) x = Math.sqrt(s.alphas[k]) * x + Math.sqrt(s.betas[k]) * r[i * t + k]; return x; });
  const m = xs.reduce((a, b) => a + b, 0) / n, v = xs.reduce((a, b) => a + (b - m) ** 2, 0) / n;
  near(m, Math.sqrt(s.abar[t - 1]) * 2, 0.03); near(v, 1 - s.abar[t - 1], 0.03);
  near(qSample(2, t, s, 0), Math.sqrt(s.abar[t - 1]) * 2, 1e-12);
});
test("diffusion: with the exact score, walking back recovers a two-bump distribution; stopping the noising early breaks it", () => {
  const mix = { w: [0.3, 0.7], mu: [-2, 2], sd: [0.3, 0.3] };
  const good = reverseSample(mix, schedule(200, "linear", 1), 3000, 4).samples, lo = good.filter((x) => x < 0).length / good.length;
  near(lo, 0.3, 0.05);
  const hi = good.filter((x) => x > 0), mh = hi.reduce((a, b) => a + b, 0) / hi.length; near(mh, 2, 0.1);
  const s2 = schedule(200, "linear", 0.04); assert.ok(s2.abar.at(-1) > 0.5);
  const bad = reverseSample(mix, s2, 3000, 4).samples, loBad = bad.filter((x) => x < 0).length / bad.length;
  assert.ok(loBad > 0.4, `the rarer bump should swell towards 50%, got ${loBad}`);
});

test("Shapley values add up to the prediction minus the average and match the linear formula", () => {
  const w = [2, -1, 0.5, 0], f = (x) => 1 + x.reduce((s, v, i) => s + w[i] * v, 0), bg = Array.from({ length: 30 }, (_, i) => [Math.sin(i), Math.cos(2 * i), i / 10, (i % 3)]), x = [1, 2, 3, 4];
  const { phi, base, value } = shapleyExact(f, x, bg), mean = [0, 1, 2, 3].map((j) => bg.reduce((s, r) => s + r[j], 0) / bg.length);
  near(phi.reduce((a, b) => a + b, 0), value - base, 1e-9);
  phi.forEach((p, i) => near(p, w[i] * (x[i] - mean[i]), 1e-9, `phi${i}`));
  assert.equal(phi[3], 0);
});
test("with two copies of a feature, permutation importance understates both while Shapley splits the credit", () => {
  const f = (x) => x[0] + x[1], X = Array.from({ length: 200 }, (_, i) => { const a = Math.sin(i * 1.7); return [a, a, Math.cos(i)]; }), y = X.map((r) => r[0] + r[1]);
  const { phi } = shapleyExact(f, [1, 1, 0.3], X);
  near(phi[0], phi[1], 1e-9);
  const imp = permutationImportance(f, X, y, 1, 4); assert.ok(imp[0] > 0 && imp[2] < 1e-9);
});

test("one threshold gives equal error rates but unequal selection and precision; forcing equal selection breaks the error rates", () => {
  const d = makeFairness(40000, [0.5, 0.3], 3), t = groupStats(d, [0, 0]);
  near(t[0].tpr, t[1].tpr, 0.03); near(t[0].fpr, t[1].fpr, 0.03);
  assert.ok(t[0].selection - t[1].selection > 0.08); assert.ok(t[0].ppv - t[1].ppv > 0.08);
  near(t[0].ppv, ppvFromRates(t[0].tpr, t[0].fpr, t[0].base), 1e-9);
  const th0 = thresholdFor(d, 0, "selection", t[0].selection), th1 = thresholdFor(d, 1, "selection", t[0].selection), f = groupStats(d, [th0, th1]);
  near(f[0].selection, f[1].selection, 0.01);
  assert.ok(Math.abs(f[0].tpr - f[1].tpr) > 0.1, `tpr gap ${Math.abs(f[0].tpr - f[1].tpr)}`);
});
