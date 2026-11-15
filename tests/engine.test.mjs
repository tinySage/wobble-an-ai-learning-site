import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { fitLinear, polyFeatures, predict, fitLasso, fitPolynomial, mse } from "../lib/ml/regression.ts";
import { gradientDescent, diverged } from "../lib/ml/gradient.ts";
import { standardise } from "../lib/ml/scale.ts";
import { kFold, trainTestSplit, timeSplit, walkForward, rng } from "../lib/ml/split.ts";
import { rocAuc, precisionRecallF1, confusion, rocCurve } from "../lib/ml/metrics.ts";
import { fitLogistic } from "../lib/ml/logistic.ts";
import { kmeans, silhouette, dbscan, agglomerative, fitPCA, fitGMM, tsneAffinities, tsne, knnPreserved, fitIsolation, makeHighD, adjustedRand } from "../lib/ml/unsup.ts";
import { forward, makeNet, lossGrad, gradCheck, Opt, optimisePath, fitPerceptron, conv2d, convOutSize, layerGradNorms, trainNet, netAccuracy, deadUnits } from "../lib/ml/nn.ts";

const F = JSON.parse(fs.readFileSync("tests/fixtures/sklearn.json", "utf8"));
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${msg ?? ""} got ${a}, want ${b}`);
const closeAll = (a, b, tol, msg) => { assert.equal(a.length, b.length); a.forEach((v, i) => close(v, b[i], tol, `${msg}[${i}]`)); };

test("OLS polynomial matches scikit-learn LinearRegression", () => {
  const { x, y, grid, ols3 } = F.poly, m = fitLinear(polyFeatures(x, 3), y);
  closeAll(m.coef, ols3.coef, 1e-7, "coef"); close(m.intercept, ols3.intercept, 1e-7, "intercept");
  closeAll(predict(m, polyFeatures(grid, 3)), ols3.pred, 1e-7, "pred");
});
test("ridge matches scikit-learn Ridge", () => {
  const { x, y, grid, ridge5 } = F.poly, m = fitLinear(polyFeatures(x, 5), y, ridge5.alpha);
  closeAll(m.coef, ridge5.coef, 1e-6, "coef"); close(m.intercept, ridge5.intercept, 1e-6, "intercept");
  closeAll(predict(m, polyFeatures(grid, 5)), ridge5.pred, 1e-6, "pred");
});
test("lasso matches scikit-learn Lasso, including exact zeros", () => {
  const { X, y, alpha, coef, intercept } = F.lasso, m = fitLasso(X, y, alpha);
  closeAll(m.coef, coef, 1e-6, "coef"); close(m.intercept, intercept, 1e-6, "intercept");
  assert.ok(coef.some((c) => c === 0) && m.coef.some((c) => c === 0));
});
test("gradient descent reproduces the reference trajectory and flags divergence", () => {
  const { x, y, ok, diverge } = F.gd, t = gradientDescent(x, y, ok.lr, ok.steps);
  t.forEach((s, i) => { close(s.w, ok.traj[i][0], 1e-9, "w"); close(s.b, ok.traj[i][1], 1e-9, "b"); close(s.loss, ok.traj[i][2], 1e-9, "loss"); });
  assert.equal(diverged(t), false);
  const d = gradientDescent(x, y, diverge.lr, diverge.steps);
  close(d[d.length - 1].loss, diverge.traj[diverge.traj.length - 1][2], 1e-9, "diverged loss");
  assert.equal(diverged(d), true);
});
test("standardise matches StandardScaler", () => {
  const s = standardise(F.scale.X);
  closeAll(s.mean, F.scale.mean, 1e-12, "mean"); closeAll(s.std, F.scale.std, 1e-12, "std");
  s.transform(F.scale.X).forEach((r, i) => closeAll(r, F.scale.z[i], 1e-12, "z"));
});
test("kFold without shuffle matches KFold, and every row is tested exactly once", () => {
  const folds = kFold(F.kfold.n, F.kfold.k);
  folds.forEach((f, i) => assert.deepEqual(f.test, F.kfold.tests[i]));
  const seen = kFold(50, 7, 3).flatMap((f) => f.test).sort((a, b) => a - b);
  assert.deepEqual(seen, Array.from({ length: 50 }, (_, i) => i));
  kFold(50, 7, 3).forEach((f) => assert.equal(f.train.filter((i) => f.test.includes(i)).length, 0));
});
test("train/test split partitions the data and is reproducible", () => {
  const a = trainTestSplit(100, 0.25, 5), b = trainTestSplit(100, 0.25, 5);
  assert.deepEqual(a, b); assert.equal(a.test.length, 25);
  assert.deepEqual([...a.train, ...a.test].sort((p, q) => p - q), Array.from({ length: 100 }, (_, i) => i));
});
test("time-ordered splits never train on the future", () => {
  const s = timeSplit(100, 0.7); assert.ok(Math.max(...s.train) < Math.min(...s.test));
  for (const w of walkForward(100, 40, 10)) assert.ok(Math.max(...w.train) < Math.min(...w.test));
});
test("metrics match scikit-learn", () => {
  const { y, score, auc, precision, recall, f1, confusion: cm } = F.metrics, pred = score.map((s) => (s >= 0.5 ? 1 : 0));
  close(rocAuc(y, score), auc, 1e-12, "auc");
  const p = precisionRecallF1(y, pred); close(p.precision, precision, 1e-12); close(p.recall, recall, 1e-12); close(p.f1, f1, 1e-12);
  const c = confusion(y, pred); assert.deepEqual([[c.tn, c.fp], [c.fn, c.tp]], cm);
  const roc = rocCurve(y, score); close(roc[roc.length - 1].tpr, 1, 1e-12); close(roc[roc.length - 1].fpr, 1, 1e-12);
});
test("logistic regression matches scikit-learn for two regularisation strengths", () => {
  for (const f of F.logistic.fits) {
    const m = fitLogistic(F.logistic.X, F.logistic.y, f.C);
    closeAll(m.coef, f.coef, 1e-5, `coef C=${f.C}`); close(m.intercept, f.intercept, 1e-5, "intercept");
    closeAll(m.proba(F.logistic.X), f.proba, 1e-5, "proba");
  }
});
test("overfitting lesson data: degree 9 fits the training points better than degree 3 but generalises worse", () => {
  const r = rng(7), f = (v) => Math.sin(2 * Math.PI * v) * 0.9, normal = () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  const x = Array.from({ length: 14 }, () => 0.04 + r() * 0.92).sort((a, b) => a - b), y = x.map((v) => f(v) + normal() * 0.22);
  const xt = Array.from({ length: 200 }, (_, i) => 0.05 + (i / 199) * 0.9), yt = xt.map(f);
  const a = fitPolynomial(x, y, 3), b = fitPolynomial(x, y, 9);
  assert.ok(mse(b.predict(x), y) < mse(a.predict(x), y), "train error should fall");
  assert.ok(mse(b.predict(xt), yt) > 5 * mse(a.predict(xt), yt), "test error should rise");
});

import { gradientDescentMulti, lambdaMax } from "../lib/ml/gradient.ts";
test("multi-feature descent and the stability limit match numpy", () => {
  const m = F.multi;
  close(lambdaMax(m.X), m.lmax, 1e-9, "lambdaMax");
  gradientDescentMulti(m.X, m.y, m.lr, 40).losses.forEach((l, i) => close(l, m.losses[i], 1e-9, `loss ${i}`));
  const ok = gradientDescentMulti(m.X, m.y, m.ok_lr, 40).losses, bad = gradientDescentMulti(m.X, m.y, m.div_lr, 40).losses;
  close(ok[39], m.ok_losses[39], 1e-8, "just under the limit");
  assert.ok(bad[bad.length - 1] > bad[0] * 10, "above 2/lambdaMax the loss must grow");
  assert.ok(ok[39] < ok[0], "below 2/lambdaMax the loss must fall");
});

import { biasVariance } from "../lib/ml/biasvar.ts";
test("bias falls and variance rises as the model gets more flexible", () => {
  const d1 = biasVariance(1, { resamples: 150 }), d3 = biasVariance(3, { resamples: 150 }), d8 = biasVariance(8, { resamples: 150 });
  assert.ok(d1.bias2 > 3 * d3.bias2, "a straight line should be far more biased than a cubic");
  assert.ok(d8.variance > 5 * d3.variance, "degree 8 should have much higher variance than degree 3");
  assert.ok(d3.total < d1.total && d3.total < d8.total, "the cubic should have the lowest total error of the three");
  close(d3.total, d3.bias2 + d3.variance + d3.noise2, 1e-12, "decomposition adds up");
});

test("lambdaMax is the true stability limit, on raw and on standardised features", () => {
  const r = rng(21), raw = Array.from({ length: 30 }, () => [r(), r() * 300]), y = raw.map((v) => 2 * v[0] + 0.01 * v[1] + 1 + (r() - 0.5) * 0.2);
  const mu = [0, 1].map((j) => raw.reduce((a, v) => a + v[j], 0) / 30), sd = [0, 1].map((j) => Math.sqrt(raw.reduce((a, v) => a + (v[j] - mu[j]) ** 2, 0) / 30));
  const std = raw.map((v) => v.map((t, j) => (t - mu[j]) / sd[j]));
  for (const X of [raw, std]) {
    const lim = 2 / lambdaMax(X), ok = gradientDescentMulti(X, y, 0.98 * lim, 400).losses, bad = gradientDescentMulti(X, y, 1.02 * lim, 400).losses;
    assert.ok(ok[399] < ok[0], "just under the limit the loss falls");
    assert.ok(bad[bad.length - 1] > bad[0] * 10, "just over the limit the loss grows");
  }
});

import { brier, reliability, averagePrecision, ece } from "../lib/ml/calibration.ts";
test("brier, reliability bins and average precision match scikit-learn", () => {
  const c = F.calib;
  close(brier(c.y, c.score), c.brier, 1e-12, "brier");
  close(averagePrecision(c.y, c.score), c.ap, 1e-12, "average precision");
  const r = reliability(c.y, c.score, 10);
  assert.equal(r.length, c.frac.length);
  r.forEach((b, i) => { close(b.frac, c.frac[i], 1e-12, `frac ${i}`); close(b.mean, c.mean[i], 1e-12, `mean ${i}`); });
  assert.ok(ece(c.y, c.score) >= 0);
});

import { knnProba, fitGaussianNB, fitTree, treeProba, fitBoost, fitSVM, fitForest, makeData } from "../lib/ml/index.ts";
test("kNN, naive Bayes, tree and boosting match scikit-learn on a grid", () => {
  const c = F.cls, agree = (got, want, tol) => got.filter((v, i) => Math.abs(v - want[i]) <= tol).length / got.length;
  assert.equal(agree(c.G.map((q) => knnProba(c.X, c.y, q, 5)), c.knn, 1e-12), 1, "knn");
  const nb = fitGaussianNB(c.X, c.y); assert.ok(agree(c.G.map((q) => nb.proba(q)), c.nb, 1e-6) === 1, "naive bayes");
  const t = fitTree(c.X, c.y, { maxDepth: 3 }); assert.ok(agree(c.G.map((q) => treeProba(t, q)), c.tree, 1e-9) >= 0.97, "tree");
  const gb = fitBoost(c.X, c.y, { rounds: 15, lr: 0.3, depth: 2 }); assert.ok(agree(c.G.map((q) => gb.proba(q)), c.gb, 1e-6) >= 0.97, "boosting");
});
test("SVM and forest behave: SMO agrees with libsvm on sign, forest beats chance", () => {
  const c = F.cls, svm = fitSVM(c.X, c.y, { C: 1, gamma: 3 }), same = c.G.filter((q, i) => (svm.decision(q) > 0) === (c.svm_rbf[i] > 0)).length / c.G.length;
  assert.ok(same >= 0.95, `svm sign agreement ${same}`);
  assert.ok(Math.abs(svm.sv.length - c.svm_n) <= Math.max(6, 0.3 * c.svm_n), `support vectors ${svm.sv.length} vs ${c.svm_n}`);
  const d = makeData("moons", 160, 0.1), f = fitForest(d.X, d.y, { trees: 25 }), acc = d.X.filter((q, i) => (f.proba(q) >= 0.5 ? 1 : 0) === d.y[i]).length / 160;
  assert.ok(acc > 0.9, `forest training accuracy ${acc}`);
});

// ---- Part 4: find structure ----
const U = F.unsup;
const canon = (l) => { const m = new Map(); return l.map((v) => (m.has(v) ? m.get(v) : (m.set(v, m.size), m.size - 1))); };
test("kmeans from the same start matches scikit-learn", () => {
  const m = kmeans(U.X, 3, { init: U.init });
  assert.ok(Math.abs(m.inertia - U.km_inertia) < 1e-9);
  assert.deepEqual(m.labels, U.km_labels);
});
test("silhouette matches scikit-learn", () => assert.ok(Math.abs(silhouette(U.X, U.km_labels) - U.sil) < 1e-9));
test("dbscan labels match scikit-learn", () => assert.deepEqual(dbscan(U.moons, 0.07, 4), U.db));
test("agglomerative partitions match scikit-learn for every linkage", () => {
  for (const l of ["single", "complete", "average", "ward"]) assert.deepEqual(canon(agglomerative(U.X, 3, l)), canon(U.agg[l]), l);
});
test("PCA variances and axes match scikit-learn", () => {
  const p = fitPCA(U.pca_X);
  p.values.forEach((v, i) => assert.ok(Math.abs(v - U.pca_values[i]) < 1e-8));
  p.components.forEach((c, i) => c.forEach((v, j) => assert.ok(Math.abs(Math.abs(v) - U.pca_abs[i][j]) < 1e-6)));
});
test("Gaussian mixture EM matches scikit-learn (likelihood, BIC, means)", () => {
  for (const cov of ["full", "diag", "spherical"]) {
    const g = U.gmm[cov], m = fitGMM(U.X, 3, { cov, tol: 1e-9, maxIter: 500, init: { weights: [1 / 3, 1 / 3, 1 / 3], means: U.init, covs: g.cov0 } });
    assert.ok(Math.abs(m.loglik - g.loglik) < 1e-4 * Math.abs(g.loglik), cov + " loglik");
    assert.ok(Math.abs(m.bic - g.bic) < 1e-3 * Math.abs(g.bic), cov + " bic");
    const sorted = [...m.means].sort((a, b) => a[0] - b[0]), ref = [...g.means].sort((a, b) => a[0] - b[0]);
    sorted.forEach((c, i) => c.forEach((v, j) => assert.ok(Math.abs(v - ref[i][j]) < 1e-4)));
  }
});
test("t-SNE affinities match scikit-learn", () => {
  const P = tsneAffinities(U.tsne_X, 12);
  let worst = 0; P.forEach((r, i) => r.forEach((v, j) => (worst = Math.max(worst, Math.abs(v - U.tsne_P[i][j])))));
  assert.ok(worst < 1e-5, "worst " + worst);
});
test("t-SNE keeps true clusters together in 2D", () => {
  const { X, truth } = makeHighD(90, 8, [0.4, 0.4, 0.4]), Y = tsne(X, { perplexity: 15, iters: 250 });
  let same = 0, tot = 0;
  Y.forEach((p, i) => { const nn = Y.map((q, j) => [(p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2, j]).filter(([, j]) => j !== i).sort((a, b) => a[0] - b[0]).slice(0, 8); nn.forEach(([, j]) => { tot++; if (truth[j] === truth[i]) same++; }); });
  assert.ok(same / tot > 0.95, "purity " + same / tot);
});
test("isolation forest ranks planted outliers highest", () => {
  const r = rng(9), X = Array.from({ length: 200 }, () => [r(), r()]), out = [[3, 3], [-2, 2], [2.5, -1]], f = fitIsolation([...X, ...out], { seed: 2 });
  const s = out.map((q) => f.score(q)), inl = X.map((q) => f.score(q)).sort((a, b) => b - a)[10];
  assert.ok(s.every((v) => v > inl));
});
test("adjusted Rand index: identical groupings score 1, relabelling does not matter", () => {
  assert.equal(adjustedRand([0, 0, 1, 1, 2, 2], [5, 5, 3, 3, 9, 9]), 1);
  assert.ok(Math.abs(adjustedRand([0, 0, 1, 1], [0, 1, 0, 1])) < 0.6);
});

// ---- Part 5: neural networks ----
const NF = F.nnfx;
test("perceptron weights match scikit-learn after 10 epochs", () => {
  const p = fitPerceptron(NF.perc.X, NF.perc.y, 10);
  p.w.forEach((v, i) => assert.ok(Math.abs(v - NF.perc.w[i]) < 1e-9)); assert.ok(Math.abs(p.b - NF.perc.b) < 1e-9);
});
test("perceptron converges on separable data and cycles on XOR", () => {
  const sep = fitPerceptron([[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8]], [0, 0, 1, 1], 20), xor = fitPerceptron([[0, 0], [1, 1], [0, 1], [1, 0]], [0, 0, 1, 1], 50);
  assert.equal(sep.mistakes.at(-1), 0); assert.ok(xor.mistakes.at(-1) > 0);
});
test("backprop loss and gradients match an independent numpy network (tanh and relu)", () => {
  for (const act of ["tanh", "relu"]) {
    const f = NF.mlp[act], n = makeNet([2, 5, 4, 1], { act }); f.W.forEach((w, l) => n.W[l].set(w)); f.b.forEach((b, l) => n.b[l].set(b));
    const g = lossGrad(n, NF.nx, NF.ny, { l2: 0.01 });
    assert.ok(Math.abs(g.loss - f.loss) < 1e-9, act + " loss");
    f.gW.forEach((gw, l) => gw.forEach((v, k) => assert.ok(Math.abs(g.W[l][k] - v) < 1e-9, act + " gW")));
    f.gb.forEach((gb, l) => gb.forEach((v, k) => assert.ok(Math.abs(g.b[l][k] - v) < 1e-9, act + " gb")));
  }
});
test("finite-difference gradient check passes, and flags a planted bug", () => {
  const n = makeNet([2, 6, 6, 1], { act: "tanh", seed: 3 });
  assert.ok(gradCheck(n, NF.nx, NF.ny, { l2: 0.01, count: 8 }) < 1e-6);
  const bad = gradCheck(n, NF.nx, NF.ny, { bug: "no-act-deriv", count: 8 }); assert.ok(bad > 1e-2, "bug error " + bad);
  const r = makeNet([2, 6, 6, 6, 1], { act: "tanh", seed: 4, residual: true }); assert.ok(gradCheck(r, NF.nx, NF.ny, { count: 8 }) < 1e-6, "residual");
});
test("optimisers follow the same updates as the reference (SGD, momentum, RMSprop, Adam)", () => {
  for (const [m, o] of Object.entries(NF.opt)) {
    const p = [Float64Array.from(NF.p0)], opt = new Opt(m, o.lr);
    for (let i = 0; i < 25; i++) opt.step(p, [Float64Array.from(p[0].map((v, j) => 2 * NF.cvec[j] * v))]);
    p[0].forEach((v, j) => assert.ok(Math.abs(v - o.final[j]) < 1e-9, m));
  }
});
test("convolution matches scipy with padding and stride, and the output-size formula", () => {
  const out = conv2d(NF.conv.img, NF.conv.k, 2, 1); assert.equal(out.length, NF.conv.out.length);
  out.forEach((r, i) => r.forEach((v, j) => assert.ok(Math.abs(v - NF.conv.out[i][j]) < 1e-9)));
  assert.equal(convOutSize(9, 3, 1, 2), NF.conv.out.length);
});
test("a hidden layer learns XOR, a straight line cannot", () => {
  const X = [], y = []; for (let i = 0; i < 80; i++) { const a = i % 2, b = (i >> 1) % 2; X.push([a + (rng(i)() - 0.5) * 0.2, b + (rng(i + 99)() - 0.5) * 0.2]); y.push(a ^ b); }
  const h = makeNet([2, 8, 1], { act: "tanh", seed: 2 }); trainNet(h, X, y, { epochs: 300, lr: 0.05 });
  const l = makeNet([2, 1], { act: "tanh", seed: 2 }); trainNet(l, X, y, { epochs: 300, lr: 0.05 });
  assert.ok(netAccuracy(h, X, y) > 0.95); assert.ok(netAccuracy(l, X, y) < 0.8);
});
test("sigmoid gradients vanish with depth, ReLU with He init and residuals keep them alive", () => {
  const { X, y } = makeData("moons", 60), ratio = (n) => { const g = layerGradNorms(n, X, y); return g[0] / g[g.length - 1]; };
  const sig = ratio(makeNet([2, ...Array(10).fill(8), 1], { act: "sigmoid" })), rel = ratio(makeNet([2, ...Array(10).fill(8), 1], { act: "relu" })), res = ratio(makeNet([2, ...Array(10).fill(8), 1], { act: "sigmoid", residual: true }));
  assert.ok(sig < 1e-3, "sigmoid " + sig); assert.ok(rel > sig * 100, "relu " + rel); assert.ok(res > sig * 100, "residual " + res);
});
test("inverted dropout keeps the average hidden activation, and the dead-unit counter works", () => {
  const net = makeNet([2, 6, 1], { act: "tanh", seed: 8 }), x = [0.3, 0.8], base = forward(net, x).A[1], r = rng(5), mean = new Float64Array(6);
  for (let i = 0; i < 6000; i++) forward(net, x, 0.4, r).A[1].forEach((v, j) => (mean[j] += v / 6000));
  base.forEach((v, j) => assert.ok(Math.abs(mean[j] - v) < 0.05, "unit " + j));
  const d = makeNet([2, 4, 1], { act: "relu" }); d.b[0].fill(-100); assert.equal(deadUnits(d, [[0.5, 0.5], [0.2, 0.9]]), 1);
});
