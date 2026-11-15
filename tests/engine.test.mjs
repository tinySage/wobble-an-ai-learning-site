import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { fitLinear, polyFeatures, predict, fitLasso, fitPolynomial, mse } from "../lib/ml/regression.ts";
import { gradientDescent, diverged } from "../lib/ml/gradient.ts";
import { standardise } from "../lib/ml/scale.ts";
import { kFold, trainTestSplit, timeSplit, walkForward, rng } from "../lib/ml/split.ts";
import { rocAuc, precisionRecallF1, confusion, rocCurve } from "../lib/ml/metrics.ts";
import { fitLogistic } from "../lib/ml/logistic.ts";

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
