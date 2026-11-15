import { fitPolynomial } from "./regression.ts";
import { rng } from "./split.ts";

export const truth = (x: number) => Math.sin(2 * Math.PI * x) * 0.9;
const gauss = (r: () => number) => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
export const GRID = Array.from({ length: 60 }, (_, i) => 0.05 + (i / 59) * 0.9);

export interface BV { degree: number; bias2: number; variance: number; noise2: number; total: number; fits: number[][]; meanFit: number[] }

/** Refit a polynomial on many fresh training sets and split the test error into bias squared, variance and noise. */
export function biasVariance(degree: number, opt: { nTrain?: number; resamples?: number; noise?: number; seed?: number; lambda?: number; keep?: number } = {}): BV {
  const { nTrain = 14, resamples = 60, noise = 0.22, seed = 1, lambda = 0, keep = 24 } = opt, r = rng(seed * 7919 + degree);
  const preds: number[][] = [];
  for (let s = 0; s < resamples; s++) {
    const x = Array.from({ length: nTrain }, (_, i) => 0.04 + 0.92 * ((i + r()) / nTrain)), y = x.map((v) => truth(v) + gauss(r) * noise);
    preds.push(fitPolynomial(x, y, degree, lambda).predict(GRID));
  }
  const meanFit = GRID.map((_, j) => preds.reduce((a, p) => a + p[j], 0) / resamples);
  const bias2 = GRID.reduce((a, g, j) => a + (meanFit[j] - truth(g)) ** 2, 0) / GRID.length;
  const variance = GRID.reduce((a, _, j) => a + preds.reduce((b, p) => b + (p[j] - meanFit[j]) ** 2, 0) / resamples, 0) / GRID.length;
  return { degree, bias2, variance, noise2: noise * noise, total: bias2 + variance + noise * noise, fits: preds.slice(0, keep), meanFit };
}
