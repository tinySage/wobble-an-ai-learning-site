export function confusion(y: number[], pred: number[]) {
  let tn = 0, fp = 0, fn = 0, tp = 0;
  y.forEach((t, i) => { if (t === 1) pred[i] === 1 ? tp++ : fn++; else pred[i] === 1 ? fp++ : tn++; });
  return { tn, fp, fn, tp };
}
export const accuracy = (y: number[], pred: number[]) => y.filter((t, i) => t === pred[i]).length / y.length;
export function precisionRecallF1(y: number[], pred: number[]) {
  const { fp, fn, tp } = confusion(y, pred);
  const precision = tp + fp ? tp / (tp + fp) : 0, recall = tp + fn ? tp / (tp + fn) : 0;
  return { precision, recall, f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0 };
}
/** Area under the ROC curve by the rank-sum formula, with tied scores sharing their average rank. */
export function rocAuc(y: number[], score: number[]): number {
  const idx = score.map((_, i) => i).sort((a, b) => score[a] - score[b]), rank = new Array(score.length).fill(0);
  for (let i = 0; i < idx.length;) { let j = i; while (j + 1 < idx.length && score[idx[j + 1]] === score[idx[i]]) j++; const r = (i + j) / 2 + 1; for (let k = i; k <= j; k++) rank[idx[k]] = r; i = j + 1; }
  const np = y.filter((t) => t === 1).length, nn = y.length - np;
  const sum = y.reduce((s, t, i) => s + (t === 1 ? rank[i] : 0), 0);
  return (sum - (np * (np + 1)) / 2) / (np * nn);
}
export function rocCurve(y: number[], score: number[]) {
  const th = [...new Set(score)].sort((a, b) => b - a), np = y.filter((t) => t === 1).length, nn = y.length - np;
  const pts = [{ fpr: 0, tpr: 0, threshold: Infinity }];
  for (const t of th) { let tp = 0, fp = 0; y.forEach((v, i) => { if (score[i] >= t) v === 1 ? tp++ : fp++; }); pts.push({ fpr: fp / nn, tpr: tp / np, threshold: t }); }
  return pts;
}
