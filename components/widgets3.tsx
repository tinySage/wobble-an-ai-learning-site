"use client";
import { useMemo, useState } from "react";
import { averagePrecision, brier, confusion, ece, fitLogistic, fitPolynomial, kFold, mse, precisionRecallF1, reliability, rng, rocAuc, rocCurve, trainTestSplit, truth } from "@/lib/ml";
import { Axes, Btn, C, H, PAD, Row, Slider, Stat, W, sx, sy } from "./widgets";

const box = { width: "100%", height: "auto", background: "var(--screen)", borderRadius: 0, border: "3px solid var(--ink)" } as const;
const col = { width: "100%", display: "flex", flexDirection: "column", gap: 14 } as const;
const gauss = (r: () => number) => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const f3 = (v: number) => v.toFixed(3);

const DATA = (() => { const r = rng(17), x = Array.from({ length: 40 }, () => 0.03 + r() * 0.94), y = x.map((v) => truth(v) + gauss(r) * 0.22); return { x, y }; })();
const fitScore = (tr: number[], te: number[], deg = 5) => {
  const m = fitPolynomial(tr.map((i) => DATA.x[i]), tr.map((i) => DATA.y[i]), deg);
  return { train: mse(m.predict(tr.map((i) => DATA.x[i])), tr.map((i) => DATA.y[i])), test: mse(m.predict(te.map((i) => DATA.x[i])), te.map((i) => DATA.y[i])), m };
};
const Dots = ({ vals, mark, lo, hi, label }: { vals: number[]; mark?: number; lo: number; hi: number; label: string }) => (
  <svg viewBox={`0 0 ${W} 90`} role="img" aria-label={label} style={box}>
    <line x1={PAD} x2={W - PAD} y1={50} y2={50} stroke="rgba(38,48,31,.2)" />
    {vals.map((v, i) => <circle key={i} cx={PAD + ((v - lo) / (hi - lo)) * (W - 2 * PAD)} cy={50 + ((i * 7) % 5 - 2) * 5} r={5} fill={C.matcha} opacity={0.55} />)}
    {mark !== undefined && <circle cx={PAD + ((mark - lo) / (hi - lo)) * (W - 2 * PAD)} cy={50} r={8} fill="none" stroke={C.rose} strokeWidth={3} />}
    <text x={PAD} y={16} fontSize={11} fill={C.mute}>{label}</text><text x={PAD} y={84} fontSize={11} fill={C.mute}>{lo.toFixed(2)}</text><text x={W - PAD} y={84} fontSize={11} textAnchor="end" fill={C.mute}>{hi.toFixed(2)}</text>
  </svg>);

/* ---- train and test ---- */
export function TrainTest() {
  const [frac, setFrac] = useState(0.3), [seed, setSeed] = useState(1), [peek, setPeek] = useState(false);
  const sp = trainTestSplit(40, frac, seed), cur = fitScore(sp.train, sp.test);
  const spread = useMemo(() => Array.from({ length: 40 }, (_, s) => { const q = trainTestSplit(40, frac, s + 100); return fitScore(q.train, q.test).test; }), [frac]);
  const hi = Math.max(...spread) * 1.05, grid = Array.from({ length: 100 }, (_, i) => 0.02 + (i / 99) * 0.96), curve = cur.m.predict(grid);
  return (<div style={col}>
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Data split into training and test points with the fitted curve" style={box}><Axes />
      <polyline fill="none" stroke={C.matcha} strokeWidth={3.5} points={curve.map((v, i) => `${sx(grid[i])},${sy(Math.max(-1.6, Math.min(1.6, v)))}`).join(" ")} />
      {sp.train.map((i) => <circle key={i} cx={sx(DATA.x[i])} cy={sy(DATA.y[i])} r={5.5} fill={C.ink} />)}
      {sp.test.map((i) => <circle key={i} cx={sx(DATA.x[i])} cy={sy(DATA.y[i])} r={6} fill="none" stroke={C.rose} strokeWidth={2.2} />)}
      <text x={PAD + 6} y={PAD + 12} fontSize={11} fill={C.mute}>dark = used to fit · rose rings = held back, never seen by the fit</text></svg>
    <Row><Stat k="train points" v={String(sp.train.length)} /><Stat k="test points" v={String(sp.test.length)} /><Stat k={peek ? "score reported (on training data)" : "score reported (on test data)"} v={f3(peek ? cur.train : cur.test)} tone={peek ? "bad" : undefined} /><Stat k="honest test error" v={f3(cur.test)} /></Row>
    <Dots vals={spread} mark={cur.test} lo={Math.min(...spread) * 0.9} hi={hi} label="test error from 40 different random splits at this split size (rose = the one shown)" />
    <Row><Slider label="Share held back for testing" value={frac} min={0.1} max={0.5} step={0.05} onChange={setFrac} show={`${Math.round(frac * 100)}%`} />
      <Btn kind="primary" onClick={() => setSeed(seed + 1)}>Re-split</Btn><Btn kind="break" onClick={() => setPeek(!peek)}>{peek ? "Score on test data again" : "Break it: score on the training data"}</Btn></Row>
    <p className="mute" style={{ margin: 0, fontSize: 13 }} aria-live="polite">{peek ? "The training score looks wonderful because the model has already seen those points." : "Each re-split changes the answer. One split is one opinion."}</p>
  </div>);
}

/* ---- cross-validation ---- */
export function CrossVal() {
  const [k, setK] = useState(5), [lucky, setLucky] = useState(false);
  const folds = useMemo(() => kFold(40, k, 3).map((f) => fitScore(f.train, f.test).test), [k]);
  const mean = folds.reduce((a, b) => a + b, 0) / k, sd = Math.sqrt(folds.reduce((a, b) => a + (b - mean) ** 2, 0) / k);
  const singles = useMemo(() => Array.from({ length: 40 }, (_, s) => { const q = trainTestSplit(40, 0.25, s + 200); return fitScore(q.train, q.test).test; }), []);
  const best = Math.min(...singles), hi = Math.max(...folds, ...singles, best) * 1.05, bw = (W - 2 * PAD) / k;
  return (<div style={col}>
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Test error in each cross-validation fold" style={box}><Axes />
      {folds.map((e, i) => { const h = (e / hi) * (H - 2 * PAD - 20); return <rect key={i} x={PAD + i * bw + 6} width={bw - 12} y={H - PAD - h} height={h} fill={C.matcha} opacity={0.8} />; })}
      <line x1={PAD} x2={W - PAD} y1={H - PAD - (mean / hi) * (H - 2 * PAD - 20)} y2={H - PAD - (mean / hi) * (H - 2 * PAD - 20)} stroke={C.ink} strokeDasharray="5 5" />
      {lucky && <line x1={PAD} x2={W - PAD} y1={H - PAD - (best / hi) * (H - 2 * PAD - 20)} y2={H - PAD - (best / hi) * (H - 2 * PAD - 20)} stroke={C.rose} strokeWidth={3} />}
      <text x={PAD + 6} y={PAD + 12} fontSize={11} fill={C.mute}>each bar = error on one held-out fold · dashed = average{lucky ? " · rose = the luckiest single split of 40 tried" : ""}</text></svg>
    <Row><Stat k="folds" v={String(k)} /><Stat k="average error" v={f3(mean)} /><Stat k="spread between folds" v={`± ${f3(sd)}`} />{lucky && <Stat k="luckiest single split" v={f3(best)} tone="bad" />}</Row>
    <Row><Slider label="Folds (k)" value={k} min={2} max={10} step={1} onChange={setK} show={String(k)} /><Btn kind="break" onClick={() => setLucky(!lucky)}>{lucky ? "Hide the lucky split" : "Break it: report the best-looking split"}</Btn></Row>
    <p className="mute" style={{ margin: 0, fontSize: 13 }} aria-live="polite">{lucky ? `Picking the best of 40 splits reports ${f3(best)} while the honest average is ${f3(mean)}.` : "Every point gets used for testing exactly once. The spread tells you how much to trust the average."}</p>
  </div>);
}

/* ---- data leakage ---- */
function leakRun(seed: number) {
  const r = rng(seed * 31 + 1), n = 60, p = 200, X = Array.from({ length: n }, () => Array.from({ length: p }, () => gauss(r))), y = Array.from({ length: n }, () => (r() < 0.5 ? 1 : 0));
  const top = (rows: number[], k = 10) => { const my = rows.reduce((a, i) => a + y[i], 0) / rows.length, sc = Array.from({ length: p }, (_, j) => { const mx = rows.reduce((a, i) => a + X[i][j], 0) / rows.length; let c = 0; rows.forEach((i) => { c += (X[i][j] - mx) * (y[i] - my); }); return Math.abs(c); }); return sc.map((v, j) => [v, j]).sort((a, b) => b[0] - a[0]).slice(0, k).map((t) => t[1]); };
  const all = Array.from({ length: n }, (_, i) => i), leakyCols = top(all), run = (leaky: boolean) => {
    let hit = 0; for (const f of kFold(n, 5, seed)) {
      const cols = leaky ? leakyCols : top(f.train), m = fitLogistic(f.train.map((i) => cols.map((j) => X[i][j])), f.train.map((i) => y[i]), 1);
      const pr = m.proba(f.test.map((i) => cols.map((j) => X[i][j]))); f.test.forEach((i, t) => { if ((pr[t] >= 0.5 ? 1 : 0) === y[i]) hit++; });
    } return hit / n; };
  return { honest: run(false), leaky: run(true) };
}
export function Leakage() {
  const [show, setShow] = useState(false);
  const runs = useMemo(() => Array.from({ length: 12 }, (_, s) => leakRun(s + 1)), []);
  const avg = (k: "honest" | "leaky") => runs.reduce((a, b) => a + b[k], 0) / runs.length;
  const row = (k: "honest" | "leaky", y: number, colr: string, label: string) => <g>
    <text x={PAD} y={y - 22} fontSize={12} fill={C.ink}>{label}</text><line x1={PAD} x2={W - PAD} y1={y} y2={y} stroke="rgba(38,48,31,.2)" />
    {runs.map((r, i) => <circle key={i} cx={PAD + ((r[k] - 0.3) / 0.7) * (W - 2 * PAD)} cy={y + ((i * 5) % 4 - 1.5) * 6} r={6} fill={colr} opacity={0.6} />)}
    <circle cx={PAD + ((avg(k) - 0.3) / 0.7) * (W - 2 * PAD)} cy={y} r={9} fill="none" stroke={C.ink} strokeWidth={2.5} /></g>;
  return (<div style={col}>
    <svg viewBox={`0 0 ${W} 230`} role="img" aria-label="Cross-validated accuracy on pure noise, with and without leakage" style={box}>
      <line x1={PAD + (0.2 / 0.7) * (W - 2 * PAD)} x2={PAD + (0.2 / 0.7) * (W - 2 * PAD)} y1={20} y2={200} stroke={C.mute} strokeDasharray="4 5" /><text x={PAD + (0.2 / 0.7) * (W - 2 * PAD) + 6} y={32} fontSize={11} fill={C.mute}>50% = guessing</text>
      {row("honest", 100, C.matcha, "Features chosen inside each fold")}{show && row("leaky", 175, C.rose, "Features chosen once, using every row first")}
      {[0.3, 0.5, 0.7, 1.0].map((v) => <text key={v} x={PAD + ((v - 0.3) / 0.7) * (W - 2 * PAD)} y={222} fontSize={11} textAnchor="middle" fill={C.mute}>{Math.round(v * 100)}%</text>)}</svg>
    <Row><Stat k="honest average" v={pct(avg("honest"))} />{show && <Stat k="leaky average" v={pct(avg("leaky"))} tone="bad" />}<Stat k="draws" v="12" /></Row>
    <Row><Btn kind="break" onClick={() => setShow(!show)}>{show ? "Hide the leak" : "Break it: choose features before splitting"}</Btn><span className="mute" style={{ fontSize: 13 }}>The labels are random coin flips, so no model can honestly beat 50%.</span></Row>
  </div>);
}

/* ---- metrics, imbalance ---- */
function scored(n: number, prev: number, seed = 5) {
  const r = rng(seed), np = Math.max(2, Math.round(n * prev)), y = Array.from({ length: n }, (_, i) => (i < np ? 1 : 0));
  return { y, s: y.map((t) => Math.max(0, Math.min(1, (t ? 0.62 : 0.4) + gauss(r) * 0.2))) };
}
const Conf = ({ c }: { c: { tn: number; fp: number; fn: number; tp: number } }) => (
  <div role="table" aria-label="Confusion matrix" style={{ display: "grid", gridTemplateColumns: "auto 1fr 1fr", gap: 4, fontSize: 13, minWidth: 260 }}>
    <span /><b className="mute" style={{ textAlign: "center" }}>said no</b><b className="mute" style={{ textAlign: "center" }}>said yes</b>
    <b className="mute">truly no</b><Stat k="true negative" v={String(c.tn)} /><Stat k="false alarm" v={String(c.fp)} tone={c.fp > c.tp ? "bad" : undefined} />
    <b className="mute">truly yes</b><Stat k="missed" v={String(c.fn)} tone={c.fn > c.tp ? "bad" : undefined} /><Stat k="true positive" v={String(c.tp)} /></div>);
export function Metrics({ imbalance }: { imbalance?: boolean }) {
  const [th, setTh] = useState(0.5), [prev, setPrev] = useState(imbalance ? 0.2 : 0.5), [base, setBase] = useState(false);
  const d = useMemo(() => scored(imbalance ? 4000 : 400, prev), [prev, imbalance]);
  const pred = d.s.map((v) => (v >= th ? 1 : 0)), c = confusion(d.y, base ? d.y.map(() => 0) : pred), prf = precisionRecallF1(d.y, base ? d.y.map(() => 0) : pred);
  const acc = (c.tp + c.tn) / d.y.length, roc = useMemo(() => rocCurve(d.y, d.s), [d]), auc = rocAuc(d.y, d.s), ap = averagePrecision(d.y, d.s);
  const pt = roc.reduce((b, p) => (Math.abs(p.threshold - th) < Math.abs(b.threshold - th) || b.threshold === Infinity ? p : b), roc[0]);
  const rp = roc.slice(1).filter((p) => p.threshold >= th).pop() ?? roc[0];
  return (<div style={col}>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(280px,1fr))", gap: 12 }}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Score distributions with the decision threshold" style={box}><Axes />
        {[0, 1].map((cls) => { const bins = Array.from({ length: 20 }, (_, b) => d.y.filter((t, i) => t === cls && d.s[i] >= b / 20 && d.s[i] < (b + 1) / 20 + (b === 19 ? 1 : 0)).length), top = Math.max(1, ...Array.from({ length: 20 }, (_, b) => d.y.filter((t, i) => d.s[i] >= b / 20 && d.s[i] < (b + 1) / 20).length));
          return bins.map((n, b) => <rect key={`${cls}${b}`} x={PAD + (b / 20) * (W - 2 * PAD) + (cls ? 14 : 2)} width={(W - 2 * PAD) / 20 / 2 - 4} y={H - PAD - (n / top) * (H - 2 * PAD - 20)} height={(n / top) * (H - 2 * PAD - 20)} fill={cls ? C.matcha : C.rose} opacity={0.75} />); })}
        <line x1={PAD + th * (W - 2 * PAD)} x2={PAD + th * (W - 2 * PAD)} y1={PAD} y2={H - PAD} stroke={C.ink} strokeWidth={3} />
        <text x={PAD + 6} y={PAD + 12} fontSize={11} fill={C.mute}>model score: green = truly yes, rose = truly no · line = threshold</text></svg>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="ROC curve" style={box}><Axes />
        <line x1={sx(0)} y1={sy(0, 0, 1)} x2={sx(1)} y2={sy(1, 0, 1)} stroke={C.mute} strokeDasharray="5 5" />
        <polyline fill="none" stroke={C.matcha} strokeWidth={3.5} points={roc.map((p) => `${sx(p.fpr)},${sy(p.tpr, 0, 1)}`).join(" ")} />
        <circle cx={sx(rp.fpr)} cy={sy(rp.tpr, 0, 1)} r={8} fill={C.rose} /><text x={PAD + 6} y={PAD + 12} fontSize={11} fill={C.mute}>ROC: x = false alarm rate, y = share found · dashed = guessing</text></svg>
    </div>
    <Row><Conf c={c} /><div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}><Stat k="accuracy" v={pct(acc)} /><Stat k="precision" v={pct(prf.precision)} /><Stat k="recall" v={pct(prf.recall)} tone={imbalance && prf.recall < 0.1 ? "bad" : undefined} /><Stat k="F1" v={f3(prf.f1)} /><Stat k="ROC AUC" v={f3(auc)} />{imbalance && <Stat k="PR AUC (avg precision)" v={f3(ap)} />}</div></Row>
    <Row><Slider label="Threshold" value={th} min={0.05} max={0.95} step={0.01} onChange={setTh} show={th.toFixed(2)} />
      {imbalance && <Slider label="Share of truly-yes cases" value={prev} min={0.005} max={0.5} step={0.005} onChange={setPrev} show={pct(prev)} />}</Row>
    <Row>{imbalance ? <><Btn kind="break" onClick={() => { setPrev(0.01); setBase(true); }}>Break it: 1% rare, model always says no</Btn><Btn onClick={() => setBase(false)}>Use the real model</Btn></> : <Btn kind="break" onClick={() => setTh(0.05)}>Break it: flag everything</Btn>}</Row>
    {imbalance && base && <p style={{ margin: 0, fontWeight: 700, color: "var(--rose)" }} aria-live="polite">Accuracy {pct(acc)} from a model that never finds a single case.</p>}
  </div>);
}

/* ---- calibration ---- */
const CAL = (() => { const r = rng(9), n = 3000, p = Array.from({ length: n }, () => Math.min(0.98, Math.max(0.02, 0.5 + gauss(r) * 0.28))), y = p.map((q) => (r() < q ? 1 : 0)); return { p, y }; })();
const logit = (p: number) => Math.log(p / (1 - p)), sig = (z: number) => 1 / (1 + Math.exp(-z));
export function Calibration() {
  const [t, setT] = useState(1), [fix, setFix] = useState(false);
  const raw = useMemo(() => CAL.p.map((q) => sig(t * logit(q))), [t]), half = 1500;
  const fixed = useMemo(() => { const m = fitLogistic(raw.slice(0, half).map((v) => [logit(Math.min(0.999, Math.max(0.001, v)))]), CAL.y.slice(0, half), 1e6); return raw.map((v) => m.proba([[logit(Math.min(0.999, Math.max(0.001, v)))]])[0]); }, [raw]);
  const p = (fix ? fixed : raw).slice(half), y = CAL.y.slice(half), rel = reliability(y, p, 10);
  return (<div style={col}>
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Reliability diagram: predicted probability against how often it came true" style={box}><Axes />
      <line x1={sx(0)} y1={sy(0, 0, 1)} x2={sx(1)} y2={sy(1, 0, 1)} stroke={C.mute} strokeDasharray="5 5" />
      <polyline fill="none" stroke={fix ? C.matcha : C.rose} strokeWidth={3} points={rel.map((b) => `${sx(b.mean)},${sy(b.frac, 0, 1)}`).join(" ")} />
      {rel.map((b, i) => <circle key={i} cx={sx(b.mean)} cy={sy(b.frac, 0, 1)} r={4 + 10 * Math.sqrt(b.n / y.length)} fill={fix ? C.matcha : C.rose} opacity={0.7} />)}
      <text x={PAD + 6} y={PAD + 12} fontSize={11} fill={C.mute}>x = what the model said · y = how often it was right · dashed = perfectly honest</text></svg>
    <Row><Stat k="Brier score" v={f3(brier(y, p))} /><Stat k="calibration gap (ECE)" v={f3(ece(y, p))} /><Stat k="ROC AUC" v={f3(rocAuc(y, p))} /><Stat k="accuracy at 0.5" v={pct(y.filter((v, i) => (p[i] >= 0.5 ? 1 : 0) === v).length / y.length)} /></Row>
    <Row><Slider label="Overconfidence" value={t} min={0.4} max={3} step={0.1} onChange={(v) => { setFix(false); setT(v); }} show={t === 1 ? "honest" : t > 1 ? `×${t.toFixed(1)} too sure` : `×${t.toFixed(1)} too shy`} />
      <Btn kind="break" onClick={() => { setFix(false); setT(3); }}>Break it: make it overconfident</Btn><Btn kind="primary" onClick={() => setFix(!fix)}>{fix ? "Show uncorrected" : "Recalibrate"}</Btn></Row>
    <p className="mute" style={{ margin: 0, fontSize: 13 }}>Measured on 1,500 held-out cases. Recalibration is fitted on a different 1,500. Ranking (AUC) does not change under a stretch of the probabilities. Accuracy at 0.5 stays put for the stretch, and can shift a little after recalibration.</p>
  </div>);
}

/* ---- hyperparameter search ---- */
const fA = (a: number) => Math.exp(-0.5 * ((a - 0.62) / 0.07) ** 2), fB = (b: number) => Math.exp(-0.5 * ((b - 0.4) / 0.25) ** 2);
export function HyperSearch() {
  const [n, setN] = useState(9), [seed, setSeed] = useState(1), [useless, setUseless] = useState(false);
  const w = useless ? 0 : 1, score = (a: number, b: number) => (fA(a) + w * fB(b)) / (1 + w);
  const side = Math.max(2, Math.round(Math.sqrt(n))), gridPts = Array.from({ length: side * side }, (_, i) => [(Math.floor(i / side) + 0.5) / side, ((i % side) + 0.5) / side]);
  const rand = (s: number, m: number) => { const r = rng(s * 977 + 3); return Array.from({ length: m }, () => [r(), r()]); };
  const rp = rand(seed, side * side), best = (pts: number[][]) => Math.max(...pts.map((q) => score(q[0], q[1])));
  const avg = useMemo(() => { let g = best(gridPts), r = 0; for (let s = 1; s <= 300; s++) r += best(rand(s + 1000, side * side)); return { grid: g, rand: r / 300 }; }, [n, useless]);
  const panel = (pts: number[][], colr: string, label: string) => <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} style={box}>
    <rect x={PAD} y={PAD} width={W - 2 * PAD} height={H - 2 * PAD} fill="none" stroke="rgba(38,48,31,.2)" />
    <rect x={PAD + 0.55 * (W - 2 * PAD)} y={PAD} width={0.14 * (W - 2 * PAD)} height={H - 2 * PAD} fill={C.soft} opacity={0.6} />
    {pts.map((q, i) => <circle key={i} cx={PAD + q[0] * (W - 2 * PAD)} cy={PAD + q[1] * (H - 2 * PAD)} r={6.5} fill={colr} opacity={0.85} />)}
    <text x={PAD + 6} y={PAD - 10} fontSize={11} fill={C.mute}>{label}</text><text x={W / 2} y={H - 8} fontSize={11} textAnchor="middle" fill={C.mute}>important setting →  (pink band = the good region)</text></svg>;
  return (<div style={col}>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(280px,1fr))", gap: 12 }}>{panel(gridPts, C.ink, `Grid: ${side * side} trials`)}{panel(rp, C.matcha, `Random: ${side * side} trials`)}</div>
    <Row><Stat k="grid: distinct values tried of the important setting" v={String(side)} /><Stat k="random: distinct values tried" v={String(side * side)} /><Stat k="best score, this draw (grid / random)" v={`${f3(best(gridPts))} / ${f3(best(rp))}`} /><Stat k="average best over 300 draws (grid / random)" v={`${f3(avg.grid)} / ${f3(avg.rand)}`} /></Row>
    <Row><Slider label="Trials" value={n} min={4} max={36} step={1} onChange={setN} show={String(side * side)} /><Btn kind="primary" onClick={() => setSeed(seed + 1)}>Draw new random trials</Btn>
      <Btn kind="break" onClick={() => setUseless(!useless)}>{useless ? "Let the second setting matter again" : "Break it: make the second setting useless"}</Btn></Row>
    <p className="mute" style={{ margin: 0, fontSize: 13 }}>Toy score surface. {useless ? "With one setting doing all the work, the grid repeats the same few values and random search tends to land closer." : "With both settings mattering, random still edges ahead here, by less than when one setting is useless. Grid search holds up best in one or two dimensions."}</p>
  </div>);
}

export function Widget3({ name }: { name: string }) {
  switch (name) {
    case "train-test": return <TrainTest />;
    case "cross-val": return <CrossVal />;
    case "leakage": return <Leakage />;
    case "metrics": return <Metrics />;
    case "imbalance": return <Metrics imbalance />;
    case "calibration": return <Calibration />;
    case "hyper-search": return <HyperSearch />;
    default: return null;
  }
}
export const PART3 = ["train-test", "cross-val", "leakage", "metrics", "imbalance", "calibration", "hyper-search"];
