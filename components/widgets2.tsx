"use client";
import { useEffect, useMemo, useState } from "react";
import { fitBoost, fitForest, fitGaussianNB, fitLogistic, fitSVM, fitTree, knnProba, makeData, trainTestSplit, treeDepth, treeProba, type Kind } from "@/lib/ml";
import { Btn, C, Row, Slider, Stat, W } from "./widgets";

const R = [196, 65, 107], G = [88, 121, 47], S = 380, GW = 44, GH = 44;
const mix = (p: number) => { const t = Math.min(1, Math.max(0, p)), a = Math.abs(t - 0.5) * 2, c = t > 0.5 ? G : R; return `rgb(${Math.round(255 + (c[0] - 255) * a)},${Math.round(255 + (c[1] - 255) * a)},${Math.round(255 + (c[2] - 255) * a)})`; };
const sig = (z: number) => 1 / (1 + Math.exp(-z));
type Fit = { proba: (q: number[]) => number; note?: [string, string] };
type Ctl = { key: string; label: string; min: number; max: number; step: number; show: (v: number) => string } | { key: string; label: string; choices: string[] };
interface Cfg { kinds: Kind[]; init: { kind: Kind; noise: number; p: Record<string, number> }; controls: Ctl[]; fit: (X: number[][], y: number[], p: Record<string, number>) => Fit; breakLabel: string; breakTo: { kind?: Kind; noise?: number; p?: Record<string, number> }; extra?: (test: { X: number[][]; y: number[] }, proba: (q: number[]) => number) => [string, string][]; }

function Lab({ cfg }: { cfg: Cfg }) {
  const [kind, setKind] = useState<Kind>(cfg.init.kind), [noise, setNoise] = useState(cfg.init.noise), [p, setP] = useState(cfg.init.p);
  const d = useMemo(() => makeData(kind, 160, noise), [kind, noise]), sp = useMemo(() => trainTestSplit(160, 0.3, 2), []);
  const tr = { X: sp.train.map((i) => d.X[i]), y: sp.train.map((i) => d.y[i]) }, te = { X: sp.test.map((i) => d.X[i]), y: sp.test.map((i) => d.y[i]) };
  const f = useMemo(() => cfg.fit(tr.X, tr.y, p), [d, p]);
  const acc = (s: { X: number[][]; y: number[] }) => s.X.filter((q, i) => (f.proba(q) >= 0.5 ? 1 : 0) === s.y[i]).length / s.y.length;
  const cells = useMemo(() => Array.from({ length: GW * GH }, (_, i) => f.proba([((i % GW) + 0.5) / GW, 1 - (Math.floor(i / GW) + 0.5) / GH])), [f]);
  const set = (k: string, v: number) => setP({ ...p, [k]: v }), px = (v: number) => 4 + v * (S - 8), py = (v: number) => 4 + (1 - v) * (S - 8), cs = (S - 8) / GW;
  const extras = cfg.extra ? cfg.extra(te, f.proba) : [];
  return (<div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 14 }}>
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0,420px) 1fr", gap: 18, alignItems: "start" }} className="lab">
      <svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label="Decision regions with training and test points" style={{ width: "100%", height: "auto", background: "#fff", borderRadius: 14, border: "1px solid var(--line)" }}>
        {cells.map((v, i) => <rect key={i} x={4 + (i % GW) * cs} y={4 + Math.floor(i / GW) * cs} width={cs + 0.4} height={cs + 0.4} fill={mix(v)} opacity={0.85} />)}
        {tr.X.map((q, i) => <circle key={i} cx={px(q[0])} cy={py(q[1])} r={4.8} fill={tr.y[i] ? C.matcha : C.rose} stroke="#26301F" strokeWidth={1.4} />)}
        {te.X.map((q, i) => <circle key={`t${i}`} cx={px(q[0])} cy={py(q[1])} r={4.8} fill="#fff" stroke={te.y[i] ? C.matcha : C.rose} strokeWidth={2.2} />)}
      </svg>
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <Row><Stat k="train accuracy" v={`${(acc(tr) * 100).toFixed(0)}%`} /><Stat k="test accuracy" v={`${(acc(te) * 100).toFixed(0)}%`} />{f.note && <Stat k={f.note[0]} v={f.note[1]} />}{extras.map(([k, v]) => <Stat key={k} k={k} v={v} />)}</Row>
        <div role="group" aria-label="Dataset" style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>{cfg.kinds.map((k) => <button key={k} className="btn" onClick={() => setKind(k)} aria-pressed={kind === k} style={{ minHeight: 44, padding: "0 16px", background: kind === k ? "var(--matcha)" : "transparent", color: kind === k ? "#fff" : "var(--ink)", borderColor: "var(--line)" }}>{k}</button>)}</div>
        <Slider label="Noise in the data" value={noise} min={0.02} max={0.2} step={0.01} onChange={setNoise} show={noise.toFixed(2)} />
        {cfg.controls.map((c) => "choices" in c
          ? <div key={c.key} role="group" aria-label={c.label} style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>{c.choices.map((o, i) => <button key={o} className="btn" onClick={() => set(c.key, i)} aria-pressed={p[c.key] === i} style={{ minHeight: 44, padding: "0 14px", background: p[c.key] === i ? "var(--matcha)" : "transparent", color: p[c.key] === i ? "#fff" : "var(--ink)", borderColor: "var(--line)" }}>{o}</button>)}</div>
          : <Slider key={c.key} label={c.label} value={p[c.key]} min={c.min} max={c.max} step={c.step} onChange={(v) => set(c.key, v)} show={c.show(p[c.key])} />)}
        <Row><Btn kind="break" onClick={() => { if (cfg.breakTo.kind) setKind(cfg.breakTo.kind); if (cfg.breakTo.noise !== undefined) setNoise(cfg.breakTo.noise); setP({ ...p, ...(cfg.breakTo.p ?? {}) }); }}>{cfg.breakLabel}</Btn>
          <Btn onClick={() => { setKind(cfg.init.kind); setNoise(cfg.init.noise); setP(cfg.init.p); }}>Reset</Btn></Row>
      </div></div>
    <p className="mute" style={{ margin: 0, fontSize: 13 }}>Filled dots trained the model. Rings were held back for testing. Green and rose regions show which class it predicts, deeper colour means more sure.</p>
    <style>{`@media(max-width:760px){.lab{grid-template-columns:1fr!important}}`}</style>
  </div>);
}

const poly2 = (q: number[]) => [q[0], q[1], q[0] * q[0], q[1] * q[1], q[0] * q[1]];
const conf = (te: { X: number[][]; y: number[] }, pr: (q: number[]) => number) => te.X.reduce((a, q) => a + Math.max(pr(q), 1 - pr(q)), 0) / te.X.length;
const fmtC = (v: number) => (10 ** v >= 100 ? (10 ** v).toExponential(0) : (10 ** v).toPrecision(2));

const CFG: Record<string, Cfg> = {
  logistic: { kinds: ["blobs", "moons"], init: { kind: "blobs", noise: 0.1, p: { c: 0 } }, breakLabel: "Break it: separable data, no penalty", breakTo: { kind: "blobs", noise: 0.03, p: { c: 4 } },
    controls: [{ key: "c", label: "Penalty (C: higher = weaker penalty)", min: -3, max: 4, step: 0.5, show: fmtC }],
    fit: (X, y, p) => { const m = fitLogistic(X, y, 10 ** p.c, 200); return { proba: (q) => m.proba([q])[0], note: ["weight size", Math.hypot(...m.coef).toFixed(1)] }; } },
  boundaries: { kinds: ["blobs", "moons", "circles", "xor"], init: { kind: "moons", noise: 0.08, p: { m: 0 } }, breakLabel: "Break it: XOR with a straight line", breakTo: { kind: "xor", noise: 0.08, p: { m: 0 } },
    controls: [{ key: "m", label: "Model", choices: ["Straight line", "Curved (squares)", "kNN (k = 5)", "Tree (depth 6)"] }],
    fit: (X, y, p) => { if (p.m === 0) { const m = fitLogistic(X, y, 1e3); return { proba: (q) => m.proba([q])[0] }; }
      if (p.m === 1) { const m = fitLogistic(X.map(poly2), y, 1e3, 200); return { proba: (q) => m.proba([poly2(q)])[0] }; }
      if (p.m === 2) return { proba: (q) => knnProba(X, y, q, 5) }; const t = fitTree(X, y, { maxDepth: 6 }); return { proba: (q) => treeProba(t, q) }; } },
  knn: { kinds: ["moons", "blobs", "circles"], init: { kind: "moons", noise: 0.1, p: { k: 7 } }, breakLabel: "Break it: k = 1 on noisy data", breakTo: { kind: "moons", noise: 0.18, p: { k: 1 } },
    controls: [{ key: "k", label: "Neighbours (k)", min: 1, max: 61, step: 2, show: (v) => String(v) }], fit: (X, y, p) => ({ proba: (q) => knnProba(X, y, q, p.k) }) },
  bayes: { kinds: ["blobs", "moons"], init: { kind: "blobs", noise: 0.1, p: { m: 1 } }, breakLabel: "Break it: copy a feature 8 times", breakTo: { p: { m: 8 } },
    controls: [{ key: "m", label: "Copies of the first feature", min: 1, max: 12, step: 1, show: (v) => String(v) }],
    fit: (X, y, p) => { const w = (q: number[]) => [...Array(p.m).fill(q[0]), q[1]], nb = fitGaussianNB(X.map(w), y); return { proba: (q) => nb.proba(w(q)) }; },
    extra: (te, pr) => [["average confidence", `${(conf(te, pr) * 100).toFixed(0)}%`]] },
  tree: { kinds: ["moons", "blobs", "xor", "circles"], init: { kind: "moons", noise: 0.1, p: { d: 3 } }, breakLabel: "Break it: grow the tree to depth 12", breakTo: { kind: "moons", noise: 0.16, p: { d: 12 } },
    controls: [{ key: "d", label: "Maximum depth", min: 1, max: 12, step: 1, show: (v) => String(v) }],
    fit: (X, y, p) => { const t = fitTree(X, y, { maxDepth: p.d }); return { proba: (q) => treeProba(t, q), note: ["depth reached", String(treeDepth(t))] }; } },
  forest: { kinds: ["moons", "circles", "blobs", "xor"], init: { kind: "moons", noise: 0.14, p: { n: 40, d: 8 } }, breakLabel: "Break it: a single deep tree", breakTo: { p: { n: 1, d: 12 } },
    controls: [{ key: "n", label: "Trees", min: 1, max: 80, step: 1, show: (v) => String(v) }, { key: "d", label: "Maximum depth of each tree", min: 1, max: 12, step: 1, show: (v) => String(v) }],
    fit: (X, y, p) => { const f = fitForest(X, y, { trees: p.n, maxDepth: p.d, seed: 3 }); return { proba: (q) => f.proba(q) }; } },
  boosting: { kinds: ["moons", "circles", "blobs", "xor"], init: { kind: "moons", noise: 0.14, p: { r: 25, lr: 0.3 } }, breakLabel: "Break it: many rounds, big steps, noisy data", breakTo: { kind: "moons", noise: 0.2, p: { r: 80, lr: 1 } },
    controls: [{ key: "r", label: "Rounds of boosting", min: 1, max: 80, step: 1, show: (v) => String(v) }, { key: "lr", label: "Learning rate", min: 0.05, max: 1, step: 0.05, show: (v) => v.toFixed(2) }],
    fit: (X, y, p) => { const g = fitBoost(X, y, { rounds: p.r, lr: p.lr, depth: 2 }); return { proba: (q) => g.proba(q) }; } },
  svm: { kinds: ["moons", "circles", "blobs", "xor"], init: { kind: "moons", noise: 0.1, p: { c: 0, g: 0.7 } }, breakLabel: "Break it: gamma = 100", breakTo: { p: { g: 2 } },
    controls: [{ key: "c", label: "C (higher = fewer slips allowed)", min: -2, max: 3, step: 0.5, show: fmtC }, { key: "g", label: "Gamma (higher = tighter reach)", min: -1, max: 2.5, step: 0.25, show: fmtC }],
    fit: (X, y, p) => { const m = fitSVM(X, y, { C: 10 ** p.c, gamma: 10 ** p.g * 1.5, seed: 1 }); return { proba: (q) => sig(3 * m.decision(q)), note: ["support vectors", String(m.sv.length)] }; } },
};
export const PART2 = ["logistic", "boundaries", "knn", "bayes", "tree", "forest", "boosting", "svm"].map((k) => `lab-${k}`);
export function Widget2({ name }: { name: string }) { const [on, setOn] = useState(false); useEffect(() => setOn(true), []); const c = CFG[name.replace("lab-", "")]; return c && on ? <Lab cfg={c} key={name} /> : <div style={{ minHeight: 420 }} aria-busy="true" />; }
