"use client";
import { useEffect, useMemo, useState } from "react";
import { ACT, centred, conv2d, convOutSize, deadUnits, fitPerceptron, gradCheck, layerGradNorms, lossGrad, makeData, makeNet, netAccuracy, optimisePath, predictProba, trainNet, trainTestSplit, type Act, type Bug, type Kind, type Method } from "@/lib/ml";
import { Btn, Row, Slider, Stat, useReduced } from "./widgets";
import { Layout, Mini, Pick, S } from "./widgets4";

const R = [196, 65, 107], G = [88, 121, 47], GW = 36;
const mix = (p: number) => { const t = Math.min(1, Math.max(0, p)), a = Math.abs(t - 0.5) * 2 * 0.85, c = t > 0.5 ? G : R; return `rgb(${[0, 1, 2].map((i) => Math.round(255 + (c[i] - 255) * a)).join(",")})`; };
const px = (v: number) => 6 + v * (S - 12), py = (v: number) => S - (6 + v * (S - 12));
const BOX: React.CSSProperties = { width: "100%", height: "auto", background: "var(--screen)", borderRadius: 0, border: "3px solid var(--ink)" };
const Dots = ({ X, y, rings }: { X: number[][]; y: number[]; rings?: boolean[] }) => <>{X.map((q, i) => <circle key={i} cx={px(q[0])} cy={py(q[1])} r={4.4} fill={y[i] ? "#58792F" : "#C4416B"} stroke={rings?.[i] ? "#26301F" : "#fff"} strokeWidth={rings?.[i] ? 2.4 : 1} />)}</>;
const Field = ({ f }: { f: (q: number[]) => number }) => { const cs = (S - 12) / GW; return <>{Array.from({ length: GW * GW }, (_, i) => <rect key={i} x={6 + (i % GW) * cs} y={6 + Math.floor(i / GW) * cs} width={cs + 1.2} height={cs + 1.2} fill={mix(f([2 * ((i % GW) + 0.5) / GW - 1, 2 * (1 - (Math.floor(i / GW) + 0.5) / GW) - 1]))} />)}</>; };
const split = (X: number[][], y: number[], seed = 2) => { const s = trainTestSplit(X.length, 0.3, seed); return { tr: { X: s.train.map((i) => X[i]), y: s.train.map((i) => y[i]) }, te: { X: s.test.map((i) => X[i]), y: s.test.map((i) => y[i]) } }; };
const tint = (c: number[], a: number) => `rgb(${c.map((v) => Math.round(255 + (v - 255) * a)).join(",")})`;
const nice = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : "blew up");

function PerceptronLab() {
  const [kind, setKind] = useState("separable"), [ep, setEp] = useState(0), [play, setPlay] = useState(false), reduced = useReduced();
  const d = useMemo(() => (kind === "XOR" ? makeData("xor", 80, 0.07) : makeData("blobs", 80, kind === "separable" ? 0.05 : 0.2)), [kind]);
  const fit = useMemo(() => fitPerceptron(d.X, d.y, 40), [d]), h = fit.hist[ep], bad = d.X.map((q, i) => (d.y[i] * 2 - 1) * (h.w[0] * q[0] + h.w[1] * q[1] + h.b) <= 0);
  useEffect(() => { if (!play) return; const t = window.setInterval(() => setEp((e) => { if (e >= 40) { setPlay(false); return e; } return e + 1; }), 380); return () => window.clearInterval(t); }, [play]);
  const settled = fit.mistakes.findIndex((m) => m === 0), line = Math.abs(h.w[1]) > 1e-9 ? [[0, -(h.b) / h.w[1]], [1, -(h.w[0] + h.b) / h.w[1]]] : Math.abs(h.w[0]) > 1e-9 ? [[-h.b / h.w[0], 0], [-h.b / h.w[0], 1]] : null;
  const maxM = Math.max(...fit.mistakes, 1);
  return <Layout note="The line moves only when a point is on the wrong side. Rings mark points currently misclassified. Bars show mistakes per pass over the data."
    left={<svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label="Perceptron boundary and data" style={BOX}>{line && <line x1={px(line[0][0])} y1={py(line[0][1])} x2={px(line[1][0])} y2={py(line[1][1])} stroke="#26301F" strokeWidth={3} />}<Dots X={d.X} y={d.y} rings={bad} /></svg>}
    right={<>
      <Row><Stat k="mistakes now" v={String(bad.filter(Boolean).length)} /><Stat k="settled" v={settled >= 0 ? `after ${settled + 1} passes` : "never (in 40 passes)"} tone={settled < 0 ? "bad" : undefined} /></Row>
      <Pick label="Dataset" items={["separable", "overlapping", "XOR"]} value={kind} set={(v) => { setKind(v); setEp(0); setPlay(false); }} />
      <Slider label="Passes over the data" value={ep} min={0} max={40} step={1} onChange={(v) => { setEp(v); setPlay(false); }} show={String(ep)} />
      <svg viewBox="0 0 320 70" role="img" aria-label="Mistakes per pass" style={{ width: "100%", maxWidth: 360 }}>{fit.mistakes.map((m, i) => <rect key={i} x={4 + i * 7.7} y={64 - (m / maxM) * 56} width={5.5} height={(m / maxM) * 56 + 0.5} rx={1.5} fill={i < ep ? "#C4416B" : "#F1C5CD"} />)}</svg>
      <Row><Btn kind="primary" onClick={() => { if (reduced) { setEp(40); return; } setEp(0); setPlay(true); }}>{play ? "Playing…" : "Play"}</Btn><Btn kind="break" onClick={() => { setKind("XOR"); setEp(40); setPlay(false); }}>Break it: XOR</Btn><Btn onClick={() => { setKind("separable"); setEp(0); setPlay(false); }}>Reset</Btn></Row></>} />;
}

function MlpLab() {
  const [kind, setKind] = useState<Kind>("circles"), [layers, setLayers] = useState(2), [width, setWidth] = useState(8), [ep, setEp] = useState(300), [act, setAct] = useState<Act>("tanh");
  const raw = useMemo(() => makeData(kind, 160, 0.08), [kind]), d = useMemo(() => ({ X: centred(raw.X), y: raw.y }), [raw]), sp = useMemo(() => split(d.X, d.y), [d]);
  const run = useMemo(() => { const net = makeNet([2, ...Array(layers).fill(width), 1], { act, seed: 1 }), h = trainNet(net, sp.tr.X, sp.tr.y, { epochs: ep, lr: 0.05, every: 5 }); return { net, h }; }, [sp, layers, width, ep, act]);
  const params = run.net.W.reduce((s, w) => s + w.length, 0) + run.net.b.reduce((s, b) => s + b.length, 0);
  return <Layout note="Dots with a dark rim trained the network. White dots with a coloured centre were held back for testing. Background colour is what it predicts at every spot."
    left={<svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label="Network decision regions" style={BOX}><Field f={(q) => predictProba(run.net, q)} />{sp.tr.X.map((q, i) => <circle key={i} cx={px((q[0] + 1) / 2)} cy={py((q[1] + 1) / 2)} r={4.4} fill={sp.tr.y[i] ? "#58792F" : "#C4416B"} stroke="#26301F" strokeWidth={1.3} />)}{sp.te.X.map((q, i) => <g key={`t${i}`}><circle cx={px((q[0] + 1) / 2)} cy={py((q[1] + 1) / 2)} r={5.4} fill="#fff" stroke="#26301F" strokeWidth={1.6} /><circle cx={px((q[0] + 1) / 2)} cy={py((q[1] + 1) / 2)} r={2.6} fill={sp.te.y[i] ? "#58792F" : "#C4416B"} /></g>)}</svg>}
    right={<>
      <Row><Stat k="train accuracy" v={`${(netAccuracy(run.net, sp.tr.X, sp.tr.y) * 100).toFixed(0)}%`} /><Stat k="test accuracy" v={`${(netAccuracy(run.net, sp.te.X, sp.te.y) * 100).toFixed(0)}%`} /><Stat k="weights and biases" v={String(params)} /></Row>
      <Pick label="Dataset" items={["circles", "xor", "moons", "blobs"]} value={kind} set={(v) => setKind(v as Kind)} />
      <Slider label="Hidden layers" value={layers} min={0} max={3} step={1} onChange={setLayers} show={String(layers)} />
      <Slider label="Units per layer" value={width} min={1} max={16} step={1} onChange={setWidth} show={String(width)} />
      <Slider label="Training passes" value={ep} min={0} max={300} step={10} onChange={setEp} show={String(ep)} />
      <Pick label="Activation" items={["tanh", "relu"]} value={act} set={(v) => setAct(v as Act)} />
      {run.h.loss.length > 1 && <Mini label="Training loss over passes" hi={run.h.loss.length - 1} series={[{ v: run.h.loss, c: "#C4416B", n: "training loss" }]} />}
      <Row><Btn kind="break" onClick={() => { setKind("circles"); setLayers(0); setEp(300); }}>Break it: no hidden layer</Btn><Btn onClick={() => { setKind("circles"); setLayers(2); setWidth(8); setEp(300); setAct("tanh"); }}>Reset</Btn></Row></>} />;
}

function BackpropLab() {
  const [bug, setBug] = useState<Bug>("none"), [ep, setEp] = useState(150);
  const d = useMemo(() => { const r = makeData("moons", 100, 0.1); return { X: centred(r.X), y: r.y }; }, []);
  const base = useMemo(() => makeNet([2, 6, 6, 1], { act: "tanh", seed: 3 }), []);
  const rows = useMemo(() => { const g = lossGrad(base, d.X, d.y, { bug }), h = 1e-5, spots: [string, Float64Array, number, Float64Array][] = [["first-layer weight 1", base.W[0], 0, g.W[0]], ["first-layer weight 2", base.W[0], 1, g.W[0]], ["middle weight", base.W[1], 5, g.W[1]], ["last-layer weight", base.W[2], 2, g.W[2]], ["first-layer bias", base.b[0], 0, g.b[0]], ["last bias", base.b[2], 0, g.b[2]]];
    return spots.map(([n, arr, k, gr]) => { const v = arr[k]; arr[k] = v + h; const up = lossGrad(base, d.X, d.y).loss; arr[k] = v - h; const dn = lossGrad(base, d.X, d.y).loss; arr[k] = v; return { n, back: gr[k], num: (up - dn) / (2 * h) }; }); }, [base, d, bug]);
  const err = useMemo(() => gradCheck(base, d.X, d.y, { bug, count: 6 }), [base, d, bug]);
  const runs = useMemo(() => (["none", bug] as Bug[]).map((b) => trainNet(makeNet([2, 6, 6, 1], { act: "tanh", seed: 3 }), d.X, d.y, { epochs: ep, lr: 0.02, bug: b, every: 5 })), [d, bug, ep]);
  const bad = err > 1e-6;
  return <Layout note="Backprop gives an exact slope for every weight in one sweep. The slow check nudges each weight and measures the change in loss. A correct implementation makes the two columns agree."
    left={<div style={{ ...BOX, padding: 14, overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse", font: "13px var(--mono)" }}><thead><tr style={{ textAlign: "left", color: "var(--mute)" }}><th>weight</th><th>backprop</th><th>nudge test</th></tr></thead><tbody>{rows.map((r) => { const off = Math.abs(r.back - r.num) / Math.max(1e-8, Math.abs(r.back) + Math.abs(r.num)) > 1e-4; return <tr key={r.n} style={{ borderTop: "1px solid var(--line)", color: off ? "var(--rose)" : "var(--ink)" }}><td style={{ padding: "7px 0", font: "600 13px var(--sans)" }}>{r.n}</td><td>{r.back.toFixed(5)}</td><td>{r.num.toFixed(5)}</td></tr>; })}</tbody></table></div>}
    right={<>
      <Row><Stat k="worst relative error" v={err.toExponential(1)} tone={bad ? "bad" : undefined} /><Stat k="check" v={bad ? "FAILS" : "passes"} tone={bad ? "bad" : undefined} /></Row>
      <Pick label="Backprop version" items={["correct", "no activation slope", "flipped bias sign"]} value={bug === "none" ? "correct" : bug === "no-act-deriv" ? "no activation slope" : "flipped bias sign"} set={(v) => setBug(v === "correct" ? "none" : v === "no activation slope" ? "no-act-deriv" : "sign")} />
      <Slider label="Training passes" value={ep} min={10} max={300} step={10} onChange={setEp} show={String(ep)} />
      <Mini label="Loss when training with correct and chosen gradients" hi={runs[0].loss.length - 1} series={[{ v: runs[0].loss, c: "#58792F", n: "correct" }, { v: runs[1].loss, c: "#C4416B", n: bug === "none" ? "same" : "buggy" }]} />
      <Row><Btn kind="break" onClick={() => setBug("no-act-deriv")}>Break it: forget the activation slope</Btn><Btn onClick={() => { setBug("none"); setEp(150); }}>Reset</Btn></Row></>} />;
}

const KNAME: Record<string, Act> = { sigmoid: "sigmoid", tanh: "tanh", ReLU: "relu", "leaky ReLU": "leaky" };
function ActivationsLab() {
  const [name, setName] = useState("ReLU"), [lr, setLr] = useState(0.5);
  const d = useMemo(() => { const r = makeData("circles", 100, 0.08); return { X: centred(r.X), y: r.y }; }, []);
  const runs = useMemo(() => Object.entries(KNAME).map(([n, a]) => { const net = makeNet([2, 8, 8, 1], { act: a, seed: 1 }), h = trainNet(net, d.X, d.y, { epochs: 200, lr, method: "sgd", every: 5 }); return { n, a, h, net }; }), [d, lr]);
  const cols: Record<string, string> = { sigmoid: "#5B7DB1", tanh: "#D9A441", ReLU: "#C4416B", "leaky ReLU": "#58792F" }, f = ACT[KNAME[name]], zs = Array.from({ length: 121 }, (_, i) => -6 + i * 0.1);
  const X = (z: number) => 20 + ((z + 6) / 12) * 300, Y = (v: number) => 130 - ((v + 1.5) / 4) * 120;
  const relu = runs.find((r) => r.n === "ReLU")!;
  return <Layout note="Top: the chosen activation (solid) and its slope (dashed). Backprop multiplies by that slope at every layer, so where the slope is near 0, learning stalls. Plain SGD is used on purpose: it shows the differences that Adam would hide."
    left={<div style={{ ...BOX, padding: 12 }}><svg viewBox="0 0 340 150" role="img" aria-label="Activation function and its slope" style={{ width: "100%", height: "auto" }}><line x1={20} y1={Y(0)} x2={320} y2={Y(0)} stroke="#d9d2c8" /><line x1={X(0)} y1={10} x2={X(0)} y2={130} stroke="#d9d2c8" />
      <polyline fill="none" stroke={cols[name]} strokeWidth={3} points={zs.map((z) => `${X(z)},${Y(Math.max(-1.45, Math.min(2.4, f.f(z))))}`).join(" ")} /><polyline fill="none" stroke={cols[name]} strokeWidth={2} strokeDasharray="5 4" points={zs.map((z) => `${X(z)},${Y(f.d(z))}`).join(" ")} /></svg>
      <div className="mute" style={{ font: "600 12px var(--sans)", margin: "8px 0 2px" }}>Training loss, same network, four activations</div>
      <Mini label="Training loss for each activation" hi={runs[0].h.loss.length - 1} h={120} series={runs.map((r) => ({ v: r.h.loss, c: cols[r.n], n: r.n }))} /></div>}
    right={<>
      <Row><Stat k={`${name}: final loss`} v={nice(runs.find((r) => r.n === name)!.h.loss.at(-1)!)} /><Stat k="ReLU units that never fire" v={`${(deadUnits(relu.net, d.X) * 100).toFixed(0)}%`} tone={deadUnits(relu.net, d.X) > 0.3 ? "bad" : undefined} /></Row>
      <Pick label="Activation to inspect" items={Object.keys(KNAME)} value={name} set={setName} />
      <Slider label="Learning rate (SGD)" value={lr} min={0.05} max={3} step={0.05} onChange={setLr} show={lr.toFixed(2)} />
      <Row><Btn kind="break" onClick={() => { setName("ReLU"); setLr(3); }}>Break it: learning rate 3</Btn><Btn onClick={() => { setName("ReLU"); setLr(0.5); }}>Reset</Btn></Row></>} />;
}

type Surf = { f: (p: number[]) => number; g: (p: number[]) => number[]; start: number[]; dom: [number, number, number, number]; lr: number };
const SURF: Record<string, Surf> = {
  ravine: { f: ([x, y]) => 0.5 * x * x + 12.5 * y * y, g: ([x, y]) => [x, 25 * y], start: [-1.3, 0.8], dom: [-1.5, 1.5, -1, 1], lr: 0.04 },
  banana: { f: ([x, y]) => (1 - x) ** 2 + 5 * (y - x * x) ** 2, g: ([x, y]) => [-2 * (1 - x) - 20 * x * (y - x * x), 10 * (y - x * x)], start: [-1.2, 1], dom: [-1.6, 1.6, -0.6, 2.2], lr: 0.01 },
};
const MCOL: Record<Method, string> = { sgd: "#7A7F6E", momentum: "#D9A441", rmsprop: "#5B7DB1", adam: "#C4416B" };
function OptimLab() {
  const [sf, setSf] = useState("ravine"), [lr, setLr] = useState(0.04), [steps, setSteps] = useState(60), s = SURF[sf];
  const paths = useMemo(() => (Object.keys(MCOL) as Method[]).map((m) => ({ m, p: optimisePath(s.g, s.start, m, m === "momentum" ? lr / 2 : m === "sgd" ? lr : lr * 2.5, steps) })), [s, lr, steps]);
  const [x0, x1, y0, y1] = s.dom, X = (v: number) => 6 + ((v - x0) / (x1 - x0)) * (S - 12), Y = (v: number) => S - (6 + ((v - y0) / (y1 - y0)) * (S - 12));
  const cells = useMemo(() => Array.from({ length: 30 * 30 }, (_, i) => Math.log(1 + s.f([x0 + (((i % 30) + 0.5) / 30) * (x1 - x0), y0 + ((Math.floor(i / 30) + 0.5) / 30) * (y1 - y0)]))), [s]), mx = Math.max(...cells), cs = (S - 12) / 30;
  const fin = (p: number[][]) => { const e = p[p.length - 1], v = s.f(e); return Number.isFinite(v) && v < 1e4 ? v : Infinity; };
  return <Layout note={`Starting point is the same for all four. SGD uses the learning rate shown, momentum half of it, RMSprop and Adam 2.5 times (they scale steps by gradient size, so their useful range differs).`}
    left={<svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label="Optimiser paths over a loss surface" style={BOX}>{cells.map((v, i) => <rect key={i} x={6 + (i % 30) * cs} y={6 + (29 - Math.floor(i / 30)) * cs} width={cs + 1.2} height={cs + 1.2} fill={tint(R, (v / mx) * 0.4)} />)}
      {paths.map(({ m, p }) => { const pts = p.filter((q) => q.every((v) => Number.isFinite(v)) && Math.abs(X(q[0])) < 2000 && Math.abs(Y(q[1])) < 2000); return <g key={m}><polyline fill="none" stroke={MCOL[m]} strokeWidth={2.6} strokeLinejoin="round" points={pts.map((q) => `${X(q[0])},${Y(q[1])}`).join(" ")} />{pts.length > 0 && <circle cx={X(pts[pts.length - 1][0])} cy={Y(pts[pts.length - 1][1])} r={5} fill={MCOL[m]} />}</g>; })}
      <circle cx={X(s.start[0])} cy={Y(s.start[1])} r={6} fill="#fff" stroke="#26301F" strokeWidth={2.5} /></svg>}
    right={<>
      <Row>{paths.map(({ m, p }) => <Stat key={m} k={`${m}: final loss`} v={Number.isFinite(fin(p)) ? fin(p).toFixed(3) : "blew up"} tone={!Number.isFinite(fin(p)) ? "bad" : undefined} />)}</Row>
      <Pick label="Loss surface" items={["ravine", "banana"]} value={sf} set={(v) => { setSf(v); setLr(SURF[v].lr); }} />
      <Slider label="Learning rate" value={lr} min={0.002} max={0.12} step={0.002} onChange={setLr} show={lr.toFixed(3)} />
      <Slider label="Steps" value={steps} min={1} max={200} step={1} onChange={setSteps} show={String(steps)} />
      <div className="mute" style={{ font: "600 12px var(--sans)", display: "flex", gap: 12, flexWrap: "wrap" }}>{(Object.keys(MCOL) as Method[]).map((m) => <span key={m} style={{ color: MCOL[m] }}>● {m}</span>)}</div>
      <Row><Btn kind="break" onClick={() => { setSf("ravine"); setLr(0.1); setSteps(60); }}>Break it: learning rate too high</Btn><Btn onClick={() => { setSf("ravine"); setLr(0.04); setSteps(60); }}>Reset</Btn></Row></>} />;
}

function VanishLab() {
  const [depth, setDepth] = useState(10), [act, setAct] = useState<Act>("sigmoid"), [init, setInit] = useState("auto"), [res, setRes] = useState(false);
  const d = useMemo(() => { const r = makeData("moons", 80, 0.1); return { X: centred(r.X), y: r.y }; }, []);
  const out = useMemo(() => { const sz = [2, ...Array(depth).fill(8), 1], ini = init === "small" ? 0.1 : init === "large" ? 1.5 : "auto", net = makeNet(sz, { act, init: ini, residual: res, seed: 1 }), g = layerGradNorms(net, d.X, d.y), h = trainNet(makeNet(sz, { act, init: ini, residual: res, seed: 1 }), d.X, d.y, { epochs: 150, lr: 0.1, method: "sgd", every: 150 }); return { g, loss: h.loss.at(-1)! }; }, [depth, act, init, res, d]);
  const logs = out.g.map((v) => Math.log10(Math.max(v, 1e-16))), lo = Math.min(-1, ...logs), hi = Math.max(0.5, ...logs), ratio = out.g[0] / out.g[out.g.length - 1];
  const bw = Math.min(26, 300 / out.g.length - 3);
  return <Layout note="Each bar is the size of the gradient reaching one layer, earliest layer on the left, on a log scale. Short bars on the left mean the early layers receive almost no signal about their mistakes."
    left={<div style={{ ...BOX, padding: 14 }}><svg viewBox="0 0 340 200" role="img" aria-label="Gradient size by layer" style={{ width: "100%", height: "auto" }}><line x1={14} y1={170} x2={330} y2={170} stroke="#d9d2c8" />{out.g.map((v, i) => { const h = ((logs[i] - lo) / (hi - lo)) * 150; return <rect key={i} x={16 + i * (bw + 3)} y={170 - Math.max(h, 1.5)} width={bw} height={Math.max(h, 1.5)} rx={3} fill={i === 0 ? "#C4416B" : "#58792F"} />; })}<text x={16} y={192} fontSize={11} fill="#66705A">first layer</text><text x={330} y={192} fontSize={11} fill="#66705A" textAnchor="end">last layer</text></svg></div>}
    right={<>
      <Row><Stat k="first vs last layer gradient" v={Number.isFinite(ratio) ? ratio.toExponential(1) : "n/a"} tone={ratio < 1e-3 ? "bad" : undefined} /><Stat k="loss after 150 steps" v={nice(out.loss)} tone={!Number.isFinite(out.loss) || out.loss > 0.68 ? "bad" : undefined} /></Row>
      <Slider label="Hidden layers" value={depth} min={2} max={16} step={1} onChange={setDepth} show={String(depth)} />
      <Pick label="Activation" items={["sigmoid", "tanh", "relu"]} value={act} set={(v) => setAct(v as Act)} />
      <Pick label="Starting weights" items={["small", "auto", "large"]} value={init} set={setInit} />
      <Row><Btn kind={res ? "primary" : "ghost"} onClick={() => setRes(!res)}>{res ? "Skip connections: on" : "Skip connections: off"}</Btn></Row>
      <Row><Btn kind="break" onClick={() => { setDepth(12); setAct("sigmoid"); setInit("auto"); setRes(false); }}>Break it: 12 sigmoid layers</Btn><Btn onClick={() => { setDepth(10); setAct("relu"); setInit("auto"); setRes(false); }}>Fix it: ReLU and sensible weights</Btn></Row></>} />;
}

const IMG: Record<string, (r: number, c: number) => number> = { "thin line": (r, c) => (c === 5 ? 1 : 0), box: (r, c) => ((r >= 3 && r <= 10 && c >= 3 && c <= 10) && (r === 3 || r === 10 || c === 3 || c === 10) ? 1 : 0), diagonal: (r, c) => (Math.abs(r - c) <= 0 ? 1 : 0), blob: (r, c) => ((r - 7) ** 2 + (c - 7) ** 2 <= 16 ? 1 : 0) };
const KERN: Record<string, number[][]> = { "vertical edges": [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], "horizontal edges": [[-1, -1, -1], [0, 0, 0], [1, 1, 1]], blur: Array(3).fill(Array(3).fill(1 / 9)), sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] };
function Grid({ m, label, big }: { m: number[][]; label: string; big: number }) {
  const mx = Math.max(...m.flat().map(Math.abs), 1e-9), n = m.length, w = m[0]?.length ?? 0, c = Math.min(big, 200 / Math.max(n, w, 1));
  return <svg viewBox={`0 0 ${Math.max(w * c, 4)} ${Math.max(n * c, 4)}`} role="img" aria-label={label} style={{ width: "100%", maxWidth: Math.max(w * c, 60), height: "auto", border: "1px solid var(--line)", background: "#fff" }}>{m.map((r, i) => r.map((v, j) => <rect key={`${i}-${j}`} x={j * c} y={i * c} width={c} height={c} fill={v >= 0 ? "#58792F" : "#C4416B"} opacity={Math.min(1, Math.abs(v) / mx)} stroke="#f1ece4" strokeWidth={0.5} />))}</svg>;
}
function ConvLab() {
  const [img, setImg] = useState("thin line"), [kern, setKern] = useState("vertical edges"), [stride, setStride] = useState(1), [pad, setPad] = useState(0);
  const I = useMemo(() => Array.from({ length: 14 }, (_, r) => Array.from({ length: 14 }, (_, c) => IMG[img](r, c))), [img]), K = KERN[kern], out = useMemo(() => conv2d(I, K, stride, pad), [I, K, stride, pad]);
  const n = convOutSize(14, 3, pad, stride), mx = Math.max(0, ...out.flat().map(Math.abs));
  return <Layout note="The same 9 numbers slide over every position, so one small detector finds its pattern anywhere. Green is a positive response, rose is negative."
    left={<div style={{ ...BOX, padding: 14, display: "grid", gap: 14, gridTemplateColumns: "1fr 1fr", alignItems: "start" }}><div><div className="mute" style={{ font: "600 12px var(--sans)", marginBottom: 4 }}>Image 14 × 14</div><Grid m={I} label="Input image" big={14} /></div><div><div className="mute" style={{ font: "600 12px var(--sans)", marginBottom: 4 }}>Output {n} × {n}</div>{out.length ? <Grid m={out} label="Feature map" big={14} /> : <p className="mute">Kernel does not fit.</p>}</div></div>}
    right={<>
      <Row><Stat k="output size" v={`⌊(14 − 3 + 2·${pad}) / ${stride}⌋ + 1 = ${n}`} /><Stat k="strongest response" v={mx.toFixed(1)} tone={mx === 0 ? "bad" : undefined} /></Row>
      <Row><Stat k="detector weights" v="9 (+1 bias)" /><Stat k="dense layer, same job" v={`${(14 * 14 * Math.max(n, 1) ** 2).toLocaleString()} weights`} /></Row>
      <Pick label="Image" items={Object.keys(IMG)} value={img} set={setImg} />
      <Pick label="Detector" items={Object.keys(KERN)} value={kern} set={setKern} />
      <Slider label="Stride" value={stride} min={1} max={4} step={1} onChange={setStride} show={String(stride)} />
      <Slider label="Padding" value={pad} min={0} max={2} step={1} onChange={setPad} show={String(pad)} />
      <Row><Btn kind="break" onClick={() => { setImg("thin line"); setKern("vertical edges"); setStride(2); setPad(0); }}>Break it: stride 2 on a thin line</Btn><Btn onClick={() => { setImg("thin line"); setKern("vertical edges"); setStride(1); setPad(0); }}>Reset</Btn></Row></>} />;
}

function DropoutLab() {
  const [rate, setRate] = useState(0);
  const all = useMemo(() => { const r = makeData("moons", 260, 0.28); return { X: centred(r.X), y: r.y }; }, []), tr = useMemo(() => ({ X: all.X.slice(0, 60), y: all.y.slice(0, 60) }), [all]), te = useMemo(() => ({ X: all.X.slice(60), y: all.y.slice(60) }), [all]);
  const run = useMemo(() => { const net = makeNet([2, 16, 16, 1], { act: "relu", seed: 1 }), h = trainNet(net, tr.X, tr.y, { epochs: 300, lr: 0.01, drop: rate, test: te, every: 10 }); return { net, h }; }, [tr, te, rate]);
  return <Layout note="Dark-rimmed dots trained the network. White dots with a coloured centre are fresh data it never saw. Watch the gap between the two losses: that gap is overfitting."
    left={<svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label="Network boundary with training and test points" style={BOX}><Field f={(q) => predictProba(run.net, q)} />{te.X.slice(0, 100).map((q, i) => <g key={`t${i}`}><circle cx={px((q[0] + 1) / 2)} cy={py((q[1] + 1) / 2)} r={4.4} fill="#fff" stroke="#26301F" strokeWidth={1.3} /><circle cx={px((q[0] + 1) / 2)} cy={py((q[1] + 1) / 2)} r={2.2} fill={te.y[i] ? "#58792F" : "#C4416B"} /></g>)}{tr.X.map((q, i) => <circle key={i} cx={px((q[0] + 1) / 2)} cy={py((q[1] + 1) / 2)} r={4.6} fill={tr.y[i] ? "#58792F" : "#C4416B"} stroke="#26301F" strokeWidth={1.5} />)}</svg>}
    right={<>
      <Row><Stat k="train accuracy" v={`${(netAccuracy(run.net, tr.X, tr.y) * 100).toFixed(0)}%`} /><Stat k="test accuracy" v={`${(netAccuracy(run.net, te.X, te.y) * 100).toFixed(0)}%`} /><Stat k="test loss minus train loss" v={(run.h.testLoss.at(-1)! - run.h.loss.at(-1)!).toFixed(2)} /></Row>
      <Slider label="Share of units dropped each step" value={rate} min={0} max={0.9} step={0.1} onChange={setRate} show={`${Math.round(rate * 100)}%`} />
      <Mini label="Training and test loss over time" hi={run.h.loss.length - 1} series={[{ v: run.h.loss, c: "#58792F", n: "train loss" }, { v: run.h.testLoss, c: "#C4416B", n: "test loss" }]} />
      <Row><Btn kind="break" onClick={() => setRate(0.9)}>Break it: drop 90%</Btn><Btn onClick={() => setRate(0)}>No dropout</Btn><Btn onClick={() => setRate(0.5)}>Drop half</Btn></Row></>} />;
}

const LABS: Record<string, () => React.ReactElement> = { "lab-perceptron": PerceptronLab, "lab-mlp": MlpLab, "lab-backprop": BackpropLab, "lab-activations": ActivationsLab, "lab-optim": OptimLab, "lab-vanish": VanishLab, "lab-conv": ConvLab, "lab-dropout": DropoutLab };
export const PART5 = Object.keys(LABS);
export function Widget5({ name }: { name: string }) { const [on, setOn] = useState(false); useEffect(() => setOn(true), []); const L = LABS[name] as (() => React.ReactElement) | undefined; return L && on ? <L key={name} /> : <div style={{ minHeight: 420 }} aria-busy="true" />; }
