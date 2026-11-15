"use client";
import { useEffect, useMemo, useState } from "react";
import { adjustedRand, agglomerative, dbscan, eigSym, fitGMM, fitIsolation, fitPCA, kmeans, makeCloud, makeHighD, rng, silhouette, standardiseCols, tsne, type Cov, type Linkage, type Shape } from "@/lib/ml";
import { Btn, Row, Slider, Stat } from "./widgets";

export const S = 380, PAL = ["#58792F", "#C4416B", "#D9A441", "#5B7DB1", "#8A6BB1", "#3E8E8A", "#B5651D", "#7A7F6E"], NOISE = "#B9BDB0";
export const col = (l: number) => (l < 0 ? NOISE : PAL[l % PAL.length]);
export const Pick = ({ label, items, value, set }: { label: string; items: string[]; value: string; set: (v: string) => void }) => (
  <div role="group" aria-label={label} className="chips">{items.map((k) => <button key={k} className="px-chip" onClick={() => set(k)} aria-pressed={value === k}>{k}</button>)}</div>
);
export const Layout = ({ left, right, note }: { left: React.ReactNode; right: React.ReactNode; note: string }) => (
  <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 14 }}>
    <div className="lab" style={{ display: "grid", gridTemplateColumns: "minmax(0,420px) 1fr", gap: 18, alignItems: "start" }}>{left}<div style={{ display: "flex", flexDirection: "column", gap: 12 }}>{right}</div></div>
    <p className="mute" style={{ margin: 0, fontSize: 13 }}>{note}</p><style>{`@media(max-width:760px){.lab{grid-template-columns:minmax(0,1fr)!important}}`}</style>
  </div>
);
type Dom = [number, number];
export function Plot({ X, colors, label, dom = [0, 1], rings, children, size = 4.6 }: { X: number[][]; colors: string[]; label: string; dom?: Dom; rings?: boolean[]; children?: React.ReactNode; size?: number }) {
  const p = (v: number) => 6 + ((v - dom[0]) / (dom[1] - dom[0])) * (S - 12);
  return (<svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label={label} style={{ width: "100%", height: "auto", background: "var(--screen)", borderRadius: 0, border: "3px solid var(--ink)" }}>
    {children}{X.map((q, i) => <circle key={i} cx={p(q[0])} cy={S - p(q[1])} r={size} fill={colors[i]} stroke={rings?.[i] ? "#26301F" : "#fff"} strokeWidth={rings?.[i] ? 2.2 : 1} opacity={0.92} />)}</svg>);
}
export const Mini = ({ series, hi, w = 300, h = 110, label }: { series: { v: number[]; c: string; n: string }[]; hi: number; w?: number; h?: number; label: string }) => {
  const all = series.flatMap((s) => s.v), lo = Math.min(...all), up = Math.max(...all), n = series[0].v.length, x = (i: number) => 14 + (i / (n - 1)) * (w - 28), y = (v: number) => h - 14 - ((v - lo) / (up - lo || 1)) * (h - 28);
  return (<svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label} style={{ width: "100%", maxWidth: 360, height: "auto" }}>
    {series.map((s) => <g key={s.n}><polyline fill="none" stroke={s.c} strokeWidth={2} points={s.v.map((v, i) => `${x(i)},${y(v)}`).join(" ")} /><circle cx={x(hi)} cy={y(s.v[hi])} r={5} fill={s.c} /></g>)}
    {series.map((s, i) => <text key={s.n} x={14 + i * Math.min(110, (w - 28) / series.length)} y={10} fontSize={11} fill={s.c} fontWeight={700}>{s.n}</text>)}
  </svg>);
};

function KMeansLab() {
  const [shape, setShape] = useState<Shape>("blobs"), [k, setK] = useState(3), [step, setStep] = useState(13), [seed, setSeed] = useState(1), [init, setInit] = useState("k-means++");
  const { X, truth } = useMemo(() => makeCloud(shape), [shape]), rnd = init === "random";
  const r = useMemo(() => kmeans(X, k, { seed, random: rnd, maxIter: step === 13 ? 300 : step }), [X, k, seed, rnd, step]);
  const sweep = useMemo(() => Array.from({ length: 8 }, (_, i) => kmeans(X, i + 1, { seed, random: rnd })), [X, seed, rnd]);
  const sil = silhouette(X, r.labels);
  return <Layout note="Big dots are the centres. Each point takes the colour of its nearest centre. Slide the iteration count to watch the centres move." left={<Plot X={X} colors={r.labels.map(col)} label="k-means clusters and centres" >{r.centers.map((c, j) => <circle key={j} cx={6 + c[0] * (S - 12)} cy={S - (6 + c[1] * (S - 12))} r={9} fill={col(j)} stroke="#26301F" strokeWidth={3} />)}</Plot>}
    right={<>
      <Row><Stat k="inertia" v={r.inertia.toFixed(2)} /><Stat k="silhouette" v={k < 2 ? "n/a" : sil.toFixed(2)} /><Stat k="agrees with true groups" v={adjustedRand(truth, r.labels).toFixed(2)} /></Row>
      <Pick label="Dataset" items={["blobs", "uneven", "stretched", "circles"]} value={shape} set={(v) => setShape(v as Shape)} />
      <Slider label="Number of clusters (k)" value={k} min={1} max={8} step={1} onChange={setK} show={String(k)} />
      <Slider label="Iterations run" value={step} min={0} max={13} step={1} onChange={setStep} show={step === 13 ? "until it settles" : String(step)} />
      <Pick label="Starting centres" items={["k-means++", "random"]} value={init} set={setInit} />
      <Mini label="Inertia and silhouette for k from 1 to 8" hi={k - 1} series={[{ v: sweep.map((s) => s.inertia), c: "#C4416B", n: "inertia" }, { v: sweep.map((s, i) => (i ? silhouette(X, s.labels) : 0)), c: "#58792F", n: "silhouette" }]} />
      <Row><Btn onClick={() => setSeed(seed + 1)}>New random start</Btn><Btn kind="break" onClick={() => { setShape("circles"); setK(2); setStep(13); }}>Break it: rings and a straight cut</Btn><Btn onClick={() => { setShape("blobs"); setK(3); setStep(13); setSeed(1); setInit("k-means++"); }}>Reset</Btn></Row></>} />;
}

function ClustersLab() {
  const [shape, setShape] = useState<Shape>("moons"), [method, setMethod] = useState("DBSCAN"), [k, setK] = useState(2), [link, setLink] = useState<Linkage>("single"), [eps, setEps] = useState(0.07), [minPts, setMinPts] = useState(4);
  const { X, truth } = useMemo(() => makeCloud(shape, 140), [shape]);
  const labels = useMemo(() => method === "k-means" ? kmeans(X, k, { seed: 2 }).labels : method === "DBSCAN" ? dbscan(X, eps, minPts) : agglomerative(X, k, link), [X, method, k, link, eps, minPts]);
  const found = new Set(labels.filter((l) => l >= 0)).size, noise = labels.filter((l) => l < 0).length;
  return <Layout note="Grey dots are noise: DBSCAN refuses to force them into a group. k-means and hierarchical clustering always assign every point." left={<Plot X={X} colors={labels.map(col)} label="Clusters found by the chosen method" rings={labels.map((l) => l < 0)} />}
    right={<>
      <Row><Stat k="groups found" v={String(found)} /><Stat k="noise points" v={String(noise)} /><Stat k="agrees with true groups" v={adjustedRand(truth, labels).toFixed(2)} /></Row>
      <Pick label="Dataset" items={["moons", "circles", "blobs", "uneven"]} value={shape} set={(v) => setShape(v as Shape)} />
      <Pick label="Method" items={["k-means", "DBSCAN", "hierarchical"]} value={method} set={setMethod} />
      {method === "DBSCAN" ? <><Slider label="Neighbourhood radius (eps)" value={eps} min={0.02} max={0.2} step={0.005} onChange={setEps} show={eps.toFixed(3)} /><Slider label="Neighbours needed to be core" value={minPts} min={2} max={12} step={1} onChange={setMinPts} show={String(minPts)} /></>
        : <Slider label="Number of clusters" value={k} min={2} max={6} step={1} onChange={setK} show={String(k)} />}
      {method === "hierarchical" && <Pick label="Linkage" items={["single", "complete", "average", "ward"]} value={link} set={(v) => setLink(v as Linkage)} />}
      <Row><Btn kind="break" onClick={() => { setShape("moons"); setMethod("k-means"); setK(2); }}>Break it: k-means on moons</Btn><Btn onClick={() => { setShape("moons"); setMethod("DBSCAN"); setEps(0.07); setMinPts(4); }}>Reset</Btn></Row></>} />;
}

function PcaLab() {
  const [u, setU] = useState(0), [std, setStd] = useState(false), [keep1, setKeep1] = useState(false);
  const raw = useMemo(() => { const r = rng(11), g = () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r()); return Array.from({ length: 120 }, () => { const a = g(); return [a, 0.7 * a + 0.35 * g()]; }); }, []);
  const D = useMemo(() => { const sc = raw.map((p) => [p[0], p[1] * 10 ** u]); return std ? standardiseCols(sc) : sc; }, [raw, u, std]);
  const p = useMemo(() => fitPCA(D), [D]);
  const lo = [0, 1].map((j) => Math.min(...D.map((q) => q[j]))), hi = [0, 1].map((j) => Math.max(...D.map((q) => q[j]))), sc = Math.max(hi[0] - lo[0], hi[1] - lo[1]) || 1;
  const nrm = (q: number[]) => [0.5 + (q[0] - (lo[0] + hi[0]) / 2) / sc * 0.88, 0.5 + (q[1] - (lo[1] + hi[1]) / 2) / sc * 0.88];
  const P = D.map(nrm), px = (v: number) => 6 + v * (S - 12), py = (v: number) => S - (6 + v * (S - 12));
  const arrow = (j: number, c: string) => { const a = nrm(p.mean), b = nrm(p.mean.map((m, t) => m + p.components[j][t] * Math.sqrt(p.values[j]) * 2.2)); return <line x1={px(a[0])} y1={py(a[1])} x2={px(b[0])} y2={py(b[1])} stroke={c} strokeWidth={4} strokeLinecap="round" />; };
  const rec = keep1 ? D.map((q) => nrm(p.reconstruct(q, 1))) : null;
  const err = keep1 ? D.reduce((s, q) => s + p.reconstruct(q, 1).reduce((t, v, j) => t + (v - q[j]) ** 2, 0), 0) / D.length : 0, tot = p.values[0] + p.values[1];
  return <Layout note="Both axes use the same scale. Raise the units and the cloud flattens into a line, which is exactly what PCA sees. The thick arrow is the first principal component."
    left={<svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label="Two-feature data with principal component arrows" style={{ width: "100%", height: "auto", background: "var(--screen)", borderRadius: 0, border: "3px solid var(--ink)" }}>
      {rec && P.map((q, i) => <line key={i} x1={px(q[0])} y1={py(q[1])} x2={px(rec[i][0])} y2={py(rec[i][1])} stroke="#F1C5CD" strokeWidth={1.5} />)}
      {P.map((q, i) => <circle key={i} cx={px(q[0])} cy={py(q[1])} r={4} fill="#58792F" opacity={0.8} />)}{rec && rec.map((q, i) => <circle key={`r${i}`} cx={px(q[0])} cy={py(q[1])} r={2.6} fill="#C4416B" />)}
      {arrow(1, "#D9A441")}{arrow(0, "#C4416B")}</svg>}
    right={<>
      <Row><Stat k="first component share" v={`${(p.ratio[0] * 100).toFixed(1)}%`} /><Stat k="second component share" v={`${(p.ratio[1] * 100).toFixed(1)}%`} />{keep1 && <Stat k="detail lost" v={`${((err / tot) * 100).toFixed(1)}%`} />}</Row>
      <Slider label="Units of the second feature (multiply by)" value={u} min={0} max={3} step={0.25} onChange={setU} show={`×${(10 ** u).toFixed(u >= 1 ? 0 : 1)}`} />
      <Row><Btn kind={std ? "primary" : "ghost"} onClick={() => setStd(!std)}>{std ? "Standardised: on" : "Standardised: off"}</Btn><Btn kind={keep1 ? "primary" : "ghost"} onClick={() => setKeep1(!keep1)}>{keep1 ? "Keeping 1 component" : "Keeping 2 components"}</Btn></Row>
      <Row><Btn kind="break" onClick={() => { setU(3); setStd(false); }}>Break it: change the units</Btn><Btn onClick={() => { setU(0); setStd(false); setKeep1(false); }}>Reset</Btn></Row></>} />;
}

function TsneLab() {
  const [perp, setPerp] = useState(15), [equal, setEqual] = useState(false), [seed, setSeed] = useState(1);
  const sp = equal ? [0.5, 0.5, 0.5] : [0.25, 1.0, 0.6], { X, truth } = useMemo(() => makeHighD(90, 8, sp), [equal]);
  const Y = useMemo(() => tsne(X, { perplexity: perp, iters: 350, seed }), [X, perp, seed]), pc = useMemo(() => { const p = fitPCA(X); return X.map((q) => p.project(q, 2)); }, [X]);
  const fit = (A: number[][]) => { const lo = [0, 1].map((j) => Math.min(...A.map((q) => q[j]))), hi = [0, 1].map((j) => Math.max(...A.map((q) => q[j]))), s = Math.max(hi[0] - lo[0], hi[1] - lo[1]) || 1; return A.map((q) => [0.04 + ((q[0] - lo[0]) / s) * 0.92, 0.04 + ((q[1] - lo[1]) / s) * 0.92]); };
  const radius = (A: number[][], g: number) => { const m = A.filter((_, i) => truth[i] === g), c = m[0].map((_, j) => m.reduce((s, q) => s + q[j], 0) / m.length); return m.reduce((s, q) => s + Math.hypot(...q.map((v, j) => v - c[j])), 0) / m.length; };
  const purity = (() => { let a = 0, t = 0; Y.forEach((p, i) => { Y.map((q, j) => [Math.hypot(p[0] - q[0], p[1] - q[1]), j]).filter(([, j]) => j !== i).sort((x, y) => x[0] - y[0]).slice(0, 8).forEach(([, j]) => { t++; if (truth[j] === truth[i]) a++; }); }); return a / t; })();
  const trueR = radius(X, 1) / radius(X, 0), drawnR = radius(Y, 1) / radius(Y, 0), colors = truth.map(col);
  return <Layout note="Left: PCA, a straight-line squash. Right: t-SNE. Colours are the true groups, which neither method was shown."
    left={<div style={{ display: "grid", gap: 10, gridTemplateColumns: "1fr 1fr" }}><div><div className="mute" style={{ font: "600 12px var(--sans)", marginBottom: 4 }}>PCA</div><Plot X={fit(pc)} colors={colors} label="PCA view of the clusters" size={3.6} /></div><div><div className="mute" style={{ font: "600 12px var(--sans)", marginBottom: 4 }}>t-SNE</div><Plot X={fit(Y)} colors={colors} label="t-SNE view of the clusters" size={3.6} /></div></div>}
    right={<>
      <Row><Stat k="neighbours in own group" v={`${(purity * 100).toFixed(0)}%`} /><Stat k="big group vs small: true size" v={`${trueR.toFixed(1)}×`} /><Stat k="…as drawn by t-SNE" v={`${drawnR.toFixed(1)}×`} /></Row>
      <Slider label="Perplexity" value={perp} min={2} max={29} step={1} onChange={setPerp} show={String(perp)} />
      <Pick label="Cluster spreads" items={["different", "equal"]} value={equal ? "equal" : "different"} set={(v) => setEqual(v === "equal")} />
      <Row><Btn onClick={() => setSeed(seed + 1)}>Re-run t-SNE</Btn><Btn kind="break" onClick={() => { setPerp(2); setEqual(false); }}>Break it: perplexity 2</Btn><Btn onClick={() => { setPerp(15); setEqual(false); setSeed(1); }}>Reset</Btn></Row></>} />;
}

function GmmLab() {
  const [shape, setShape] = useState<Shape>("blobs"), [k, setK] = useState(3), [cov, setCov] = useState<Cov>("full");
  const { X } = useMemo(() => makeCloud(shape, 150), [shape]);
  const fits = useMemo(() => Object.fromEntries((["spherical", "diag", "full"] as Cov[]).map((c) => [c, Array.from({ length: 7 }, (_, i) => fitGMM(X, i + 1, { cov: c, seed: 1, maxIter: 150 }))])), [X]);
  const m = fits[cov][k - 1], mixc = (r: number[]) => { let c = [0, 0, 0]; r.forEach((w, j) => { const h = PAL[j % PAL.length]; [1, 3, 5].forEach((o, t) => (c[t] += w * parseInt(h.slice(o, o + 2), 16))); }); return `rgb(${c.map((v) => Math.round(v)).join(",")})`; };
  const px = (v: number) => 6 + v * (S - 12), py = (v: number) => S - (6 + v * (S - 12)), unsure = m.resp.filter((r) => Math.max(...r) < 0.8).length;
  const ell = m.covs.map((c, j) => { const e = eigSym(c), a = Math.atan2(e.vectors[0][1], e.vectors[0][0]) * (180 / Math.PI); return <ellipse key={j} cx={px(m.means[j][0])} cy={py(m.means[j][1])} rx={2 * Math.sqrt(Math.max(e.values[0], 1e-12)) * (S - 12)} ry={2 * Math.sqrt(Math.max(e.values[1], 1e-12)) * (S - 12)} transform={`rotate(${-a} ${px(m.means[j][0])} ${py(m.means[j][1])})`} fill="none" stroke={PAL[j % PAL.length]} strokeWidth={3} />; });
  return <Layout note="Ellipses cover about 95% of each component. Point colours blend when the model is unsure which component owns them."
    left={<Plot X={X} colors={m.resp.map(mixc)} label="Gaussian mixture components and soft assignments">{ell}</Plot>}
    right={<>
      <Row><Stat k="log-likelihood" v={m.loglik.toFixed(0)} /><Stat k="BIC (lower is better)" v={m.bic.toFixed(0)} /><Stat k="unsure points" v={String(unsure)} /></Row>
      <Pick label="Dataset" items={["blobs", "uneven", "stretched"]} value={shape} set={(v) => setShape(v as Shape)} />
      <Slider label="Components" value={k} min={1} max={7} step={1} onChange={setK} show={String(k)} />
      <Pick label="Shape of each component" items={["spherical", "diag", "full"]} value={cov} set={(v) => setCov(v as Cov)} />
      <Mini label="BIC against number of components for three covariance shapes" hi={k - 1} series={(["spherical", "diag", "full"] as Cov[]).map((c, i) => ({ v: fits[c].map((f) => f.bic), c: ["#5B7DB1", "#D9A441", "#C4416B"][i], n: c }))} />
      <Row><Btn kind="break" onClick={() => { setShape("blobs"); setK(7); setCov("full"); }}>Break it: too many components</Btn><Btn onClick={() => { setShape("blobs"); setK(3); setCov("full"); }}>Reset</Btn></Row></>} />;
}

const OUT = [[-0.1, -0.1], [1.1, 1.05], [-0.12, 0.5], [0.5, 1.15], [1.12, 0.2], [0.2, -0.12], [0.9, 1.12], [-0.1, 0.95], [1.15, 0.6], [0.6, -0.1], [0.05, 1.1], [1.1, -0.08]];
function AnomalyLab() {
  const [shape, setShape] = useState<Shape>("blobs"), [trees, setTrees] = useState(60), [rate, setRate] = useState(4), [nOut, setNOut] = useState(4);
  const base = useMemo(() => makeCloud(shape, 150).X.map((p) => p.map((v) => 0.08 + v * 0.84)), [shape]), X = useMemo(() => [...base, ...OUT.slice(0, nOut)], [base, nOut]);
  const f = useMemo(() => fitIsolation(X, { trees, seed: 3 }), [X, trees]), sc = useMemo(() => X.map((q) => f.score(q)), [f, X]);
  const cut = Math.ceil((rate / 100) * X.length), order = sc.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]), flag = new Set(order.slice(0, cut).map(([, i]) => i));
  const caught = [...flag].filter((i) => i >= base.length).length, wrong = flag.size - caught;
  const grid = useMemo(() => Array.from({ length: 28 * 28 }, (_, i) => f.score([-0.2 + ((i % 28) + 0.5) / 28 * 1.4, -0.2 + (Math.floor(i / 28) + 0.5) / 28 * 1.4])), [f]), mn = Math.min(...grid), mx = Math.max(...grid), cs = (S - 12) / 28;
  return <Layout note="Rose background: the forest finds that region easy to isolate, so a point there is suspicious. Rings mark the points flagged as anomalies."
    left={<Plot X={X} dom={[-0.2, 1.2]} colors={X.map((_, i) => (i >= base.length ? "#26301F" : "#58792F"))} rings={X.map((_, i) => flag.has(i))} label="Isolation forest scores and flagged points" size={4.4}>
      {grid.map((v, i) => <rect key={i} x={6 + (i % 28) * cs} y={6 + (27 - Math.floor(i / 28)) * cs} width={cs + 1.2} height={cs + 1.2} fill={`rgb(${[196, 65, 107].map((c) => Math.round(255 + (c - 255) * ((v - mn) / (mx - mn || 1)) * 0.55)).join(",")})`} />)}</Plot>}
    right={<>
      <Row><Stat k="flagged" v={String(flag.size)} /><Stat k="real anomalies caught" v={`${caught} of ${nOut}`} /><Stat k="false alarms" v={String(wrong)} tone={wrong > 3 ? "bad" : undefined} /></Row>
      <Pick label="Normal data" items={["blobs", "moons", "uneven"]} value={shape} set={(v) => setShape(v as Shape)} />
      <Slider label="Planted anomalies (dark dots)" value={nOut} min={0} max={12} step={1} onChange={setNOut} show={String(nOut)} />
      <Slider label="Share of points flagged" value={rate} min={0} max={25} step={1} onChange={setRate} show={`${rate}%`} />
      <Slider label="Trees in the forest" value={trees} min={1} max={100} step={1} onChange={setTrees} show={String(trees)} />
      <Row><Btn kind="break" onClick={() => { setRate(25); setNOut(2); }}>Break it: flag a quarter of everything</Btn><Btn onClick={() => { setShape("blobs"); setTrees(60); setRate(4); setNOut(4); }}>Reset</Btn></Row></>} />;
}

const LABS: Record<string, () => React.ReactElement> = { "lab-kmeans": KMeansLab, "lab-clusters": ClustersLab, "lab-pca": PcaLab, "lab-tsne": TsneLab, "lab-gmm": GmmLab, "lab-anomaly": AnomalyLab };
export const PART4 = Object.keys(LABS);
export function Widget4({ name }: { name: string }) { const [on, setOn] = useState(false); useEffect(() => setOn(true), []); const L = LABS[name] as (() => React.ReactElement) | undefined; return L && on ? <L key={name} /> : <div style={{ minHeight: 420 }} aria-busy="true" />; }
