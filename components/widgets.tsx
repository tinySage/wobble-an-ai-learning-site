"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { biasVariance, diverged, fitLinear, fitPolynomial, gradientDescent, gradientDescentMulti, lambdaMax, mse, rng, stepsToConverge, truth } from "@/lib/ml";

const W = 640, H = 360, PAD = 36;
const sx = (x: number) => PAD + x * (W - 2 * PAD);
const sy = (y: number, lo = -1.6, hi = 1.6) => H - PAD - ((y - lo) / (hi - lo)) * (H - 2 * PAD);
const C = { ink: "var(--ink)", matcha: "var(--matcha)", rose: "var(--rose)", mute: "var(--mute)", soft: "var(--soft)" };

function useReduced() {
  const [r, setR] = useState(false);
  useEffect(() => { const m = window.matchMedia("(prefers-reduced-motion: reduce)"); setR(m.matches); }, []);
  return r;
}
/** Steps a value through a list on a timer (>= 350ms per step so changes are readable). Jumps to the end with reduced motion. */
function useSweep(values: number[], set: (v: number) => void, ms = 380) {
  const reduced = useReduced(), t = useRef<number | null>(null), [on, setOn] = useState(false);
  const stop = () => { if (t.current) window.clearInterval(t.current); t.current = null; setOn(false); };
  useEffect(() => stop, []);
  const play = () => {
    stop(); if (reduced) { set(values[values.length - 1]); return; }
    let i = 0; set(values[0]); setOn(true);
    t.current = window.setInterval(() => { i++; if (i >= values.length) { stop(); return; } set(values[i]); }, ms);
  };
  return { play, on, stop };
}
const Frame = ({ children, label }: { children: React.ReactNode; label: string }) => (
  <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 14 }}><svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} style={{ width: "100%", height: "auto", background: "#fff", borderRadius: 14, border: "1px solid var(--line)" }}>{children}</svg>{null}</div>
);
const Row = ({ children }: { children: React.ReactNode }) => <div style={{ display: "flex", flexWrap: "wrap", gap: 14, alignItems: "center" }}>{children}</div>;
const Btn = ({ onClick, children, kind = "ghost" }: { onClick: () => void; children: React.ReactNode; kind?: "ghost" | "primary" | "break" }) => (
  <button className={`btn ${kind === "primary" ? "primary" : "ghost"}`} style={kind === "break" ? { background: "var(--rose)", color: "#fff", borderColor: "transparent" } : undefined} onClick={onClick}>{children}</button>
);
const Slider = ({ label, value, min, max, step, onChange, show }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; show: string }) => (
  <label style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 220, flex: 1, font: "600 13px var(--sans)" }}>
    <span>{label}: <span className="mono">{show}</span></span>
    <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} aria-label={label} style={{ accentColor: "#58792F", minHeight: 44 }} />
  </label>
);
const Stat = ({ k, v, tone }: { k: string; v: string; tone?: "bad" }) => (
  <div style={{ padding: "8px 14px", border: "1px solid var(--line)", borderRadius: 12, background: "var(--panel)" }}><div className="eyebrow mute" style={{ fontSize: 10 }}>{k}</div><div className="mono" style={{ fontSize: 18, color: tone === "bad" ? "var(--rose)" : undefined }}>{v}</div></div>
);
const Axes = () => <g stroke="rgba(38,48,31,.2)"><line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} /><line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} /></g>;
const fmt = (v: number) => !Number.isFinite(v) ? "blew up" : Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-3) ? v.toExponential(1) : v.toFixed(3);

/* ---------- 1. fit a line, add points, break it with an outlier ---------- */
export function LinearFit() {
  const base = useMemo(() => { const r = rng(3); return Array.from({ length: 10 }, (_, i) => { const x = 0.06 + i * 0.09 + r() * 0.04; return [x, 0.9 * (x - 0.5) * 2 * 0.6 + (r() - 0.5) * 0.35] as [number, number]; }); }, []);
  const [pts, setPts] = useState<[number, number][]>(base);
  const m = useMemo(() => fitLinear(pts.map((p) => [p[0]]), pts.map((p) => p[1])), [pts]);
  const pred = (x: number) => m.coef[0] * x + m.intercept;
  const err = mse(pts.map((p) => pred(p[0])), pts.map((p) => p[1]));
  const add = (e: React.MouseEvent<SVGSVGElement>) => {
    const b = e.currentTarget.getBoundingClientRect(), x = ((e.clientX - b.left) / b.width * W - PAD) / (W - 2 * PAD), yy = (1 - ((e.clientY - b.top) / b.height * H - PAD) / (H - 2 * PAD)) * 3.2 - 1.6;
    if (x >= 0 && x <= 1) setPts([...pts, [x, Math.max(-1.6, Math.min(1.6, yy))]]);
  };
  return (<div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 14 }}>
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Scatter plot with a fitted line. Click to add points." onClick={add} style={{ width: "100%", height: "auto", background: "#fff", borderRadius: 14, border: "1px solid var(--line)", cursor: "crosshair" }}>
      <Axes />
      {pts.map((p, i) => <line key={i} x1={sx(p[0])} x2={sx(p[0])} y1={sy(p[1])} y2={sy(pred(p[0]))} stroke={C.rose} strokeWidth={2} opacity={0.55} />)}
      <line x1={sx(0)} y1={sy(pred(0))} x2={sx(1)} y2={sy(pred(1))} stroke={C.matcha} strokeWidth={3.5} strokeLinecap="round" style={{ transition: "all .35s ease" }} />
      {pts.map((p, i) => <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={6} fill={C.ink} />)}
    </svg>
    <Row><Stat k="slope" v={m.coef[0].toFixed(2)} /><Stat k="intercept" v={m.intercept.toFixed(2)} /><Stat k="mean squared error" v={err.toFixed(3)} /><Stat k="points" v={String(pts.length)} /></Row>
    <Row><Btn kind="break" onClick={() => setPts([...pts, [0.95, -1.5]])}>Break it: add an outlier</Btn><Btn onClick={() => setPts(base)}>Reset</Btn><span className="mute" style={{ fontSize: 13 }}>Click the plot to add your own points. The rose lines are the errors being squared.</span></Row>
  </div>);
}

/* ---------- 2/3. gradient descent and learning rate ---------- */
const GD = (() => { const r = rng(11), x = Array.from({ length: 16 }, (_, i) => i / 15), y = x.map((v) => 2 * v + 0.5 + (r() - 0.5) * 0.4); return { x, y, lim: 2 / lambdaMax(x.map((v) => [v])) }; })();
export function Descent({ rate }: { rate?: boolean }) {
  const [lr, setLr] = useState(rate ? 0.3 : 0.25), [step, setStep] = useState(0), [run, setRun] = useState(false), reduced = useReduced(), auto = useRef(false);
  const traj = useMemo(() => gradientDescent(GD.x, GD.y, lr, 80), [lr]);
  const bad = diverged(traj), best = useMemo(() => { const m = fitLinear(GD.x.map((v) => [v]), GD.y); return mse(GD.x.map((v) => m.coef[0] * v + m.intercept), GD.y); }, []);
  useEffect(() => { setStep(0); setRun(auto.current); auto.current = false; }, [lr]);
  useEffect(() => { if (!run) return; if (reduced) { setStep(traj.length - 1); setRun(false); return; } const t = window.setInterval(() => setStep((s) => { if (s >= traj.length - 1) { setRun(false); return s; } return s + 1; }), 90); return () => window.clearInterval(t); }, [run, traj, reduced]);
  const s = traj[Math.min(step, traj.length - 1)], line = (x: number) => s.w * x + s.b;
  const shown = traj.slice(0, step + 1).map((t) => t.loss), top = Math.max(...traj.slice(0, 80).map((t) => Math.min(t.loss, 1e6)), 1);
  const lx = (i: number) => PAD + (i / 79) * (W - 2 * PAD), ly = (l: number) => H - PAD - (Math.log10(Math.max(l, 1e-4)) + 4) / (Math.log10(top) + 4) * (H - 2 * PAD);
  return (<div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 14 }}>
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={rate ? "Loss against step, log scale" : "Data with the line gradient descent is currently using"} style={{ width: "100%", height: "auto", background: "#fff", borderRadius: 14, border: "1px solid var(--line)" }}>
      <Axes />
      {rate ? <>
        <polyline fill="none" stroke={bad ? C.rose : C.matcha} strokeWidth={3} points={shown.map((l, i) => `${lx(i)},${Math.max(PAD, Math.min(H - PAD, ly(l)))}`).join(" ")} />
        <line x1={PAD} x2={W - PAD} y1={ly(best)} y2={ly(best)} stroke={C.mute} strokeDasharray="5 5" /><text x={W - PAD} y={ly(best) - 6} textAnchor="end" fontSize={11} fill={C.mute}>best possible</text>
        <text x={PAD + 6} y={PAD + 12} fontSize={11} fill={C.mute}>loss (log scale)</text></> : <>
        {GD.x.map((x, i) => <line key={i} x1={sx(x)} x2={sx(x)} y1={sy(GD.y[i], -0.5, 3)} y2={sy(line(x), -0.5, 3)} stroke={C.rose} opacity={0.4} strokeWidth={2} />)}
        <line x1={sx(0)} x2={sx(1)} y1={sy(line(0), -0.5, 3)} y2={sy(line(1), -0.5, 3)} stroke={bad ? C.rose : C.matcha} strokeWidth={3.5} strokeLinecap="round" />
        {GD.x.map((x, i) => <circle key={i} cx={sx(x)} cy={sy(GD.y[i], -0.5, 3)} r={5.5} fill={C.ink} />)}</>}
    </svg>
    <Row><Stat k="step" v={String(step)} /><Stat k="loss" v={fmt(s.loss)} tone={bad ? "bad" : undefined} /><Stat k="slope / intercept" v={`${fmt(s.w)} / ${fmt(s.b)}`} />{rate && <Stat k="stable below" v={GD.lim.toFixed(2)} />}</Row>
    <Row><Slider label="Learning rate" value={lr} min={0.02} max={1.4} step={0.01} onChange={setLr} show={lr.toFixed(2)} />
      <Btn kind="primary" onClick={() => { setStep(0); setRun(true); }}>{run ? "Running…" : "Run"}</Btn>{rate && <Btn kind="break" onClick={() => { auto.current = true; setLr(+(GD.lim * 1.08).toFixed(2)); }}>Break it: go past the limit</Btn>}</Row>
    <p className="mute" style={{ margin: 0, fontSize: 13 }} aria-live="polite">{bad ? "Each step overshoots further than the last. The loss grows instead of falling." : step >= 79 || traj[step].loss <= best * 1.01 ? "Settled close to the best fit." : "Stepping downhill."}</p>
  </div>);
}

/* ---------- 4. feature scaling ---------- */
export function Scaling() {
  const [scaled, setScaled] = useState(false);
  const d = useMemo(() => { const r = rng(5), raw = Array.from({ length: 30 }, () => [r(), r() * 300]), y = raw.map((v) => 2 * v[0] + 0.01 * v[1] + 1 + (r() - 0.5) * 0.2);
    const mu = [0, 1].map((j) => raw.reduce((a, v) => a + v[j], 0) / 30), sd = [0, 1].map((j) => Math.sqrt(raw.reduce((a, v) => a + (v[j] - mu[j]) ** 2, 0) / 30));
    return { raw, std: raw.map((v) => v.map((t, j) => (t - mu[j]) / sd[j])), y }; }, []);
  const run = (X: number[][]) => { const lim = 2 / lambdaMax(X), lr = 0.9 * lim, g = gradientDescentMulti(X, d.y, lr, 300).losses, m = fitLinear(X, d.y), opt = mse(X.map((v) => m.coef[0] * v[0] + m.coef[1] * v[1] + m.intercept), d.y); return { lr, g, steps: stepsToConverge(g, opt, 0.05) }; };
  const raw = useMemo(() => run(d.raw), [d]), std = useMemo(() => run(d.std), [d]), cur = scaled ? std : raw;
  const both = [...raw.g, ...std.g].filter(Number.isFinite), hiL = Math.log10(Math.max(...both)), loL = Math.log10(Math.max(Math.min(...both), 1e-6)), lx = (i: number) => PAD + (i / 299) * (W - 2 * PAD), ly = (l: number) => H - PAD - ((Math.log10(Math.max(l, 1e-6)) - loL) / (hiL - loL || 1)) * (H - 2 * PAD - 20);
  return (<div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 14 }}>
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Loss against step for raw and scaled features, log scale" style={{ width: "100%", height: "auto", background: "#fff", borderRadius: 14, border: "1px solid var(--line)" }}>
      <Axes /><text x={PAD + 6} y={PAD + 12} fontSize={11} fill={C.mute}>loss (log scale), 300 steps, each at the fastest safe learning rate</text>
      <polyline fill="none" stroke={C.rose} strokeWidth={scaled ? 1.5 : 3.5} opacity={scaled ? 0.3 : 1} points={raw.g.map((l, i) => `${lx(i)},${ly(l)}`).join(" ")} />
      <polyline fill="none" stroke={C.matcha} strokeWidth={scaled ? 3.5 : 1.5} opacity={scaled ? 1 : 0.3} points={std.g.map((l, i) => `${lx(i)},${ly(l)}`).join(" ")} />
    </svg>
    <Row><Stat k="features" v={scaled ? "standardised" : "raw (0–1 and 0–300)"} /><Stat k="learning rate used" v={cur.lr < 0.01 ? cur.lr.toExponential(1) : cur.lr.toFixed(3)} /><Stat k="steps to within 5% of best" v={cur.steps === null ? "not in 300" : String(cur.steps)} tone={cur.steps === null ? "bad" : undefined} /></Row>
    <Row><Btn kind="primary" onClick={() => setScaled(!scaled)}>{scaled ? "Show raw features" : "Standardise the features"}</Btn><span className="mute" style={{ fontSize: 13 }}>Same data, same method. Only the units changed.</span></Row>
  </div>);
}

/* ---------- 5-7. polynomial, overfitting, regularisation ---------- */
const DATA = (() => { const r = rng(7), g = () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r()), noise = 0.22,
  tx = Array.from({ length: 14 }, (_, i) => 0.04 + 0.92 * ((i + r() * 0.8) / 14)), ty = tx.map((v) => truth(v) + g() * noise),
  ex = Array.from({ length: 60 }, (_, i) => 0.05 + (i / 59) * 0.9), ey = ex.map((v) => truth(v) + g() * noise); return { tx, ty, ex, ey }; })();
const CURVE = Array.from({ length: 120 }, (_, i) => 0.02 + (i / 119) * 0.96);
export function PolyLab({ mode }: { mode: "poly" | "overfit" | "reg" }) {
  const [deg, setDeg] = useState(mode === "reg" ? 9 : mode === "poly" ? 3 : 2), [li, setLi] = useState(0);
  const lam = li === 0 ? 0 : 10 ** (li - 8), sweep = useSweep([1, 2, 3, 4, 5, 6, 7, 8, 9], setDeg), sweepL = useSweep([0, 1, 2, 3, 4, 5, 6, 7, 8], setLi);
  const f = useMemo(() => fitPolynomial(DATA.tx, DATA.ty, deg, lam), [deg, lam]);
  const trE = mse(f.predict(DATA.tx), DATA.ty), teE = mse(f.predict(DATA.ex), DATA.ey);
  const all = useMemo(() => [1, 2, 3, 4, 5, 6, 7, 8, 9].map((d) => { const m = fitPolynomial(DATA.tx, DATA.ty, d, lam); return [mse(m.predict(DATA.tx), DATA.ty), mse(m.predict(DATA.ex), DATA.ey)]; }), [lam]);
  const curve = f.predict(CURVE), lo = -1.6, hi = 1.6, clampY = (v: number) => sy(Math.max(lo, Math.min(hi, v)));
  const flat = all.flat().filter((v) => v > 0), eLo = Math.log10(Math.min(...flat)), eHi = Math.log10(Math.max(...flat)), ex = (e: number) => H - PAD - ((Math.log10(Math.max(e, 1e-12)) - eLo) / (eHi - eLo || 1)) * (H - 2 * PAD - 20), bx = (d: number) => PAD + ((d - 1) / 8) * (W - 2 * PAD);
  return (<div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 14 }}>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(280px,1fr))", gap: 12 }}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Training points, test points and the fitted curve" style={{ width: "100%", height: "auto", background: "#fff", borderRadius: 14, border: "1px solid var(--line)" }}>
        <Axes />
        <polyline fill="none" stroke="rgba(38,48,31,.25)" strokeDasharray="4 5" strokeWidth={2} points={CURVE.map((x) => `${sx(x)},${sy(truth(x))}`).join(" ")} />
        {mode !== "poly" && DATA.ex.map((x, i) => i % 2 === 0 && <circle key={i} cx={sx(x)} cy={sy(DATA.ey[i])} r={4} fill="none" stroke={C.rose} strokeWidth={1.6} />)}
        <polyline fill="none" stroke={C.matcha} strokeWidth={3.5} strokeLinejoin="round" points={curve.map((v, i) => `${sx(CURVE[i])},${clampY(v)}`).join(" ")} />
        {DATA.tx.map((x, i) => <circle key={i} cx={sx(x)} cy={sy(DATA.ty[i])} r={5.5} fill={C.ink} />)}
        <text x={PAD + 6} y={PAD + 12} fontSize={11} fill={C.mute}>dark = training · rose rings = unseen test points · dashed = the real pattern</text>
      </svg>
      {mode !== "poly" && <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Training and test error by degree, log scale" style={{ width: "100%", height: "auto", background: "#fff", borderRadius: 14, border: "1px solid var(--line)" }}>
        <Axes /><text x={PAD + 6} y={PAD + 12} fontSize={11} fill={C.mute}>error by degree (log scale). green = training, rose = test</text>
        {[0, 1].map((k) => <polyline key={k} fill="none" stroke={k ? C.rose : C.matcha} strokeWidth={3} points={all.map((e, i) => `${bx(i + 1)},${ex(e[k])}`).join(" ")} />)}
        <line x1={bx(deg)} x2={bx(deg)} y1={PAD} y2={H - PAD} stroke={C.ink} strokeDasharray="3 4" opacity={0.4} />
        {[1, 3, 5, 7, 9].map((d) => <text key={d} x={bx(d)} y={H - 12} fontSize={11} textAnchor="middle" fill={C.mute}>{d}</text>)}
      </svg>}
    </div>
    <Row><Stat k="degree" v={String(deg)} />{mode !== "poly" && <><Stat k="training error" v={fmt(trE)} /><Stat k="test error" v={fmt(teE)} tone={teE > trE * 5 && teE > 0.1 ? "bad" : undefined} /></>}{mode === "reg" && <Stat k="penalty λ" v={lam === 0 ? "0" : lam.toExponential(0)} />}</Row>
    <Row><Slider label="Degree" value={deg} min={1} max={9} step={1} onChange={(v) => { sweep.stop(); setDeg(v); }} show={String(deg)} />
      {mode === "reg" && <Slider label="Penalty strength" value={li} min={0} max={8} step={1} onChange={(v) => { sweepL.stop(); setLi(v); }} show={lam === 0 ? "off" : lam.toExponential(0)} />}</Row>
    <Row>{mode === "overfit" && <Btn kind="break" onClick={sweep.play}>{sweep.on ? "Adding freedom…" : "Break it: add freedom"}</Btn>}
      {mode === "reg" && <><Btn kind="primary" onClick={sweepL.play}>{sweepL.on ? "Tightening…" : "Turn the penalty up"}</Btn><Btn onClick={() => { sweepL.stop(); setLi(0); }}>Penalty off</Btn></>}
      {mode === "poly" && <Btn kind="primary" onClick={sweep.play}>Play degrees 1 to 9</Btn>}
      <span className="mute" style={{ fontSize: 13 }}>Only 14 training points, so results swing with the dataset. Degrees stop at 9 on purpose.</span></Row>
  </div>);
}

/* ---------- 8. bias and variance ---------- */
export function BiasVariance() {
  const [deg, setDeg] = useState(3), [seed, setSeed] = useState(1);
  const all = useMemo(() => [1, 2, 3, 4, 5, 6, 7, 8].map((d) => biasVariance(d, { seed, resamples: 60 })), [seed]);
  const cur = all[deg - 1], top = Math.max(...all.map((a) => a.total)) * 1.05, bw = (W - 2 * PAD) / 8;
  const GRIDX = Array.from({ length: 60 }, (_, i) => 0.05 + (i / 59) * 0.9);
  return (<div style={{ width: "100%", display: "flex", flexDirection: "column", gap: 14 }}>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(280px,1fr))", gap: 12 }}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Many fits from different training sets, and their average" style={{ width: "100%", height: "auto", background: "#fff", borderRadius: 14, border: "1px solid var(--line)" }}>
        <Axes />
        {cur.fits.map((p, k) => <polyline key={k} fill="none" stroke={C.matcha} opacity={0.3} strokeWidth={1.5} points={p.map((v, i) => `${sx(GRIDX[i])},${sy(Math.max(-1.6, Math.min(1.6, v)))}`).join(" ")} />)}
        <polyline fill="none" stroke={C.ink} strokeDasharray="4 5" strokeWidth={2} points={GRIDX.map((x) => `${sx(x)},${sy(truth(x))}`).join(" ")} />
        <polyline fill="none" stroke={C.rose} strokeWidth={3} points={cur.meanFit.map((v, i) => `${sx(GRIDX[i])},${sy(Math.max(-1.6, Math.min(1.6, v)))}`).join(" ")} />
        <text x={PAD + 6} y={PAD + 12} fontSize={11} fill={C.mute}>green = 24 refits · rose = their average · dashed = truth</text>
      </svg>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Bias squared, variance and noise by degree" style={{ width: "100%", height: "auto", background: "#fff", borderRadius: 14, border: "1px solid var(--line)" }}>
        <Axes />
        {all.map((a, i) => { const h = (v: number) => (v / top) * (H - 2 * PAD); let y = H - PAD; const seg = [[a.noise2, C.soft], [a.bias2, C.rose], [a.variance, C.matcha]] as const;
          return <g key={i} opacity={i + 1 === deg ? 1 : 0.55}>{seg.map(([v, c], k) => { y -= h(v); return <rect key={k} x={PAD + i * bw + 6} width={bw - 12} y={y} height={Math.max(0, h(v))} fill={c} />; })}<text x={PAD + i * bw + bw / 2} y={H - 12} textAnchor="middle" fontSize={11} fill={C.mute}>{i + 1}</text></g>; })}
        <text x={PAD + 6} y={PAD + 12} fontSize={11} fill={C.mute}>test error split: rose bias², green variance, pale noise</text>
      </svg>
    </div>
    <Row><Stat k="bias²" v={cur.bias2.toFixed(3)} /><Stat k="variance" v={cur.variance.toFixed(3)} /><Stat k="noise (fixed)" v={cur.noise2.toFixed(3)} /><Stat k="total" v={cur.total.toFixed(3)} /></Row>
    <Row><Slider label="Degree" value={deg} min={1} max={8} step={1} onChange={setDeg} show={String(deg)} /><Btn kind="break" onClick={() => setSeed(seed + 1)}>Break it: draw new training sets</Btn></Row>
  </div>);
}

export function Widget({ name }: { name: string }) {
  switch (name) {
    case "linear-fit": return <LinearFit />;
    case "descent": return <Descent />;
    case "learning-rate": return <Descent rate />;
    case "scaling": return <Scaling />;
    case "poly": return <PolyLab mode="poly" />;
    case "overfit-curve": return <PolyLab mode="overfit" />;
    case "regularisation": return <PolyLab mode="reg" />;
    case "bias-variance": return <BiasVariance />;
    default: return <p className="mute">Widget “{name}” arrives in a later stage.</p>;
  }
}
