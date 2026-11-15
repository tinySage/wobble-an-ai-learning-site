"use client";
import { useEffect, useMemo, useState } from "react";
import { analogy, attention, attentionSpread, baumWelch, cooccur, embedSym, encode, entropy, fitLinear, forwardHMM, greedyPath, groupStats, learnCliff, makeArc, makeCorpus, makeFairness, makeSeries, nearest, permutationImportance, ppmi, predict, qSample, randMatrix, reconError, reconstruct, reverseSample, runBandit, sampleHMM, sampleMix, schedule, scoreSeries, shapleyExact, thresholdFor, trainAE, tsSplit, viterbi, vocabOf, CH, CW, GOAL, START, isCliff, type HMM, type Policy } from "@/lib/ml";
import { Btn, Row, Slider, Stat } from "./widgets";
import { Layout, Mini, Pick, S } from "./widgets4";

const G = "#58792F", R = "#C4416B", AMBER = "#D9A441", BLUE = "#5B7DB1", INK = "#26301F";
const BOX: React.CSSProperties = { width: "100%", height: "auto", background: "#fff", borderRadius: 14, border: "1px solid var(--line)" };
const pct = (v: number) => `${(v * 100).toFixed(0)}%`;

/** Horizontal bars with labels. Positive values go right in green, negative left in rose. */
function HBars({ rows, label, w = 380, fmt = (v: number) => v.toFixed(2), tone }: { rows: { n: string; v: number }[]; label: string; w?: number; fmt?: (v: number) => string; tone?: string }) {
  const mx = Math.max(1e-9, ...rows.map((r) => Math.abs(r.v))), mid = rows.some((r) => r.v < 0) ? w * 0.6 : w * 0.34, span = (rows.some((r) => r.v < 0) ? w * 0.3 : w * 0.56);
  return <svg viewBox={`0 0 ${w} ${rows.length * 30 + 12}`} role="img" aria-label={label} style={BOX}>
    {rows.map((r, i) => <g key={r.n} transform={`translate(0 ${10 + i * 30})`}><text x={8} y={15} fontSize={12} fill={INK}>{r.n}</text>
      <rect x={r.v < 0 ? mid - (Math.abs(r.v) / mx) * span : mid} y={3} width={Math.max(1, (Math.abs(r.v) / mx) * span)} height={18} rx={4} fill={r.v < 0 ? R : tone ?? G} />
      <text x={r.v < 0 ? mid + 6 : mid + (Math.abs(r.v) / mx) * span + 6} y={16} fontSize={11} fill="var(--mute)" fontFamily="var(--mono)">{fmt(r.v)}</text></g>)}
  </svg>;
}

/** Counts as vertical bars (for histograms). */
function Hist({ xs, lo, hi, bins = 40, label, color = G, h = 110 }: { xs: number[]; lo: number; hi: number; bins?: number; label: string; color?: string; h?: number }) {
  const c = new Array(bins).fill(0); xs.forEach((x) => { const b = Math.floor(((x - lo) / (hi - lo)) * bins); if (b >= 0 && b < bins) c[b]++; });
  const mx = Math.max(1, ...c), w = 360;
  return <svg viewBox={`0 0 ${w} ${h + 18}`} role="img" aria-label={label} style={BOX}>
    {c.map((v, i) => <rect key={i} x={8 + (i * (w - 16)) / bins} y={h - (v / mx) * (h - 14)} width={(w - 16) / bins - 1} height={(v / mx) * (h - 14)} fill={color} opacity={0.85} />)}
    <text x={8} y={h + 13} fontSize={11} fill="var(--mute)">{label}</text></svg>;
}

// ---------- 1. Time series ----------
function SeriesLab() {
  const [mode, setMode] = useState<"shuffled" | "walk">("shuffled"), [peek, setPeek] = useState(false);
  const y = useMemo(() => makeSeries(144, 3), []);
  const sc = useMemo(() => ({ shufNo: scoreSeries(y, "shuffled", false), walkNo: scoreSeries(y, "walk", false), shufPeek: scoreSeries(y, "shuffled", true), walkPeek: scoreSeries(y, "walk", true) }), [y]);
  const reported = mode === "shuffled" ? (peek ? sc.shufPeek : sc.shufNo) : peek ? sc.walkPeek : sc.walkNo, honest = sc.walkNo;
  const rows = y.length - 13, folds = tsSplit(rows, 5), W = 420, H = 190, lo = Math.min(...y) - 0.3, hi = Math.max(...y) + 0.3;
  const X = (t: number) => 10 + (t / (y.length - 1)) * (W - 20), Y = (v: number) => H - 10 - ((v - lo) / (hi - lo)) * (H - 20);
  const testT = useMemo(() => { if (mode === "walk") return new Set(folds.flatMap((f) => f.test.map((i) => i + 12))); const s = new Set<number>(); let a = 7; for (let i = 0; i < rows; i++) { a = (a * 48271) % 2147483647; if (a % 5 === 0) s.add(i + 12); } return s; }, [mode, rows]);
  const wrong = reported < honest * 0.6;
  return <Layout note="Rose dots are the points held back for testing. Shuffled folds scatter them among the training points, so the model has seen both neighbours of every test point. Walk-forward trains only on the past, which is how a forecast really works."
    left={<svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Time series with the test points marked" style={BOX}>
      {mode === "walk" && folds.map((f, i) => <rect key={i} x={X(f.test[0] + 12)} y={6} width={X(f.test.at(-1)! + 12) - X(f.test[0] + 12)} height={H - 12} fill={R} opacity={0.1 + i * 0.03} />)}
      <polyline fill="none" stroke={G} strokeWidth={2} points={y.map((v, t) => `${X(t)},${Y(v)}`).join(" ")} />
      {y.map((v, t) => testT.has(t) && <circle key={t} cx={X(t)} cy={Y(v)} r={3.2} fill={R} />)}</svg>}
    right={<>
      <Row><Stat k="error you would report" v={reported.toFixed(3)} tone={wrong ? "bad" : undefined} /><Stat k="honest forecast error" v={honest.toFixed(3)} /></Row>
      <Pick label="How to test" items={["shuffled folds", "walk-forward"]} value={mode === "shuffled" ? "shuffled folds" : "walk-forward"} set={(v) => setMode(v === "walk-forward" ? "walk" : "shuffled")} />
      <Pick label="Extra feature" items={["past values only", "average of yesterday and tomorrow"]} value={peek ? "average of yesterday and tomorrow" : "past values only"} set={(v) => setPeek(v !== "past values only")} />
      {wrong && <p style={{ margin: 0, color: R, fontWeight: 700 }}>This estimate is too good to be true. Tomorrow is not available when you forecast tomorrow.</p>}
      <Row><Btn kind="break" onClick={() => { setMode("shuffled"); setPeek(true); }}>Break it: shuffle and peek</Btn><Btn onClick={() => { setMode("walk"); setPeek(false); }}>Fix it: walk-forward</Btn></Row></>} />;
}

// ---------- 2. Hidden Markov models ----------
const TRUE_HMM: HMM = { pi: [0.6, 0.4], A: [[0.9, 0.1], [0.15, 0.85]], B: [[0.7, 0.25, 0.05], [0.05, 0.3, 0.65]] };
const POOR: HMM = { pi: [0.5, 0.5], A: [[0.6, 0.4], [0.4, 0.6]], B: [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]] };
const FLAT: HMM = { pi: [0.5, 0.5], A: [[0.5, 0.5], [0.5, 0.5]], B: [[0.34, 0.33, 0.33], [0.34, 0.33, 0.33]] };
function HmmLab() {
  const [start, setStart] = useState<"poor" | "flat">("poor"), [passes, setPasses] = useState(15);
  const d = useMemo(() => sampleHMM(TRUE_HMM, 300, 11), []);
  const fit = useMemo(() => baumWelch(start === "poor" ? POOR : FLAT, d.obs, passes), [start, passes, d]);
  const path = useMemo(() => viterbi(fit.model, d.obs).path, [fit, d]);
  const acc = useMemo(() => { const a = path.reduce((s, p, i) => s + (p === d.states[i] ? 1 : 0), 0) / path.length; return Math.max(a, 1 - a); }, [path, d]);
  const common = useMemo(() => { const k = d.states.reduce((a, v) => a + v, 0) / d.states.length; return Math.max(k, 1 - k); }, [d]);
  const post = useMemo(() => { const f = forwardHMM(fit.model, d.obs); return f.alpha.map((a) => a[1]); }, [fit, d]);
  const n = 60, W = 440, cw = (W - 20) / n, split = fit.model.B[0].every((v, k) => Math.abs(v - fit.model.B[1][k]) < 1e-6);
  const symCol = ["#E9E2C8", "#C9D6A8", "#8FB05A"];
  return <Layout note="Top row: what you can see (three kinds of reading). Middle: the hidden state that really produced them. Bottom: the state the model decodes, after learning only from the readings. Labels may be swapped, so accuracy counts the better of the two labelings."
    left={<svg viewBox={`0 0 ${W} 190`} role="img" aria-label="Observations, true hidden state and decoded state" style={BOX}>
      {d.obs.slice(0, n).map((o, i) => <rect key={`o${i}`} x={10 + i * cw} y={14} width={cw - 1} height={28} fill={symCol[o]} />)}
      {d.states.slice(0, n).map((s, i) => <rect key={`s${i}`} x={10 + i * cw} y={62} width={cw - 1} height={28} fill={s ? R : G} opacity={0.85} />)}
      {path.slice(0, n).map((s, i) => <rect key={`p${i}`} x={10 + i * cw} y={110} width={cw - 1} height={28} fill={s ? R : G} opacity={0.85} />)}
      {post.slice(0, n).map((p, i) => <rect key={`q${i}`} x={10 + i * cw} y={170 - p * 24} width={cw - 1} height={p * 24} fill={AMBER} />)}
      <text x={10} y={10} fontSize={11} fill="var(--mute)">readings</text><text x={10} y={58} fontSize={11} fill="var(--mute)">true hidden state</text><text x={10} y={106} fontSize={11} fill="var(--mute)">decoded (Viterbi)</text><text x={10} y={186} fontSize={11} fill="var(--mute)">chance of state 2 given the readings</text></svg>}
    right={<>
      <Row><Stat k="decoding accuracy" v={pct(acc)} tone={acc < common + 0.02 ? "bad" : undefined} /><Stat k="always guess the common state" v={pct(common)} /><Stat k="log-likelihood" v={fit.hist.at(-1)!.toFixed(1)} /><Stat k="states told apart" v={split ? "no" : "yes"} tone={split ? "bad" : undefined} /></Row>
      <Pick label="Starting guess" items={["different guesses", "identical states"]} value={start === "poor" ? "different guesses" : "identical states"} set={(v) => setStart(v === "identical states" ? "flat" : "poor")} />
      <Slider label="Learning passes (Baum-Welch)" value={passes} min={0} max={40} step={1} onChange={setPasses} show={String(passes)} />
      <Mini label="Log-likelihood over passes" hi={fit.hist.length - 1} series={[{ v: fit.hist, c: G, n: "log-likelihood" }]} />
      <Row><Btn kind="break" onClick={() => { setStart("flat"); setPasses(30); }}>Break it: identical starting states</Btn><Btn onClick={() => { setStart("poor"); setPasses(15); }}>Reset</Btn></Row></>} />;
}

// ---------- 3. Embeddings ----------
function EmbedLab() {
  const [dim, setDim] = useState(8), [size, setSize] = useState(400), [word, setWord] = useState("paris");
  const emb = useMemo(() => { const c = makeCorpus(size, 2), v = vocabOf(c), M = ppmi(cooccur(c, v, 2)), e = embedSym(M, dim), p = embedSym(M, 2); return { v, vecs: e.vecs, xy: p.vecs }; }, [dim, size]);
  const has = (w: string) => emb.v.includes(w), tests: [string, string, string, string][] = [["man", "king", "woman", "queen"], ["france", "paris", "italy", "rome"]];
  const ranges = [0, 1].map((j) => [Math.min(...emb.xy.map((p) => p[j])), Math.max(...emb.xy.map((p) => p[j]))]);
  const X = (v: number) => 20 + ((v - ranges[0][0]) / (ranges[0][1] - ranges[0][0] || 1)) * (S - 40), Y = (v: number) => S - 20 - ((v - ranges[1][0]) / (ranges[1][1] - ranges[1][0] || 1)) * (S - 40);
  const nn = has(word) ? nearest(emb.vecs, emb.v, word, 4) : [], group = (w: string) => (["cat", "dog", "horse", "cow", "fish", "bone", "hay", "grass", "eats"].includes(w) ? G : ["paris", "rome", "madrid", "berlin", "france", "italy", "spain", "germany", "city"].includes(w) ? BLUE : ["king", "queen", "prince", "princess", "throne", "sits"].includes(w) ? R : AMBER);
  const words = ["paris", "cat", "king", "woman", "italy"];
  return <Layout note="Each word becomes a list of numbers built only from which words sit near it in a tiny made-up corpus. The picture squashes those numbers to two. Colour marks the kind of word."
    left={<svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label="Word vectors in two dimensions" style={BOX}>{emb.v.map((w, i) => <g key={w}><circle cx={X(emb.xy[i][0])} cy={Y(emb.xy[i][1])} r={w === word ? 6 : 4} fill={group(w)} stroke={w === word ? INK : "#fff"} strokeWidth={w === word ? 2 : 1} opacity={0.9} /><text x={X(emb.xy[i][0]) + 6} y={Y(emb.xy[i][1]) - 5} fontSize={10} fill={INK}>{w}</text></g>)}</svg>}
    right={<>
      <Pick label="Word" items={words} value={word} set={(v) => setWord(v)} />
      <p style={{ margin: 0 }}>Closest to <b>{word}</b>: {nn.length ? nn.map((x) => `${x.w} (${x.s.toFixed(2)})`).join(", ") : "not in this small corpus"}</p>
      {tests.map(([a, b, c, ans]) => { const top = has(a) && has(b) && has(c) ? analogy(emb.vecs, emb.v, a, b, c, 3).map((x) => x.w) : []; const hit = top[0] === ans, near = top.includes(ans); return <p key={ans} style={{ margin: 0 }}>{a} is to {b} as {c} is to <b>{top[0] ?? "?"}</b> <span style={{ color: hit ? G : near ? AMBER : R, fontWeight: 700 }}>{hit ? "✓" : near ? `(${ans} is in the top 3)` : `✗ wanted ${ans}`}</span></p>; })}
      <Slider label="Numbers per word" value={dim} min={1} max={16} step={1} onChange={setDim} show={String(dim)} />
      <Slider label="Sentences in the corpus" value={size} min={20} max={400} step={20} onChange={setSize} show={String(size)} />
      <Row><Btn kind="break" onClick={() => { setDim(1); setSize(40); }}>Break it: one number, tiny corpus</Btn><Btn onClick={() => { setDim(8); setSize(400); }}>Reset</Btn></Row></>} />;
}

// ---------- 4. Attention ----------
const DIMS = [4, 16, 64, 256, 1024];
function AttentionLab() {
  const [dim, setDim] = useState(64), [scaled, setScaled] = useState(true), [draw, setDraw] = useState(1);
  const w = useMemo(() => { const Q = randMatrix(1, dim, 40 + draw), K = randMatrix(8, dim, 90 + draw); return attention(Q, K, K, { scale: scaled }).weights[0]; }, [dim, scaled, draw]);
  const spread = useMemo(() => DIMS.map((d) => ({ d, raw: attentionSpread(d, 8, false, 5, 30).maxWeight, sc: attentionSpread(d, 8, true, 5, 30).maxWeight })), []);
  const mx = Math.max(...w), ent = entropy(w), sharp = mx > 0.95;
  const cw = 380;
  return <Layout note="One query looks at eight keys. Each bar is how much attention that key gets. Random vectors of larger dimension produce larger dot products, which makes softmax pick one key almost alone unless the scores are divided by the square root of the dimension."
    left={<div style={{ display: "grid", gap: 12 }}>
      <svg viewBox={`0 0 ${cw} 150`} role="img" aria-label="Attention weights over eight keys" style={BOX}>{w.map((v, i) => <g key={i}><rect x={20 + i * 44} y={130 - v * 110} width={34} height={Math.max(1, v * 110)} rx={4} fill={sharp ? R : G} /><text x={37 + i * 44} y={145} fontSize={11} textAnchor="middle" fill="var(--mute)">key {i + 1}</text></g>)}</svg>
      <svg viewBox={`0 0 ${cw} 130`} role="img" aria-label="Largest attention weight by dimension, with and without scaling" style={BOX}>{spread.map((s, i) => <g key={s.d}><rect x={22 + i * 70} y={100 - s.raw * 84} width={26} height={s.raw * 84} fill={R} /><rect x={50 + i * 70} y={100 - s.sc * 84} width={26} height={s.sc * 84} fill={G} /><text x={50 + i * 70} y={118} fontSize={11} textAnchor="middle" fill="var(--mute)">d={s.d}</text></g>)}<text x={8} y={12} fontSize={11} fill={R}>no scaling</text><text x={90} y={12} fontSize={11} fill={G}>divided by √d</text><text x={200} y={12} fontSize={11} fill="var(--mute)">largest weight, many draws</text></svg></div>}
    right={<>
      <Row><Stat k="largest weight" v={pct(mx)} tone={sharp ? "bad" : undefined} /><Stat k="spread (entropy)" v={`${ent.toFixed(2)} of ${Math.log(8).toFixed(2)}`} /></Row>
      <Pick label="Vector length" items={DIMS.map(String)} value={String(dim)} set={(v) => setDim(Number(v))} />
      <Pick label="Scores" items={["divided by √d", "raw dot product"]} value={scaled ? "divided by √d" : "raw dot product"} set={(v) => setScaled(v === "divided by √d")} />
      <Row><Btn onClick={() => setDraw((d) => d + 1)}>New random draw</Btn><Btn kind="break" onClick={() => { setDim(1024); setScaled(false); }}>Break it: long vectors, no scaling</Btn><Btn onClick={() => { setDim(64); setScaled(true); }}>Reset</Btn></Row></>} />;
}

// ---------- 5. Bandits and Q-learning ----------
const MEANS = [0.25, 0.45, 0.6, 0.7];
const POL: { p: Policy; n: string; c: string }[] = [{ p: "greedy", n: "greedy", c: "#7A7F6E" }, { p: "epsilon", n: "epsilon-greedy", c: AMBER }, { p: "ucb", n: "UCB", c: BLUE }, { p: "thompson", n: "Thompson", c: G }];
function BanditLab() {
  const [eps, setEps] = useState(0.1), [steps, setSteps] = useState(600);
  const runs = useMemo(() => POL.map((p) => { const seeds = Array.from({ length: 16 }, (_, i) => i + 1), all = seeds.map((s) => runBandit(MEANS, p.p, steps, s, { eps }).regret), avg = all[0].map((_, t) => all.reduce((s, r) => s + r[t], 0) / all.length); return { ...p, avg }; }), [eps, steps]);
  const thin = runs.map((r) => { const k = Math.ceil(steps / 120), v = r.avg.filter((_, i) => i % k === 0 || i === steps - 1); return { v, c: r.c, n: r.n }; });
  const cliff = useMemo(() => ({ q: learnCliff("q", 800, 3), s: learnCliff("sarsa", 800, 3) }), []);
  const pq = greedyPath(cliff.q.Q), ps = greedyPath(cliff.s.Q), cs = 30;
  const cell = (s: number) => ({ x: 10 + (s % CW) * cs, y: 8 + Math.floor(s / CW) * cs });
  const draw = (p: number[], c: string) => <polyline fill="none" stroke={c} strokeWidth={3} strokeLinejoin="round" points={p.map((s) => `${cell(s).x + cs / 2},${cell(s).y + cs / 2}`).join(" ")} />;
  const greedyStuck = runs[0].avg.at(-1)! > runs[3].avg.at(-1)! * 2;
  return <Layout note="Top: total regret (reward lost against always pulling the best of four arms), averaged over 16 runs. Bottom: the cliff-walking grid. Q-learning learns the shortest route along the cliff edge. SARSA learns a safer, longer route because it counts its own exploring slips."
    left={<div style={{ display: "grid", gap: 12 }}>
      <Mini label="Regret over time for four policies" hi={thin[0].v.length - 1} w={380} h={170} series={thin} />
      <svg viewBox={`0 0 ${CW * cs + 20} ${CH * cs + 16}`} role="img" aria-label="Cliff walking grid with the Q-learning and SARSA routes" style={BOX}>
        {Array.from({ length: CW * CH }, (_, s) => <rect key={s} x={cell(s).x} y={cell(s).y} width={cs - 1} height={cs - 1} rx={3} fill={isCliff(s) ? "#F1C5CD" : s === GOAL ? "#C9D6A8" : s === START ? "#E9E2C8" : "#F7F4EA"} />)}
        {draw(pq, R)}{draw(ps, G)}<text x={cell(START).x + 4} y={cell(START).y + 19} fontSize={11} fill={INK}>S</text><text x={cell(GOAL).x + 8} y={cell(GOAL).y + 19} fontSize={11} fill={INK}>G</text></svg>
      <div className="mute" style={{ fontSize: 12 }}><span style={{ color: R, fontWeight: 700 }}>Q-learning</span> route is {pq.length - 1} steps. <span style={{ color: G, fontWeight: 700 }}>SARSA</span> route is {ps.length - 1} steps. Pink squares are the cliff.</div></div>}
    right={<>
      <Row>{runs.map((r) => <Stat key={r.n} k={`${r.n} regret`} v={r.avg.at(-1)!.toFixed(0)} tone={(r.p === "greedy" || (r.p === "epsilon" && eps === 0)) && greedyStuck ? "bad" : undefined} />)}</Row>
      <Slider label="Epsilon (share of random pulls)" value={eps} min={0} max={0.5} step={0.05} onChange={setEps} show={eps.toFixed(2)} />
      <Slider label="Pulls" value={steps} min={100} max={1000} step={100} onChange={setSteps} show={String(steps)} />
      <Row><Btn kind="break" onClick={() => { setEps(0); setSteps(1000); }}>Break it: epsilon 0, never explore</Btn><Btn onClick={() => { setEps(0.1); setSteps(600); }}>Reset</Btn></Row></>} />;
}

// ---------- 6. Autoencoders ----------
function AutoencoderLab() {
  const [code, setCode] = useState(1), [kind, setKind] = useState<"linear" | "nonlinear">("nonlinear"), [go, setGo] = useState(false);
  const X = useMemo(() => makeArc(100, 0.02, 2), []);
  const mean = useMemo(() => [0, 1].map((j) => X.reduce((s, r) => s + r[j], 0) / X.length), [X]), Xc = useMemo(() => X.map((r) => r.map((v, j) => v - mean[j])), [X, mean]);
  const pca = useMemo(() => { const cov = [[0, 0], [0, 0]]; Xc.forEach((r) => { for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) cov[i][j] += (r[i] * r[j]) / Xc.length; }); const tr = cov[0][0] + cov[1][1], det = cov[0][0] * cov[1][1] - cov[0][1] ** 2, a = tr / 2 + Math.sqrt((tr / 2) ** 2 - det), vx = cov[0][1] ? a - cov[1][1] : 1, vy = cov[0][1] ? cov[0][1] : 0, nrm = Math.hypot(vx, vy); return { err: tr / 2 - Math.sqrt((tr / 2) ** 2 - det), dir: [vx / nrm, vy / nrm] }; }, [Xc]);
  useEffect(() => setGo(true), []);
  const model = useMemo(() => { if (!go) return null; const sizes = kind === "linear" ? [2, code, 2] : [2, 10, code, 10, 2], acts = kind === "linear" ? (["linear", "linear"] as const) : (["tanh", "linear", "tanh", "linear"] as const); return trainAE(Xc, sizes, [...acts], { epochs: kind === "linear" ? 500 : 700, lr: kind === "linear" ? 0.04 : 0.02, seed: 1 }).ae; }, [go, code, kind, Xc]);
  const rec = model ? Xc.map((x) => reconstruct(model, x)) : [], err = model ? reconError(model, Xc) : 0;
  const sx = (v: number) => 20 + ((v + 1.4) / 2.8) * (S - 40), sy = (v: number) => S - 30 - ((v + 0.75) / 1.5) * (S - 60);
  const trivial = code >= 2, codes = model ? Xc.map((x) => encode(model, x)[0]) : [];
  return <Layout note="Grey dots are the data: a curved arc. Rose dots are what the network rebuilds after squeezing each point through the code. Sticks join each dot to its rebuild. The green line is the best straight-line summary (PCA)."
    left={<svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label="Data and autoencoder reconstruction" style={BOX}>
      <line x1={sx(-pca.dir[0] * 1.3)} y1={sy(-pca.dir[1] * 1.3)} x2={sx(pca.dir[0] * 1.3)} y2={sy(pca.dir[1] * 1.3)} stroke={G} strokeWidth={2} strokeDasharray="6 4" />
      {Xc.map((x, i) => rec[i] && <line key={`l${i}`} x1={sx(x[0])} y1={sy(x[1])} x2={sx(rec[i][0])} y2={sy(rec[i][1])} stroke={R} strokeWidth={1} opacity={0.35} />)}
      {Xc.map((x, i) => <circle key={`d${i}`} cx={sx(x[0])} cy={sy(x[1])} r={3.4} fill="#B9BDB0" />)}{rec.map((x, i) => <circle key={`r${i}`} cx={sx(x[0])} cy={sy(x[1])} r={3} fill={R} />)}{!model && <text x={S / 2} y={S / 2} textAnchor="middle" fill="var(--mute)">training…</text>}</svg>}
    right={<>
      <Row><Stat k="rebuild error" v={err.toFixed(4)} /><Stat k="best straight line" v={pca.err.toFixed(4)} /><Stat k="numbers per point" v={`${code} of 2`} tone={trivial ? "bad" : undefined} /></Row>
      <Pick label="Code size" items={["1", "2"]} value={String(code)} set={(v) => setCode(Number(v))} />
      <Pick label="Layers" items={["linear", "nonlinear"]} value={kind} set={(v) => setKind(v as "linear" | "nonlinear")} />
      {trivial && <p style={{ margin: 0, color: R, fontWeight: 700 }}>Near-perfect rebuild, but nothing was compressed: the code is as big as the input.</p>}
      {code === 1 && model && <p style={{ margin: 0 }} className="mute">Code values run from {Math.min(...codes).toFixed(2)} to {Math.max(...codes).toFixed(2)}: one number places each point along the arc.</p>}
      <Row><Btn kind="break" onClick={() => { setCode(2); setKind("nonlinear"); }}>Break it: code as big as the input</Btn><Btn onClick={() => { setCode(1); setKind("nonlinear"); }}>Reset</Btn></Row></>} />;
}

// ---------- 7. Diffusion ----------
const MIX = { w: [0.3, 0.7], mu: [-2, 2], sd: [0.3, 0.3] };
function DiffusionLab() {
  const [T, setT] = useState(100), [strength, setStrength] = useState(1), [t, setTt] = useState(40);
  const data = useMemo(() => sampleMix(MIX, 1500, 1), []), sch = useMemo(() => schedule(T, "linear", strength), [T, strength]);
  const tt = Math.min(t, T);
  const noisedG = useMemo(() => { let a = 12345; const g = () => { a = (a * 1103515245 + 12345) & 0x7fffffff; return a / 0x7fffffff; }; return data.map((x) => qSample(x, tt, sch, Math.sqrt(-2 * Math.log(g() + 1e-9)) * Math.cos(2 * Math.PI * g()))); }, [data, tt, sch]);
  const back = useMemo(() => reverseSample(MIX, sch, 1500, 4).samples, [sch]);
  const share = back.filter((x) => x < 0).length / back.length, left = sch.abar.at(-1)!, broken = left > 0.2;
  return <Layout note="Top: the data, two bumps with 30% of points in the left one. Middle: the same points after t steps of noising. Bottom: points made by starting from pure noise and undoing the noise one step at a time. The undoing uses the exact answer in place of a trained network, so any failure comes from the setup, not from learning."
    left={<div style={{ display: "grid", gap: 8 }}><Hist xs={data} lo={-5} hi={5} label="data" /><Hist xs={noisedG} lo={-5} hi={5} label={`after ${tt} noising steps`} color={AMBER} /><Hist xs={back} lo={-5} hi={5} label="generated by undoing the noise" color={broken ? R : G} /></div>}
    right={<>
      <Row><Stat k="signal left after the last step" v={pct(left)} tone={broken ? "bad" : undefined} /><Stat k="generated share in the left bump" v={`${pct(share)} (true 30%)`} tone={Math.abs(share - 0.3) > 0.08 ? "bad" : undefined} /></Row>
      <Slider label="Noising steps" value={T} min={20} max={200} step={10} onChange={setT} show={String(T)} />
      <Slider label="Total amount of noise" value={strength} min={0.04} max={1} step={0.02} onChange={setStrength} show={`${Math.round(strength * 100)}%`} />
      <Slider label="Look at step" value={tt} min={1} max={T} step={1} onChange={setTt} show={String(tt)} />
      <Row><Btn kind="break" onClick={() => { setStrength(0.04); setT(200); }}>Break it: stop noising too early</Btn><Btn onClick={() => { setStrength(1); setT(100); setTt(40); }}>Reset</Btn></Row></>} />;
}

// ---------- 8. Interpretability ----------
const NAMES = ["tenure", "usage", "age", "noise", "usage copy"];
function ShapLab() {
  const [dup, setDup] = useState(false), [row, setRow] = useState(3);
  const data = useMemo(() => { let a = 99; const g = () => { a = (a * 1103515245 + 12345) & 0x7fffffff; return a / 0x7fffffff; }, n = () => Math.sqrt(-2 * Math.log(g() + 1e-9)) * Math.cos(2 * Math.PI * g()); const X = Array.from({ length: 160 }, () => { const u = n(); return [n(), u, n(), n(), u + 0.02 * n()]; }); return { X, y: X.map((r) => 2 * r[0] + 1.5 * r[1] - r[2] + 0.3 * n()) }; }, []);
  const cols = dup ? [0, 1, 2, 3, 4] : [0, 1, 2, 3], Xs = useMemo(() => data.X.map((r) => cols.map((c) => r[c])), [data, dup]);
  const model = useMemo(() => fitLinear(Xs, data.y, 0.5), [Xs, data]), f = (x: number[]) => predict(model, [x])[0];
  const sh = useMemo(() => shapleyExact(f, Xs[row], Xs.slice(0, 40)), [model, Xs, row]), perm = useMemo(() => permutationImportance(f, Xs, data.y, 2, 5), [model, Xs, data]);
  const total = sh.phi.reduce((a, b) => a + b, 0);
  return <Layout note="A model predicts a score from a few customer features. Left: how much each feature pushed this one customer's prediction up or down, from the average. Right: how much the model's error grows when a feature is shuffled. The last feature is a near copy of usage."
    left={<div style={{ display: "grid", gap: 10 }}><HBars label="Shapley contribution for one row" rows={cols.map((c, i) => ({ n: NAMES[c], v: sh.phi[i] }))} /><HBars label="Permutation importance across all rows" rows={cols.map((c, i) => ({ n: NAMES[c], v: perm[i] }))} tone={AMBER} /></div>}
    right={<>
      <Row><Stat k="average prediction" v={sh.base.toFixed(2)} /><Stat k="this row's prediction" v={sh.value.toFixed(2)} /><Stat k="credits add up to the gap" v={Math.abs(total - (sh.value - sh.base)) < 1e-9 ? "yes" : "no"} /></Row>
      <Slider label="Customer row" value={row} min={0} max={30} step={1} onChange={setRow} show={String(row + 1)} />
      <Pick label="Features" items={["four features", "add a copy of usage"]} value={dup ? "add a copy of usage" : "four features"} set={(v) => setDup(v === "add a copy of usage")} />
      <p className="mute" style={{ margin: 0 }}>Green bars: credit for pushing the prediction up. Rose bars: pushing it down. Top bars use one customer, bottom bars use everyone.</p>
      <Row><Btn kind="break" onClick={() => setDup(true)}>Break it: add a copy of usage</Btn><Btn onClick={() => { setDup(false); setRow(3); }}>Reset</Btn></Row></>} />;
}

// ---------- 9. Fairness ----------
function FairLab() {
  const [baseB, setBaseB] = useState(0.3), [rule, setRule] = useState<"same" | "selection" | "tpr">("same"), [thr, setThr] = useState(0);
  const d = useMemo(() => makeFairness(6000, [0.5, baseB], 3), [baseB]);
  const th = useMemo<[number, number]>(() => {
    if (rule === "same") return [thr, thr];
    const st = groupStats(d, [thr, thr])[0];
    return rule === "selection" ? [thr, thresholdFor(d, 1, "selection", st.selection)] : [thr, thresholdFor(d, 1, "tpr", st.tpr)];
  }, [d, rule, thr]);
  const st = groupStats(d, th), gap = (k: "selection" | "tpr" | "fpr" | "ppv") => Math.abs(st[0][k] - st[1][k]);
  const rowsM: { k: "selection" | "tpr" | "fpr" | "ppv"; n: string }[] = [{ k: "selection", n: "selected" }, { k: "tpr", n: "true positive rate" }, { k: "fpr", n: "false positive rate" }, { k: "ppv", n: "precision" }];
  const W = 380;
  return <Layout note="Two groups, A and B, get scores that are equally informative: the same noise for both. They differ only in how often the true outcome is positive. Pick a rule for the cut-off and see which gaps open."
    left={<svg viewBox={`0 0 ${W} 250`} role="img" aria-label="Rates for group A and group B" style={BOX}>
      {rowsM.map((m, i) => <g key={m.k} transform={`translate(0 ${12 + i * 58})`}><text x={10} y={10} fontSize={12} fill={INK}>{m.n}</text>
        {[0, 1].map((g) => <g key={g}><rect x={10} y={16 + g * 16} width={Math.max(1, st[g][m.k] * 250)} height={13} rx={3} fill={g ? AMBER : G} /><text x={16 + st[g][m.k] * 250} y={27 + g * 16} fontSize={10} fill="var(--mute)" fontFamily="var(--mono)">{pct(st[g][m.k])}</text></g>)}
        <text x={W - 10} y={30} fontSize={12} textAnchor="end" fontWeight={700} fill={gap(m.k) > 0.05 ? R : G}>gap {pct(gap(m.k))}</text></g>)}
      <text x={10} y={246} fontSize={11} fill={G}>group A</text><text x={70} y={246} fontSize={11} fill={AMBER}>group B</text></svg>}
    right={<>
      <Row><Stat k="selected gap" v={pct(gap("selection"))} tone={gap("selection") > 0.05 ? "bad" : undefined} /><Stat k="true positive gap" v={pct(gap("tpr"))} tone={gap("tpr") > 0.05 ? "bad" : undefined} /><Stat k="precision gap" v={pct(gap("ppv"))} tone={gap("ppv") > 0.05 ? "bad" : undefined} /></Row>
      <Pick label="Cut-off rule" items={["same cut-off", "equal selection", "equal true positives"]} value={rule === "same" ? "same cut-off" : rule === "selection" ? "equal selection" : "equal true positives"} set={(v) => setRule(v === "same cut-off" ? "same" : v === "equal selection" ? "selection" : "tpr")} />
      <Slider label="Group A cut-off on the score" value={thr} min={-1.5} max={1.5} step={0.1} onChange={setThr} show={thr.toFixed(1)} />
      <Slider label="Share of group B with a positive outcome" value={baseB} min={0.1} max={0.5} step={0.05} onChange={setBaseB} show={pct(baseB)} />
      <p className="mute" style={{ margin: 0 }}>Group A has 50% positive outcomes. Set group B to 50% as well and every gap closes together.</p>
      <Row><Btn kind="break" onClick={() => { setRule("selection"); setBaseB(0.3); setThr(0); }}>Break it: force equal selection</Btn><Btn onClick={() => { setRule("same"); setBaseB(0.3); setThr(0); }}>Reset</Btn></Row></>} />;
}

const LABS: Record<string, () => React.ReactElement> = { "lab-series": SeriesLab, "lab-hmm": HmmLab, "lab-embed": EmbedLab, "lab-attention": AttentionLab, "lab-bandit": BanditLab, "lab-autoencoder": AutoencoderLab, "lab-diffusion": DiffusionLab, "lab-shap": ShapLab, "lab-fair": FairLab };
export const PART6 = Object.keys(LABS);
export function Widget6({ name }: { name: string }) { const [on, setOn] = useState(false); useEffect(() => setOn(true), []); const L = LABS[name] as (() => React.ReactElement) | undefined; return L && on ? <L key={name} /> : <div style={{ minHeight: 420 }} aria-busy="true" />; }
