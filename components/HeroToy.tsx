"use client";
import Link from "next/link";
import { useMemo, useRef, useState } from "react";
import { fitPolynomial, mse, rng } from "@/lib/ml";

/** The home-page toy: a little handheld with a matcha LCD. Drag the dots, then press B to overfit. */
const SW = 120, SH = 80, PLOT_T = 12, PLOT_B = 74;
const INK = "#22361A", MID = "#4E6E2A", LIT = "#C8D99A", BG = "#B4CC7E";

function start(seed = 7) {
  const r = rng(seed);
  return Array.from({ length: 9 }, (_, i) => { const x = 0.06 + (i / 8) * 0.88 + (r() - 0.5) * 0.04; return [x, Math.sin(2 * Math.PI * x) * 0.7 + (r() - 0.5) * 0.45] as [number, number]; });
}
const px = (x: number) => 4 + x * (SW - 8), py = (y: number) => PLOT_T + ((1 - (y + 1.2) / 2.4)) * (PLOT_B - PLOT_T);

export default function HeroToy({ first }: { first: string }) {
  const [pts, setPts] = useState(start), [deg, setDeg] = useState(3), [drag, setDrag] = useState<number | null>(null), [seed, setSeed] = useState(7), svg = useRef<SVGSVGElement>(null);
  const fit = useMemo(() => fitPolynomial(pts.map((p) => p[0]), pts.map((p) => p[1]), deg, 1e-9), [pts, deg]);
  const err = mse(fit.predict(pts.map((p) => p[0])), pts.map((p) => p[1]));
  const cols = useMemo(() => { const xs = Array.from({ length: SW - 8 }, (_, i) => i / (SW - 9)); return fit.predict(xs).map((y, i) => [4 + i, Math.round(py(y))] as [number, number]); }, [fit]);
  const curve: [number, number][] = [];
  cols.forEach(([x, y], i) => { const prev = i ? cols[i - 1][1] : y, lo = Math.max(PLOT_T, Math.min(y, prev)), hi = Math.min(PLOT_B, Math.max(y, prev)); for (let v = lo; v <= hi; v++) curve.push([x, v]); });
  const wild = deg >= 7, ok = !wild;
  const at = (e: React.PointerEvent) => { const b = svg.current!.getBoundingClientRect(); return [((e.clientX - b.left) / b.width) * SW, ((e.clientY - b.top) / b.height) * SH]; };
  const down = (e: React.PointerEvent) => { const [x, y] = at(e); let best = -1, bd = 64; pts.forEach((p, i) => { const d = (px(p[0]) - x) ** 2 + (py(p[1]) - y) ** 2; if (d < bd) { bd = d; best = i; } }); if (best >= 0) { setDrag(best); (e.target as Element).setPointerCapture?.(e.pointerId); } };
  const move = (e: React.PointerEvent) => { if (drag === null) return; const [, y] = at(e), v = Math.max(-1.15, Math.min(1.15, (1 - (y - PLOT_T) / (PLOT_B - PLOT_T)) * 2.4 - 1.2)); setPts((P) => P.map((p, i) => (i === drag ? [p[0], v] : p))); };
  const nudge = (i: number, d: number) => setPts((P) => P.map((p, k) => (k === i ? [p[0], Math.max(-1.15, Math.min(1.15, p[1] + d))] : p)));
  const text = (s: string, x: number, y: number, c = INK) => <text x={x} y={y} fontFamily="Silkscreen, monospace" fontSize={5.2} fill={c}>{s}</text>;
  return (
    <div className="handheld" aria-label="A little handheld game: drag the dots and press B to break the fit">
      <div className="hh-top"><span className="hh-led" data-on={wild} /> <span className="hh-brand">WOBBLE<b>BOY</b></span></div>
      <div className="hh-bezel">
        <svg ref={svg} viewBox={`0 0 ${SW} ${SH}`} shapeRendering="crispEdges" role="img" aria-label={`Nine dots and a curve of degree ${deg}. ${wild ? "The curve is overfitting: it bends wildly to hit every dot." : "The curve follows the general shape."}`}
          onPointerDown={down} onPointerMove={move} onPointerUp={() => setDrag(null)} onPointerLeave={() => setDrag(null)} style={{ width: "100%", display: "block", touchAction: "none", cursor: drag !== null ? "grabbing" : "grab", background: BG }}>
          {Array.from({ length: SH / 2 }, (_, r) => <rect key={`s${r}`} x={0} y={r * 2 + 1} width={SW} height={0.35} fill={LIT} opacity={0.35} />)}
          {text(`DEG ${String(deg).padStart(2, "0")}`, 4, 8)}{text(`ERR ${err < 1e-4 ? "0.000" : err.toFixed(3)}`, 40, 8)}{text(wild ? "!! OVERFIT" : "OK", wild ? 82 : 104, 8, wild ? "#7A1F3D" : INK)}
          <rect x={0} y={PLOT_T - 2} width={SW} height={0.6} fill={MID} />
          {curve.map(([x, y], i) => <rect key={i} x={x} y={y} width={1} height={1} fill={wild ? "#7A1F3D" : INK} />)}
          {pts.map((p, i) => <g key={i}><rect x={Math.round(px(p[0])) - 2} y={Math.round(py(p[1])) - 2} width={5} height={5} fill={drag === i ? "#7A1F3D" : MID} /><rect x={Math.round(px(p[0])) - 1} y={Math.round(py(p[1])) - 1} width={3} height={3} fill={LIT} /></g>)}
        </svg>
      </div>
      <p className="sr-only">Keyboard: use the D-pad buttons to change the curve degree, or the Select and Start buttons.</p>
      <div className="hh-controls">
        <div className="dpad" role="group" aria-label="Curve degree">
          <button className="dp up" aria-label="Raise a dot" onClick={() => nudge(4, 0.15)} />
          <button className="dp left" aria-label="Lower the degree" onClick={() => setDeg((d) => Math.max(1, d - 1))} />
          <span className="dp mid" />
          <button className="dp right" aria-label="Raise the degree" onClick={() => setDeg((d) => Math.min(8, d + 1))} />
          <button className="dp down" aria-label="Lower a dot" onClick={() => nudge(4, -0.15)} />
        </div>
        <div className="ab">
          <span className="abwrap b"><button aria-label="B: break it, overfit with degree 8" onClick={() => setDeg(8)}>B</button><small>BREAK</small></span>
          <span className="abwrap a"><button aria-label="A: fix it, degree 3" onClick={() => setDeg(3)}>A</button><small>FIX</small></span>
        </div>
      </div>
      <div className="hh-pills">
        <span><button aria-label="Select: new dots" onClick={() => { const s = seed + 1; setSeed(s); setPts(start(s)); }} /><small>NEW DOTS</small></span>
        <span><Link href={`/lessons/${first}/`} aria-label="Start: first lesson" /><small>START</small></span>
      </div>
      <p className="hh-hint">{ok ? "drag a dot. then press B." : "see? it hits every dot and means nothing. press A."}</p>
    </div>
  );
}
