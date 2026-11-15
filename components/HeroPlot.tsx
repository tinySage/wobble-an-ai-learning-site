import { demo } from "@/lib/demo";

export default function HeroPlot() {
  const d = demo(), W = 700, H = 420, X = (x: number) => 30 + x * (W - 60), Y = (y: number) => H / 2 - y * 150;
  const path = "M" + d.curve.map((p) => `${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join(" L");
  return (
    <div className="card" style={{ padding: 8 }}>
      <svg role="img" aria-label="Fourteen data points with a smooth fitted curve and error lines to each point" viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }}>
        <rect x="1" y="1" width={W - 2} height={H - 2} rx="16" fill="#FFF9F7" />
        {d.xs.map((x, i) => <line key={`e${i}`} x1={X(x)} x2={X(x)} y1={Y(d.ys[i])} y2={Y(d.fit(x))} stroke="#C4416B" strokeWidth="2.2" strokeLinecap="round" />)}
        <path d={path} fill="none" stroke="rgba(88,121,47,.2)" strokeWidth="16" strokeLinecap="round" />
        <path d={path} fill="none" stroke="#58792F" strokeWidth="4.5" strokeLinecap="round" strokeLinejoin="round" />
        {d.xs.map((x, i) => <g key={i}><circle cx={X(x)} cy={Y(d.ys[i])} r="11" fill="rgba(38,48,31,.1)" /><circle cx={X(x)} cy={Y(d.ys[i])} r="6" fill="#26301F" /></g>)}
      </svg>
    </div>
  );
}
