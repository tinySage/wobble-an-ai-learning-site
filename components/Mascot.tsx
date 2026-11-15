"use client";
import { useEffect, useRef, useState } from "react";

/** Wobble, the mascot: a jelly with a matcha sprout. Drawn on a 16 x 16 pixel grid. */
export type Mood = "idle" | "happy" | "sad" | "broken";

const BODY = [
  "........GG......",
  ".......GgG......",
  "......Gg........",
  ".......g........",
  "....oooooooo....",
  "...opplllpppo...",
  "..opllpppppppo..",
  ".oppppppppppppo.",
  ".oppEEppppEEppo.",
  ".oppEEppppEEppo.",
  ".opcppppppppcpo.",
  ".oppppMMMMppppo.",
  ".oppppppppppppo.",
  "..oppppppppppo..",
  "...oooooooooo...",
  "................",
];
const COL: Record<string, string> = { o: "#5A2236", p: "#F4A7B9", l: "#FFD9E1", g: "#58792F", G: "#8FB05A", c: "#E8789A", E: "#26301F", M: "#26301F", w: "#FFFFFF" };
const BROKEN: Record<string, string> = { ...COL, p: "#B9BDB0", l: "#DADDD2", c: "#B9BDB0", G: "#C4416B", g: "#C4416B" };

function cells(mood: Mood, blink: boolean): [number, number, string][] {
  const pal = mood === "broken" ? BROKEN : COL, out: [number, number, string][] = [], ink = COL.E;
  BODY.forEach((row, y) => [...row].forEach((ch, x) => { if (ch !== ".") out.push([x, y, pal[ch === "E" || ch === "M" ? "p" : ch]]); }));
  const put = (pts: number[][], c = ink) => pts.forEach(([x, y]) => out.push([x, y, c]));
  if (mood === "broken") put([[3, 8], [5, 8], [4, 9], [3, 10], [5, 10], [9, 8], [11, 8], [10, 9], [9, 10], [11, 10]]);
  else if (mood === "happy") put([[3, 9], [4, 8], [5, 8], [6, 9], [9, 9], [10, 8], [11, 8], [12, 9]]);
  else if (blink) put([[4, 9], [5, 9], [10, 9], [11, 9]]);
  else { put([[4, 8], [5, 8], [4, 9], [5, 9], [10, 8], [11, 8], [10, 9], [11, 9]]); put([[4, 8], [10, 8]], COL.w); }
  if (mood === "happy") put([[5, 11], [10, 11], [6, 12], [7, 12], [8, 12], [9, 12]]);
  else if (mood === "sad") put([[7, 11], [8, 11], [6, 12], [9, 12]]);
  else if (mood === "broken") put([[5, 12], [6, 11], [7, 12], [8, 11], [9, 12], [10, 11]]);
  else put([[6, 11], [9, 11], [7, 12], [8, 12]]);
  return out;
}

const LINES = ["poke me again, I dare you", "I am 90% jelly, 10% gradients", "error: too wobbly to compute", "ok that tickled", "please do not overfit me", "I contain multitudes (of parameters)"];

export default function Mascot({ size = 96, mood: forced, talk = false, still = false, label = "Wobble the jelly" }: { size?: number; mood?: Mood; talk?: boolean; still?: boolean; label?: string }) {
  const [blink, setBlink] = useState(false), [poked, setPoked] = useState(false), [line, setLine] = useState(0), t = useRef<number | null>(null);
  useEffect(() => { const id = window.setInterval(() => { setBlink(true); window.setTimeout(() => setBlink(false), 140); }, 3200 + Math.random() * 1500); return () => window.clearInterval(id); }, []);
  useEffect(() => () => { if (t.current) window.clearTimeout(t.current); }, []);
  const mood: Mood = poked ? "broken" : forced ?? "idle";
  const poke = () => { setPoked(true); setLine((l) => (l + 1) % LINES.length); if (t.current) window.clearTimeout(t.current); t.current = window.setTimeout(() => setPoked(false), 1100); };
  return (
    <span className="mascot-wrap" style={{ position: "relative", display: "inline-block" }}>
      {talk && <span className="bubble" aria-live="polite">{poked ? LINES[line] : "hi, I'm Wobble. poke me."}</span>}
      {still ? <span className="mascot" role="img" aria-label={label} style={{ width: size, height: size }}><svg viewBox="0 0 16 16" width={size} height={size} shapeRendering="crispEdges" aria-hidden="true">{cells(mood, blink).map(([x, y, c], i) => <rect key={i} x={x} y={y} width={1.02} height={1.02} fill={c} />)}</svg></span> :
      <button type="button" onClick={poke} aria-label={`${label}. Poke to make it glitch.`} className={`mascot ${mood === "broken" ? "glitch" : ""}`} style={{ width: size, height: size }}>
        <svg viewBox="0 0 16 16" width={size} height={size} shapeRendering="crispEdges" aria-hidden="true">
          {cells(mood, blink).map(([x, y, c], i) => <rect key={i} x={x} y={y} width={1.02} height={1.02} fill={c} />)}
        </svg>
      </button>}
    </span>
  );
}
