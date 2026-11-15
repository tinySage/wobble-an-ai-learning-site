"use client";
import dynamic from "next/dynamic";
import { useState } from "react";
import * as Ic from "@/components/shell/Icons";

const LossWorld = dynamic(() => import("@/components/three/LossWorld"), { ssr: false, loading: () => <div className="stage3d loading"><span>Loading the 3D world…</span></div> });
const PRESETS: Record<string, { label: string; lr: number; kind?: "break" }[]> = {
  "gradient-descent": [{ label: "Small steps 0.05", lr: 0.05 }, { label: "Good 0.12", lr: 0.12 }, { label: "Bouncing 1.2", lr: 1.2 }, { label: "Break it: 4.5", lr: 4.5, kind: "break" }],
  "learning-rate": [{ label: "Too small 0.02", lr: 0.02 }, { label: "About right 0.15", lr: 0.15 }, { label: "Bouncing 1.2", lr: 1.2 }, { label: "Break it: 4.5", lr: 4.5, kind: "break" }],
  optimisers: [{ label: "Plain SGD 0.12", lr: 0.12 }, { label: "Bigger steps 0.4", lr: 0.4 }, { label: "Bouncing 2", lr: 2 }, { label: "Break it: 4.5", lr: 4.5, kind: "break" }],
};

/** The same 3D loss landscape as the home page, with learning-rate presets for this lesson. */
export default function Lesson3D({ slug }: { slug: string }) {
  const P = PRESETS[slug] ?? PRESETS["gradient-descent"], [lr, setLr] = useState(P[1]?.lr ?? P[0].lr), [running, setRunning] = useState(true), [seed, setSeed] = useState(0), [st, setSt] = useState({ step: 0, loss: 0, state: "rolling" });
  return (<section className="card lesson3d" aria-label="3D view">
    <div className="l3-head"><b><Ic.Cube size={20} /> The same idea in 3D</b><div className="chips">{P.map((p) => <button key={p.label} className="px-chip" aria-pressed={lr === p.lr} onClick={() => { setLr(p.lr); setSeed((s) => s + 1); setRunning(true); }}>{p.label}</button>)}</div></div>
    <LossWorld lr={lr} running={running} seed={seed} setRunning={setRunning} onStatus={setSt} />
    <div className="l3-out"><div className="lcd"><div className="lcd-k">learning rate</div><div className="lcd-v">{lr}</div></div><div className="lcd"><div className="lcd-k">step</div><div className="lcd-v">{st.step}</div></div><div className={`lcd ${st.state === "flew off" ? "bad" : ""}`}><div className="lcd-k">loss</div><div className="lcd-v">{Number.isFinite(st.loss) ? st.loss.toFixed(3) : "∞"}</div></div><div className={`lcd ${st.state === "flew off" ? "bad" : ""}`}><div className="lcd-k">state</div><div className="lcd-v">{st.state === "flew off" ? "flew off" : st.state}</div></div></div>
  </section>);
}
