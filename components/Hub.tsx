"use client";
import dynamic from "next/dynamic";
import Link from "next/link";
import { useState } from "react";
import * as Ic from "@/components/shell/Icons";

const LossWorld = dynamic(() => import("@/components/three/LossWorld"), { ssr: false, loading: () => <div className="stage3d loading"><span>Loading the 3D world…</span></div> });
const TABS = ["Overview", "How it works", "Worlds"] as const;

export default function Hub({ worlds, first }: { worlds: { id: number; title: string; blurb: string; n: number; first: string }[]; first: string }) {
  const [lrText, setLrText] = useState("0.12"), [lr, setLr] = useState(0.12), [running, setRunning] = useState(true), [seed, setSeed] = useState(0), [st, setSt] = useState({ step: 0, loss: 0, state: "rolling" }), [tab, setTab] = useState<(typeof TABS)[number]>("Overview");
  const run = (v = lrText) => { const n = Math.max(0.01, Math.min(6, parseFloat(v) || 0.12)); setLrText(String(n)); setLr(n); setSeed((s) => s + 1); setRunning(true); };
  const broke = st.state === "flew off";
  return (<div className="hub">
    <LossWorld lr={lr} running={running} seed={seed} setRunning={setRunning} onStatus={setSt} />
    <aside className="panel">
      <div className="tabs" role="tablist" aria-label="About this page">{TABS.map((t) => <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}>{t}</button>)}</div>
      <div className="card info" role="tabpanel">
        {tab === "Overview" && <><h1>Gradient descent</h1>
          <p>Almost every model you will meet learns the same way: measure the error, find which way is downhill, take a step. The 3D world on the left is that idea, made solid.</p>
          <ul className="feats">
            <li><span><Ic.Bolt /></span>46 small interactive lessons</li>
            <li><span><Ic.Shield /></span>Maths tested against scikit-learn</li>
            <li><span><Ic.Book /></span>Sources and disagreements for every claim</li>
          </ul></>}
        {tab === "How it works" && <><h1>Make it, break it</h1>
          <ol className="steps"><li><b>Make it.</b> Drag, slide and train a tiny model live in your browser.</li><li><b>Break it.</b> Press the red button. The model fails on purpose.</li><li><b>Get why.</b> Read the maths, the code and where sources disagree.</li><li><b>Check it.</b> One question per lesson. Get it right to tick it off.</li></ol></>}
        {tab === "Worlds" && <><h1>Six worlds</h1><ul className="wlist">{worlds.map((w) => <li key={w.id}><Link href={`/lessons/${w.first}/`}><span className="wn">{w.id}</span><span><b>{w.title}</b><small>{w.blurb}</small></span><span className="wc">{w.n}</span></Link></li>)}</ul></>}
      </div>
      <div className="card code-card">
        <div className="code-head"><b>Try it yourself</b><div className="code-btns"><button className="ibtn" onClick={() => { setSeed((s) => s + 1); setRunning(true); }} aria-label="Restart"><Ic.Rotl size={18} /></button><button className="ibtn run" onClick={() => run()} aria-label="Run with this learning rate"><Ic.Run size={16} /></button></div></div>
        <pre className="code"><span className="ln">1</span><span className="k">import</span> numpy <span className="k">as</span> np{"\n"}<span className="ln">2</span>{"\n"}<span className="ln">3</span>lr = <input aria-label="Learning rate" className="code-in" value={lrText} onChange={(e) => setLrText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && run()} />  <span className="c"># try 1.2, then 4.5</span>{"\n"}<span className="ln">4</span>w = np.array([-2.45, 2.25]){"\n"}<span className="ln">5</span><span className="k">for</span> step <span className="k">in</span> range(<span className="n">160</span>):{"\n"}<span className="ln">6</span>    w = w - lr * <span className="f">grad</span>(w){"\n"}</pre>
        <div className={`code-out ${broke ? "bad" : ""}`} aria-live="polite"><span>step {st.step}</span><span>loss {Number.isFinite(st.loss) ? st.loss.toFixed(3) : "∞"}</span><span className="state">{broke ? "flew off the map" : st.state}</span></div>
        <div className="code-actions"><button className="btn break" onClick={() => { setLrText("4.5"); run("4.5"); }}>Break it: lr = 4.5</button><button className="btn" onClick={() => { setLrText("0.12"); run("0.12"); }}>Fix it</button><Link href={`/lessons/${first}/`} className="btn primary">Start lesson 1 <Ic.Chevron size={16} /></Link></div>
      </div>
    </aside>
  </div>);
}
