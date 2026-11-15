"use client";
import { useState } from "react";
import type { LessonContent } from "@/lib/lessons";
import { markDone } from "@/lib/progress";
import Maths from "@/components/Maths";
import * as Ic from "@/components/shell/Icons";

const TABS = ["Overview", "Data", "Maths", "Sources"] as const;
export default function LessonBody({ c }: { c: LessonContent }) {
  const [tab, setTab] = useState<(typeof TABS)[number]>("Overview"), [pick, setPick] = useState<number | null>(null), [copied, setCopied] = useState(false);
  const right = pick !== null && pick === c.check.answer, wrong = pick !== null && !right;
  const lines = c.code.split("\n");
  return (<aside className="panel">
    <div className="tabs" role="tablist" aria-label="Lesson sections" onKeyDown={(e) => { const i = TABS.indexOf(tab), d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0; if (d) { e.preventDefault(); const n = TABS[(i + d + TABS.length) % TABS.length]; setTab(n); (e.currentTarget.querySelector(`[data-tab="${n}"]`) as HTMLElement | null)?.focus(); } }}>
      {TABS.map((t) => <button key={t} role="tab" data-tab={t} tabIndex={tab === t ? 0 : -1} aria-selected={tab === t} onClick={() => setTab(t)}>{t}</button>)}
    </div>
    <div className="card info prose" role="tabpanel">
      {tab === "Overview" && <><h2>{c.headline}</h2>{c.overview.map((t, i) => <p key={i}>{t}</p>)}</>}
      {tab === "Data" && <><h2>The data</h2><p>{c.data}</p></>}
      {tab === "Maths" && <><h2>The maths</h2>{c.maths.map((t, i) => <Maths key={i} line={t} />)}</>}
      {tab === "Sources" && <><h2>Sources</h2><p className="mute">Checked against more than one source. Where they disagree, the lesson says so.</p><ul className="sources">{c.sources.map((s) => <li key={s.url}><a href={s.url}>{s.title}</a><span>{s.note}</span></li>)}</ul></>}
    </div>
    <div className="card code-card">
      <div className="code-head"><b>Try it yourself</b><button className="ibtn" onClick={() => { navigator.clipboard?.writeText(c.code).then(() => { setCopied(true); window.setTimeout(() => setCopied(false), 1400); }).catch(() => {}); }} aria-label={copied ? "Copied" : "Copy code"}>{copied ? <Ic.Check size={18} /> : <Ic.Copy size={18} />}</button></div>
      <pre className="code">{lines.map((l, i) => <span key={i}><span className="ln">{i + 1}</span>{l}{"\n"}</span>)}</pre>
    </div>
    <section className="card quiz" aria-label="Check yourself">
      <h2>Check yourself</h2><p className="q">{c.check.question}</p>
      <div className="answers">{c.check.options.map((o, i) => <button key={o} onClick={() => { setPick(i); if (i === c.check.answer) markDone(c.slug); }} className={`answer ${pick === i ? (i === c.check.answer ? "yes" : "no") : ""}`} aria-pressed={pick === i}><span className="key">{"ABC"[i]}</span>{o}</button>)}</div>
      <p aria-live="polite" className={`verdict ${right ? "yes" : wrong ? "no" : ""}`}>{right ? "Correct. Lesson ticked off." : wrong ? "Not quite. Try another." : "Answer to tick this lesson off."}</p>
    </section>
    <section className="card real" aria-label="In real life"><span><Ic.Sparkle /></span><div><b>In real life</b><p>{c.realLife}</p></div></section>
  </aside>);
}
