"use client";
import { useState } from "react";
import type { LessonContent } from "@/lib/lessons";
import { markDone } from "@/lib/progress";
import Maths from "@/components/Maths";
import Mascot from "@/components/Mascot";

const TABS = ["Overview", "Data", "Maths", "Code", "Sources"] as const;
export default function LessonBody({ c }: { c: LessonContent }) {
  const [tab, setTab] = useState<(typeof TABS)[number]>("Overview"), [pick, setPick] = useState<number | null>(null), [tries, setTries] = useState(0);
  const right = pick !== null && pick === c.check.answer, wrong = pick !== null && !right;
  return (<div className="lesson-grid">
    <section className="window notes" aria-label="Lesson notes">
      <div role="tablist" aria-label="Lesson sections" className="tabs" onKeyDown={(e) => { const i = TABS.indexOf(tab), d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0; if (d) { e.preventDefault(); const n = TABS[(i + d + TABS.length) % TABS.length]; setTab(n); (e.currentTarget.querySelector(`[data-tab="${n}"]`) as HTMLElement | null)?.focus(); } }}>
        {TABS.map((t) => <button key={t} role="tab" data-tab={t} tabIndex={tab === t ? 0 : -1} aria-selected={tab === t} onClick={() => setTab(t)}>{t}</button>)}
      </div>
      <div role="tabpanel" className="win-body prose">
        {tab === "Overview" && <><h2>{c.headline}</h2>{c.overview.map((t, i) => <p key={i}>{t}</p>)}</>}
        {tab === "Data" && <><h2>What you are playing with</h2><p>{c.data}</p></>}
        {tab === "Maths" && <>{c.maths.map((t, i) => <Maths key={i} line={t} />)}</>}
        {tab === "Code" && <><h2>Same idea, real library</h2><pre className="code">{c.code}</pre></>}
        {tab === "Sources" && <><h2>Where this comes from</h2><p className="mute">Checked against more than one source before writing. Where they disagree, the lesson says so.</p><ul className="sources">{c.sources.map((s) => <li key={s.url}><a href={s.url}>{s.title}</a><span>{s.note}</span></li>)}</ul></>}
      </div>
    </section>

    <aside className="side-col">
      <section className="hazard" aria-label="What the red button does">
        <span className="kicker">WHAT THE RED BUTTON DOES</span>
        <p>{c.break.explains}</p>
      </section>

      <section className={`window quiz ${right ? "clear" : ""}`} aria-label="Check yourself">
        <div className="win-bar"><span className="dots"><i /><i /><i /></span><span>BOSS QUESTION</span></div>
        <div className="win-body">
          <div className="quiz-q"><Mascot size={52} mood={right ? "happy" : wrong ? "sad" : "idle"} still label="Wobble, waiting for your answer" /><p>{c.check.question}</p></div>
          <div className="answers">
            {c.check.options.map((o, i) => <button key={o} onClick={() => { setPick(i); setTries((t) => t + 1); if (i === c.check.answer) markDone(c.slug); }} className={`answer ${pick === i ? (i === c.check.answer ? "yes" : "no") : ""}`} aria-pressed={pick === i}><span className="key">{"ABC"[i]}</span>{o}</button>)}
          </div>
          <p aria-live="polite" className={`verdict ${right ? "yes" : wrong ? "no" : ""}`}>{right ? (tries === 1 ? "★ LEVEL CLEAR! First try." : "★ LEVEL CLEAR!") : wrong ? "Nope. Wobble is disappointed. Try another." : "Answer to earn this level's star."}</p>
        </div>
      </section>

      <section className="sticky-note" aria-label="In real life"><span className="kicker">IN REAL LIFE</span><p>{c.realLife}</p></section>
    </aside>
  </div>);
}
