"use client";
import { useState } from "react";
import type { LessonContent } from "@/lib/lessons";
import { markDone } from "@/lib/progress";
import Maths from "@/components/Maths";

const TABS = ["Overview", "Data", "The maths", "Code", "Sources"] as const;
export default function LessonBody({ c }: { c: LessonContent }) {
  const [tab, setTab] = useState<(typeof TABS)[number]>("Overview"), [pick, setPick] = useState<number | null>(null);
  return (<>
    <section className="card" style={{ padding: "18px 26px 24px" }} aria-label="Lesson notes">
      <div role="tablist" aria-label="Lesson sections" onKeyDown={(e) => { const i = TABS.indexOf(tab), d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0; if (d) { e.preventDefault(); const n = TABS[(i + d + TABS.length) % TABS.length]; setTab(n); (e.currentTarget.querySelector(`[data-tab="${n}"]`) as HTMLElement | null)?.focus(); } }} style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 14 }}>
        {TABS.map((t) => <button key={t} role="tab" data-tab={t} tabIndex={tab === t ? 0 : -1} aria-selected={tab === t} onClick={() => setTab(t)} className="btn" style={{ minHeight: 44, padding: "0 18px", background: tab === t ? "var(--matcha)" : "transparent", color: tab === t ? "#fff" : "var(--ink)", borderColor: "var(--line)" }}>{t}</button>)}
      </div>
      <div role="tabpanel">
        {tab === "Overview" && <><h2 style={{ fontSize: 32, marginBottom: 10 }}>{c.headline}</h2>{c.overview.map((t, i) => <p key={i} style={{ margin: "0 0 10px" }}>{t}</p>)}</>}
        {tab === "Data" && <p style={{ margin: 0 }}>{c.data}</p>}
        {tab === "The maths" && <>{c.maths.map((t, i) => <Maths key={i} line={t} />)}</>}
        {tab === "Code" && <pre className="mono" style={{ margin: 0, fontSize: 13, overflowX: "auto", background: "var(--blush)", padding: 16, borderRadius: 12 }}>{c.code}</pre>}
        {tab === "Sources" && <><p className="mute" style={{ marginTop: 0 }}>Checked against more than one source before writing. Where they differ, it is noted.</p><ul style={{ paddingLeft: 18, margin: 0 }}>{c.sources.map((s) => <li key={s.url} style={{ marginBottom: 6 }}><a href={s.url} style={{ textDecoration: "underline" }}>{s.title}</a> <span className="mute">· {s.note}</span></li>)}</ul></>}
      </div>
    </section>
    <section className="card" style={{ padding: "20px 26px" }} aria-label="Break it">
      <h2 style={{ fontSize: 26, marginBottom: 6 }}>{c.break.label}</h2><p style={{ margin: 0 }} className="mute">{c.break.explains}</p>
    </section>
    <section className="card" style={{ padding: "20px 26px" }} aria-label="Check yourself">
      <h2 style={{ fontSize: 26, marginBottom: 8 }}>Check yourself</h2><p style={{ margin: "0 0 10px" }}>{c.check.question}</p>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {c.check.options.map((o, i) => <button key={o} onClick={() => { setPick(i); if (i === c.check.answer) markDone(c.slug); }} className="btn ghost" style={{ justifyContent: "flex-start", minHeight: 48, borderColor: pick === i ? (i === c.check.answer ? "var(--matcha)" : "var(--rose)") : undefined, borderWidth: pick === i ? 2 : 1 }}>{o}</button>)}
      </div>
      <p aria-live="polite" style={{ margin: "10px 0 0", fontWeight: 700, color: pick === null ? undefined : pick === c.check.answer ? "var(--matcha)" : "var(--rose)" }}>{pick === null ? "" : pick === c.check.answer ? "Right." : "Not quite. Try another."}</p>
    </section>
    <p className="mute" style={{ margin: 0 }}><strong style={{ color: "var(--ink)" }}>In real life.</strong> {c.realLife}</p>
  </>);
}
