"use client";
import { useEffect, useState } from "react";
import { readDone } from "@/lib/progress";

export function useDone(): string[] {
  const [done, setDone] = useState<string[]>([]);
  useEffect(() => { const f = () => setDone([...readDone()]); f(); window.addEventListener("wobble-progress", f); window.addEventListener("storage", f); return () => { window.removeEventListener("wobble-progress", f); window.removeEventListener("storage", f); }; }, []);
  return done;
}
/** Small tick shown next to a finished lesson. */
export function DoneMark({ slug }: { slug: string }) {
  const done = useDone().includes(slug);
  return done ? <span aria-label="finished" title="Finished" style={{ color: "var(--matcha)", fontWeight: 800 }}>✓</span> : null;
}
export function Tracker({ slugs }: { slugs: string[] }) {
  const done = useDone().filter((s) => slugs.includes(s)).length, pct = slugs.length ? (done / slugs.length) * 100 : 0;
  return (<div className="card" style={{ padding: "14px 20px", display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
    <span className="mono" style={{ fontSize: 14 }} aria-live="polite">{done} / {slugs.length} lessons finished</span>
    <div role="progressbar" aria-label="Lessons finished" aria-valuemin={0} aria-valuemax={slugs.length} aria-valuenow={done} style={{ flex: 1, minWidth: 160, height: 10, borderRadius: 99, background: "var(--soft)", overflow: "hidden" }}>
      <div style={{ width: `${pct}%`, height: "100%", background: "var(--matcha)", transition: "width .3s ease" }} /></div>
    <span className="mute" style={{ fontSize: 12 }}>Saved in this browser only</span></div>);
}
