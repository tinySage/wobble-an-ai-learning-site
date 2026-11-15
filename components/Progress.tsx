"use client";
import { useEffect, useState } from "react";
import { readDone } from "@/lib/progress";

export function useDone(): string[] {
  const [done, setDone] = useState<string[]>([]);
  useEffect(() => { const f = () => setDone([...readDone()]); f(); window.addEventListener("wobble-progress", f); window.addEventListener("storage", f); return () => { window.removeEventListener("wobble-progress", f); window.removeEventListener("storage", f); }; }, []);
  return done;
}
/** Small star shown next to a cleared level. */
export function DoneMark({ slug }: { slug: string }) {
  const done = useDone().includes(slug);
  return done ? <span aria-label="cleared" title="Cleared" className="mini-star">★</span> : null;
}
/** A chunky pixel XP bar: one block per level. */
export function Tracker({ slugs }: { slugs: string[] }) {
  const d = useDone(), done = d.filter((s) => slugs.includes(s)).length;
  return (<div className="xp">
    <span className="xp-label" aria-live="polite">★ {done}/{slugs.length}</span>
    <div className="xp-bar" role="progressbar" aria-label="Levels cleared" aria-valuemin={0} aria-valuemax={slugs.length} aria-valuenow={done}>
      {slugs.map((s) => <i key={s} className={d.includes(s) ? "on" : ""} />)}
    </div>
  </div>);
}
