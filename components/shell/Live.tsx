"use client";
import { useEffect, useState } from "react";
import { useDone } from "@/components/Progress";
import * as Ic from "./Icons";

export function Clock() {
  const [t, setT] = useState<Date | null>(null);
  useEffect(() => { setT(new Date()); const id = window.setInterval(() => setT(new Date()), 1000); return () => window.clearInterval(id); }, []);
  const d = t ? t.toLocaleDateString("en-US", { month: "short", day: "2-digit", year: "numeric" }).toUpperCase() : "—", h = t ? t.toLocaleTimeString("en-GB", { hour12: false }) : "--:--:--";
  return <div className="clock" aria-hidden="true"><span>{d}</span><span>{h}</span></div>;
}
export function ProgressCard({ label, slugs }: { label: string; slugs: string[] }) {
  const done = useDone().filter((s) => slugs.includes(s)).length, pct = slugs.length ? (done / slugs.length) * 100 : 0;
  return (<div className="prog">
    <Ic.Book size={30} />
    <div><b>{label}</b><div className="prog-row"><span className="prog-bar" role="progressbar" aria-label={`${label} progress`} aria-valuemin={0} aria-valuemax={slugs.length} aria-valuenow={done}><i style={{ width: `${pct}%` }} /></span><span className="prog-n">{done}/{slugs.length}</span></div></div>
  </div>);
}
