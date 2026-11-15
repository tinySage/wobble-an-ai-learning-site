"use client";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { useDone } from "@/components/Progress";
import * as Ic from "./Icons";

export type StripItem = { slug: string; title: string; code: string; world: number; worldTitle: string };
export default function LessonStrip({ items, current, worlds, title = "Lessons" }: { items: StripItem[]; current?: string; worlds?: { id: number; title: string }[]; title?: string }) {
  const done = useDone(), [w, setW] = useState<number>(() => items.find((i) => i.slug === current)?.world ?? 1), row = useRef<HTMLOListElement>(null);
  const shown = worlds ? items.filter((i) => i.world === w) : items;
  useEffect(() => { const el = row.current?.querySelector<HTMLElement>("[aria-current=page]"); if (el && row.current) row.current.scrollLeft = el.offsetLeft - 40; }, [w]);
  const scroll = (d: number) => row.current?.scrollBy({ left: d * 420, behavior: "smooth" });
  return (<section className="strip" aria-label={title}>
    <div className="strip-head">
      <h2>{title}</h2>
      {worlds && <div className="worldtabs" role="tablist" aria-label="Worlds">{worlds.map((x) => <button key={x.id} role="tab" aria-selected={w === x.id} onClick={() => setW(x.id)}><span>{x.id}</span>{x.title}</button>)}</div>}
    </div>
    <div className="strip-body">
      <ol ref={row}>
        {shown.map((l) => <li key={l.slug}><Link href={`/lessons/${l.slug}/`} className={`lcard ${done.includes(l.slug) ? "done" : ""}`} aria-current={l.slug === current ? "page" : undefined}>
          <img src={`/thumbs/${l.slug}.png`} alt="" width={120} height={90} loading="lazy" />
          <span className="lcard-t"><small>{l.code}</small>{l.title}</span>
          {done.includes(l.slug) ? <span className="lcard-ok" aria-label="finished"><Ic.Check size={16} /></span> : <Ic.Chevron size={18} />}
        </Link></li>)}
      </ol>
      <button className="strip-next" onClick={() => scroll(1)} aria-label="Scroll lessons right"><Ic.Chevron /></button>
    </div>
  </section>);
}
