import Link from "next/link";
import Shell from "@/components/shell/Shell";
import { TAGLINES } from "@/content/taglines";
import { PARTS, liveInOrder } from "@/lib/lessons";

export const metadata = { title: "Map · Wobble" };
export default function Lessons() {
  return (
    <Shell current="map" progress={{ label: "ALL LESSONS", slugs: liveInOrder() }}>
      <div className="maphead"><span className="kicker">The map</span><h1>Six worlds, 46 lessons</h1><p>From one straight line to diffusion. Go in order, or jump to whatever you are stuck on.</p></div>
      <div className="worldgrid">
        {PARTS.map((p) => <section key={p.id} className="card wcard" aria-labelledby={`w${p.id}`}>
          <header><img src={`/thumbs/${p.lessons[0].slug}.png`} alt="" /><div><span className="kicker">World {p.id}</span><h2 id={`w${p.id}`}>{p.title}</h2><p>{TAGLINES[p.id] ?? p.blurb}</p></div></header>
          <ol>{p.lessons.map((l, i) => <li key={l.slug}><Link href={`/lessons/${l.slug}/`}><small>{p.id}.{i + 1}</small><span>{l.title}</span><i>{l.verb}</i></Link></li>)}</ol>
        </section>)}
      </div>
    </Shell>
  );
}
