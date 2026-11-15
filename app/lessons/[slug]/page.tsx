import Link from "next/link";
import { notFound } from "next/navigation";
import WidgetRouter from "@/components/WidgetRouter";
import LessonBody from "@/components/LessonBody";
import { DoneMark } from "@/components/Progress";
import Chrome from "@/components/Chrome";
import { WorldIcon } from "@/components/WorldMap";
import { PARTS, isLive, lessonMeta, liveSlugs, loadLesson } from "@/lib/lessons";

export const dynamicParams = false;
export const generateStaticParams = () => liveSlugs().map((slug) => ({ slug }));
export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params, m = lessonMeta(slug), c = loadLesson(slug);
  return { title: m ? `${m.part.id}-${m.index} ${m.title} · Wobble` : "Wobble", description: c?.summary };
}

export default async function LessonPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const c = loadLesson(slug), m = lessonMeta(slug);
  if (!c || !m) notFound();
  const part = m.part, nextPart = PARTS.find((p) => p.id === part.id + 1);
  const next = m.next ?? nextPart?.lessons[0] ?? null;
  return (
    <Chrome current="lessons">
      <nav className="levelbar" aria-label={`World ${part.id}: ${part.title}`}>
        <Link href="/lessons/" className="crumb">◂ MAP</Link>
        <span className="lb-world"><WorldIcon id={part.id} size={22} /> WORLD {part.id}</span>
        <ol>
          {part.lessons.map((l, i) => isLive(l.slug)
            ? <li key={l.slug}><Link href={`/lessons/${l.slug}/`} aria-current={l.slug === slug ? "page" : undefined} title={l.title} aria-label={`${part.id}-${i + 1} ${l.title}`}>{i + 1}<DoneMark slug={l.slug} /></Link></li>
            : <li key={l.slug}><span>{i + 1}</span></li>)}
        </ol>
      </nav>

      <header className="level-head">
        <div className="lvl-badge" aria-hidden="true"><small>LEVEL</small>{part.id}-{m.index}</div>
        <div>
          <span className="kicker">{part.title} · {m.verb}</span>
          <h1>{m.title}</h1>
          <p className="lede">{c.summary}</p>
        </div>
      </header>

      <section className="window play" aria-label="Playground">
        <div className="win-bar"><span className="dots"><i /><i /><i /></span><span>PLAYGROUND</span><span className="win-hint">drag things · then press the red button</span></div>
        <div className="win-body"><WidgetRouter name={c.widget} /></div>
      </section>

      <LessonBody c={c} />

      <nav className="next-nav" aria-label="Previous and next level">
        {m.prev && isLive(m.prev.slug) ? <Link href={`/lessons/${m.prev.slug}/`} className="px-btn">◂ {m.prev.title}</Link> : <Link href="/lessons/" className="px-btn">◂ Map</Link>}
        {next && isLive(next.slug) ? <Link href={`/lessons/${next.slug}/`} className="px-btn primary">{m.next ? "Next" : `World ${part.id + 1}`}: {next.title} ▸</Link> : <Link href="/lessons/" className="px-btn primary">You finished the game ▸</Link>}
      </nav>
    </Chrome>
  );
}
