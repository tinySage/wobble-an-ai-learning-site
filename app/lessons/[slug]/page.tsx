import Link from "next/link";
import { notFound } from "next/navigation";
import WidgetRouter from "@/components/WidgetRouter";
import LessonBody from "@/components/LessonBody";
import Shell from "@/components/shell/Shell";
import LessonStrip from "@/components/shell/LessonStrip";
import * as Ic from "@/components/shell/Icons";
import { PARTS, isLive, lessonMeta, liveSlugs, loadLesson } from "@/lib/lessons";

export const dynamicParams = false;
export const generateStaticParams = () => liveSlugs().map((slug) => ({ slug }));
export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params, m = lessonMeta(slug), c = loadLesson(slug);
  return { title: m ? `${m.title} · Wobble` : "Wobble", description: c?.summary };
}

export default async function LessonPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const c = loadLesson(slug), m = lessonMeta(slug);
  if (!c || !m) notFound();
  const part = m.part, nextPart = PARTS.find((p) => p.id === part.id + 1), next = m.next ?? nextPart?.lessons[0] ?? null;
  const items = part.lessons.map((l, i) => ({ slug: l.slug, title: l.title, code: `${part.id}.${i + 1}`, world: part.id, worldTitle: part.title }));
  return (
    <Shell current="map" progress={{ label: `WORLD ${part.id} · ${part.title.toUpperCase()}`, slugs: part.lessons.map((l) => l.slug) }}>
      <div className="lhead">
        <div>
          <nav className="crumbs" aria-label="Breadcrumb"><Link href="/lessons/">Map</Link><span>/</span><span>World {part.id} · {part.title}</span><span>/</span><span>{part.id}.{m.index}</span></nav>
          <h1>{m.title}</h1>
          <p>{c.summary}</p>
        </div>
        <div className="lhead-r"><img src={`/thumbs/${slug}.png`} alt="" width={150} height={112} /><span className="verb"><i />{m.verb}</span></div>
      </div>
      <div className="lesson">
        <section className="card play" aria-label="Playground">
          <div className="play-head"><b>Playground</b><span>Drag · slide · then press the red button</span></div>
          <WidgetRouter name={c.widget} />
          <div className="breakbox"><span className="bi"><Ic.Bolt size={20} /></span><div><b>What the red button does</b><p>{c.break.explains}</p></div></div>
        </section>
        <LessonBody c={c} />
      </div>
      <nav className="pager" aria-label="Previous and next lesson">
        {m.prev && isLive(m.prev.slug) ? <Link href={`/lessons/${m.prev.slug}/`} className="btn">← {m.prev.title}</Link> : <Link href="/lessons/" className="btn">← Map</Link>}
        {next && isLive(next.slug) ? <Link href={`/lessons/${next.slug}/`} className="btn primary">{m.next ? "Next" : `World ${part.id + 1}`}: {next.title} →</Link> : <Link href="/lessons/" className="btn primary">All done · back to the map</Link>}
      </nav>
      <LessonStrip items={items} current={slug} title={`World ${part.id} · ${part.title}`} />
    </Shell>
  );
}
