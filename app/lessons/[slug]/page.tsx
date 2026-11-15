import Link from "next/link";
import { notFound } from "next/navigation";
import WidgetRouter from "@/components/WidgetRouter";
import LessonBody from "@/components/LessonBody";
import { DoneMark } from "@/components/Progress";
import Chrome from "@/components/Chrome";
import { PARTS, isLive, lessonMeta, liveSlugs, loadLesson } from "@/lib/lessons";

export const dynamicParams = false;
export const generateStaticParams = () => liveSlugs().map((slug) => ({ slug }));

export default async function LessonPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const c = loadLesson(slug), m = lessonMeta(slug);
  if (!c || !m) notFound();
  const part = m.part;
  return (
    <Chrome current="lessons">
      <div className="lesson">
        <nav className="card side" aria-label={`Part ${part.id}: ${part.title}`}>
          <span className="eyebrow mute" style={{ padding: "4px 10px 8px" }}>Part {part.id} · {part.title}</span>
          {part.lessons.map((l, i) => isLive(l.slug)
            ? <Link key={l.slug} href={`/lessons/${l.slug}/`} aria-current={l.slug === slug ? "page" : undefined}><span>{l.title}</span><span className="mono" style={{ fontSize: 12 }}><DoneMark slug={l.slug} /> {i + 1}</span></Link>
            : <span key={l.slug} className="off">{l.title} · soon</span>)}
        </nav>
        <article style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <header>
            <span className="eyebrow mute">Lesson {m.index} of {m.total} · {m.verb}</span>
            <h1 style={{ fontSize: "clamp(44px,6vw,64px)", marginTop: 6 }}>{m.title}</h1>
          </header>
          <div className="card" style={{ padding: 18 }}><WidgetRouter name={c.widget} /></div>
          <LessonBody c={c} />
          <nav aria-label="Previous and next lesson" style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
            {m.prev && isLive(m.prev.slug) ? <Link href={`/lessons/${m.prev.slug}/`} className="btn ghost">← {m.prev.title}</Link> : <span />}
            {m.next && isLive(m.next.slug) ? <Link href={`/lessons/${m.next.slug}/`} className="btn primary">{m.next.title} →</Link> : <span />}
          </nav>
        </article>
      </div>
    </Chrome>
  );
}
