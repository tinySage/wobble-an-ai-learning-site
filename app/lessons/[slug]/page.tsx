import Link from "next/link";
import { notFound } from "next/navigation";
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
            ? <Link key={l.slug} href={`/lessons/${l.slug}/`} aria-current={l.slug === slug ? "page" : undefined}><span>{l.title}</span><span className="mono" style={{ fontSize: 12 }}>{i + 1}</span></Link>
            : <span key={l.slug} className="off">{l.title} · soon</span>)}
        </nav>
        <article style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <header>
            <span className="eyebrow mute">Lesson {m.index} of {m.total} · {m.verb}</span>
            <h1 style={{ fontSize: "clamp(44px,6vw,64px)", marginTop: 6 }}>{m.title}</h1>
          </header>
          <div className="card stage" role="img" aria-label={`Interactive widget placeholder: ${c.widget}`}>
            <div><div className="mono" style={{ fontSize: 13 }}>widget: {c.widget}</div><p style={{ margin: "8px 0 0" }}>The interactive model for this lesson arrives in a later stage.</p></div>
          </div>
          <section className="card" style={{ padding: "22px 26px" }} aria-label="Overview">
            <h2 style={{ fontSize: 32, marginBottom: 10 }}>{c.headline}</h2>
            {c.overview.map((t, i) => <p key={i} style={{ margin: "0 0 10px" }}>{t}</p>)}
          </section>
          <section className="card" style={{ padding: "20px 26px" }} aria-label="Break it">
            <h2 style={{ fontSize: 26, marginBottom: 6 }}>{c.break.label}</h2>
            <p style={{ margin: 0 }} className="mute">{c.break.explains}</p>
          </section>
          <section className="card" style={{ padding: "20px 26px" }} aria-label="Check yourself">
            <h2 style={{ fontSize: 26, marginBottom: 8 }}>Check yourself</h2>
            <p style={{ margin: "0 0 8px" }}>{c.check.question}</p>
            <ol style={{ margin: 0, paddingLeft: 20 }}>{c.check.options.map((o) => <li key={o}>{o}</li>)}</ol>
          </section>
          <p className="mute" style={{ margin: 0 }}><strong style={{ color: "var(--ink)" }}>In real life.</strong> {c.realLife}</p>
        </article>
      </div>
    </Chrome>
  );
}
