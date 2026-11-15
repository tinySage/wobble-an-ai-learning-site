import Shell from "@/components/shell/Shell";
import Hub from "@/components/Hub";
import LessonStrip from "@/components/shell/LessonStrip";
import { TAGLINES } from "@/content/taglines";
import { PARTS, firstLive, liveInOrder } from "@/lib/lessons";

export default function Home() {
  const items = PARTS.flatMap((p) => p.lessons.map((l, i) => ({ slug: l.slug, title: l.title, code: `${p.id}.${i + 1}`, world: p.id, worldTitle: p.title })));
  return (
    <Shell current="home" progress={{ label: "ALL LESSONS", slugs: liveInOrder() }}>
      <Hub first={firstLive()} worlds={PARTS.map((p) => ({ id: p.id, title: p.title, blurb: TAGLINES[p.id] ?? p.blurb, n: p.lessons.length, first: p.lessons[0].slug }))} />
      <LessonStrip items={items} worlds={PARTS.map((p) => ({ id: p.id, title: p.title }))} />
    </Shell>
  );
}
