import Link from "next/link";
import Chrome from "@/components/Chrome";
import HeroPlot from "@/components/HeroPlot";
import PartGrid from "@/components/PartGrid";
import { allPlanned } from "@/content/curriculum";
import { firstLive, liveInOrder } from "@/lib/lessons";

export default function Home() {
  return (
    <Chrome>
      <section className="hero" aria-label="Intro">
        <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
          <span className="eyebrow" style={{ color: "var(--matcha)" }}>Machine learning, hands on</span>
          <h1>Learn it by <em>breaking</em> it.</h1>
          <p style={{ margin: 0, fontSize: 18, maxWidth: 480 }}>Short lessons from linear regression to transformers. Each one has a model you can drag, and a button that makes it fail. See why it fails, then fix it.</p>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <Link href={`/lessons/${firstLive()}/`} className="btn primary">Start learning</Link>
            <Link href="/lessons/" className="btn ghost">See all lessons</Link>
          </div>
          <div className="mono mute" style={{ display: "flex", gap: 22, flexWrap: "wrap", fontSize: 13 }}><span>{liveInOrder().length === allPlanned().length ? `${liveInOrder().length} lessons` : `${liveInOrder().length} lessons live, ${allPlanned().length} planned`}</span><span>No sign-up</span><span>Runs in your browser</span></div>
        </div>
        <HeroPlot />
      </section>
      <PartGrid />
    </Chrome>
  );
}
