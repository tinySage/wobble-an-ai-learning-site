import Link from "next/link";
import Chrome from "@/components/Chrome";
import HeroToy from "@/components/HeroToy";
import WorldMap from "@/components/WorldMap";
import { Tracker } from "@/components/Progress";
import { PARTS, firstLive, liveInOrder } from "@/lib/lessons";

export default function Home() {
  const first = firstLive();
  return (
    <Chrome>
      <section className="hero" aria-label="Intro">
        <div className="hero-copy">
          <span className="kicker">46 LEVELS · 6 WORLDS · 0 SIGN-UPS</span>
          <h1>Learn machine learning by <span className="hl">breaking</span> it.</h1>
          <p className="lede">Every lesson is a tiny toy. Drag it until it works, then hit the red button and watch it fall apart. The falling apart is the lesson.</p>
          <div className="cta"><Link href={`/lessons/${first}/`} className="px-btn primary">Press start ▸</Link><Link href="/lessons/" className="px-btn">Open the map</Link></div>
          <ul className="perks"><li>runs in your browser</li><li>maths tested against scikit-learn</li><li>sources for every claim</li></ul>
        </div>
        <HeroToy first={first} />
      </section>

      <section className="how" aria-label="How a level works">
        <div className="how-step"><span className="how-n">01</span><b>Make it</b><p>Drag points, slide knobs, train a tiny model live.</p></div>
        <div className="how-step break"><span className="how-n">02</span><b>Break it</b><p>One red button pushes it to fail on purpose.</p></div>
        <div className="how-step"><span className="how-n">03</span><b>Get why</b><p>Read the maths, the code and where the textbooks disagree.</p></div>
        <div className="how-step"><span className="how-n">04</span><b>Earn a star</b><p>Answer one question to clear the level.</p></div>
      </section>

      <section aria-labelledby="map-title" className="map-sec">
        <div className="sec-head"><h2 id="map-title">The map</h2><Tracker slugs={liveInOrder()} /></div>
        <WorldMap parts={PARTS} />
      </section>
    </Chrome>
  );
}
