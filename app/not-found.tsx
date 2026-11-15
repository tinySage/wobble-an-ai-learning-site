import Link from "next/link";
import Shell from "@/components/shell/Shell";
import { liveInOrder } from "@/lib/lessons";

export default function NotFound() {
  return <Shell progress={{ label: "ALL LESSONS", slugs: liveInOrder() }}><div className="nf"><span className="kicker">Error 404</span><h1>This page rolled off the map</h1><p>Like a learning rate set too high, this link overshot. The lessons are all still here.</p><div style={{ display: "flex", gap: 10, marginTop: 16 }}><Link href="/" className="btn primary">Home</Link><Link href="/lessons/" className="btn">The map</Link></div></div></Shell>;
}
