import Chrome from "@/components/Chrome";
import WorldMap from "@/components/WorldMap";
import { Tracker } from "@/components/Progress";
import { PARTS, liveInOrder } from "@/lib/lessons";

export const metadata = { title: "Map · Wobble" };
export default function Lessons() {
  return <Chrome current="lessons"><div className="page-head"><span className="kicker">SELECT A LEVEL</span><h1>The map</h1><p className="lede">Six worlds, from a straight line to diffusion. Go in order or jump about. Nobody is checking.</p><Tracker slugs={liveInOrder()} /></div><WorldMap parts={PARTS} /></Chrome>;
}
