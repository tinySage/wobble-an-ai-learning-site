import Chrome from "@/components/Chrome";
import PartGrid from "@/components/PartGrid";
import { Tracker } from "@/components/Progress";
import { liveInOrder } from "@/lib/lessons";

export const metadata = { title: "Lessons · Wobble" };
export default function Lessons() {
  return <Chrome current="lessons"><h1 style={{ fontSize: 64, margin: "24px 0 8px" }}>All lessons</h1><Tracker slugs={liveInOrder()} /><div style={{ height: 28 }} /><PartGrid /></Chrome>;
}
