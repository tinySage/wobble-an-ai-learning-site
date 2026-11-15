import Link from "next/link";
import { PARTS, isLive } from "@/lib/lessons";

export default function PartGrid() {
  return (
    <div className="parts">
      {PARTS.map((p) => (
        <section key={p.id} aria-labelledby={`part-${p.id}`}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12, marginBottom: 14, flexWrap: "wrap" }}>
            <h2 id={`part-${p.id}`} style={{ fontSize: 38 }}><span className="mono mute" style={{ fontSize: 14, marginRight: 12 }}>PART {p.id}</span>{p.title}</h2>
            <span className="mute" style={{ fontSize: 15 }}>{p.blurb}</span>
          </div>
          <div className="grid">
            {p.lessons.map((l, i) => {
              const live = isLive(l.slug);
              const inner = (<><div style={{ display: "flex", justifyContent: "space-between" }}><span className="tag">{String(i + 1).padStart(2, "0")}</span>{!live && <span className="pill">SOON</span>}</div><span className="t">{l.title}</span><span className="tag">{l.verb}</span></>);
              return live
                ? <Link key={l.slug} href={`/lessons/${l.slug}/`} className="card lcard">{inner}</Link>
                : <div key={l.slug} className="card lcard soon" aria-disabled="true">{inner}</div>;
            })}
          </div>
        </section>
      ))}
    </div>
  );
}
