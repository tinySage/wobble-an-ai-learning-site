"use client";
import Link from "next/link";
import { useDone } from "@/components/Progress";
import type { Part } from "@/content/curriculum";

/** 8 x 8 pixel icons, one per world. */
const ICONS: Record<number, string[]> = {
  1: ["......xx", ".....x..", "....x...", "...x..o.", "..x.....", ".x..o...", "x.o.....", "........"],
  2: ["oo....xx", "oo....xx", "...l....", "...l....", "...l....", "...l....", "xx....oo", "xx....oo"],
  3: ["..xxxx..", ".x....x.", "x..oo..x", "x.o..o.x", "x.o..o.x", "x..oo..x", ".x....x.", "..xxxx.."],
  4: ["oo......", "oo..x...", "...xx...", "........", ".....oo.", "x....oo.", "xx......", "........"],
  5: ["o.....o.", ".l...l..", "..x.x...", "...x....", "..x.x...", ".l...l..", "o.....o.", "........"],
  6: ["...x....", "...x....", ".x.x.x..", "..xxx...", "xxxoxxx.", "..xxx...", ".x.x.x..", "...x...."],
};
const IC: Record<string, string> = { x: "#58792F", o: "#C4416B", l: "#26301F" };
export function WorldIcon({ id, size = 40 }: { id: number; size?: number }) {
  return <svg viewBox="0 0 8 8" width={size} height={size} shapeRendering="crispEdges" aria-hidden="true">{(ICONS[id] ?? ICONS[1]).flatMap((r, y) => [...r].map((c, x) => (c === "." ? null : <rect key={`${x}-${y}`} x={x} y={y} width={1} height={1} fill={IC[c]} />)))}</svg>;
}
export const TAGLINES: Record<number, string> = {
  1: "Draw a line. Then draw far too many.",
  2: "Put things in boxes. Watch the boxes lie.",
  3: "Your accuracy is lying. Catch it.",
  4: "No labels, just geometry and nerve.",
  5: "One neuron, then far too many.",
  6: "Time, words, agents, noise, and who gets hurt.",
};

export default function WorldMap({ parts }: { parts: Part[] }) {
  const done = useDone(), order = parts.flatMap((p) => p.lessons.map((l) => l.slug)), next = order.find((s) => !done.includes(s));
  return (
    <div className="worlds">
      {parts.map((p) => {
        const got = p.lessons.filter((l) => done.includes(l.slug)).length;
        return (
          <section key={p.id} className={`world w${p.id}`} aria-labelledby={`world-${p.id}`}>
            <header className="world-head">
              <WorldIcon id={p.id} />
              <div>
                <span className="kicker">WORLD {p.id} · {got}/{p.lessons.length} CLEARED</span>
                <h2 id={`world-${p.id}`}>{p.title}</h2>
                <p>{TAGLINES[p.id] ?? p.blurb}</p>
              </div>
            </header>
            <ol className="path">
              {p.lessons.map((l, i) => {
                const isDone = done.includes(l.slug), isNext = l.slug === next;
                return (
                  <li key={l.slug} className={`stop ${i % 2 ? "low" : ""}`}>
                    <Link href={`/lessons/${l.slug}/`} className={`tile ${isDone ? "done" : ""} ${isNext ? "next" : ""}`} aria-label={`${p.id}-${i + 1}: ${l.title}${isDone ? ", cleared" : ""}${isNext ? ", next up" : ""}`}>
                      <span className="num">{p.id}-{i + 1}</span>
                      {isDone && <span className="star" aria-hidden="true">★</span>}
                      {isNext && <span className="you" aria-hidden="true">YOU</span>}
                    </Link>
                    <span className="stop-t">{l.title}</span>
                    <span className="stop-v">{l.verb}</span>
                  </li>
                );
              })}
            </ol>
          </section>
        );
      })}
    </div>
  );
}
