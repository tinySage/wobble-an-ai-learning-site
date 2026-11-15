import fs from "node:fs";
import path from "node:path";
import { PARTS, allPlanned } from "@/content/curriculum";

export interface LessonContent {
  slug: string; summary: string; headline: string; overview: string[]; widget: string;
  break: { label: string; explains: string };
  check: { question: string; options: string[]; answer: number };
  realLife: string;
  data: string; maths: string[]; code: string;
  sources: { title: string; url: string; note: string }[];
}
const DIR = path.join(process.cwd(), "content", "lessons");

export function loadLesson(slug: string): LessonContent | null {
  try { return JSON.parse(fs.readFileSync(path.join(DIR, `${slug}.json`), "utf8")) as LessonContent; } catch { return null; }
}
export const liveSlugs = (): string[] => {
  try { return fs.readdirSync(DIR).filter((f) => f.endsWith(".json")).map((f) => f.replace(".json", "")); } catch { return []; }
};
export const isLive = (slug: string) => liveSlugs().includes(slug);
export function lessonMeta(slug: string) {
  const l = allPlanned().find((x) => x.slug === slug);
  if (!l) return null;
  const list = l.part.lessons, i = list.findIndex((x) => x.slug === slug);
  return { ...l, index: i + 1, total: list.length, prev: list[i - 1] ?? null, next: list[i + 1] ?? null };
}
export { PARTS };

/** Live lessons in teaching order (curriculum order, not file-name order). */
export const liveInOrder = (): string[] => allPlanned().map((l) => l.slug).filter(isLive);
export const firstLive = (): string => liveInOrder()[0];
