import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { PARTS, allPlanned } from "../content/curriculum.ts";

const files = fs.readdirSync("content/lessons").filter((f) => f.endsWith(".json"));

test("every planned slug is unique", () => {
  const s = allPlanned().map((l) => l.slug);
  assert.equal(new Set(s).size, s.length);
});
test("curriculum has six parts and lessons in each", () => {
  assert.equal(PARTS.length, 6);
  assert.ok(PARTS.every((p) => p.lessons.length >= 6));
});
test("every lesson file matches a planned lesson and has the required fields", () => {
  const planned = new Set(allPlanned().map((l) => l.slug));
  for (const f of files) {
    const c = JSON.parse(fs.readFileSync(`content/lessons/${f}`, "utf8"));
    assert.equal(c.slug + ".json", f);
    assert.ok(planned.has(c.slug), `${c.slug} is not in the curriculum`);
    for (const k of ["summary", "headline", "overview", "widget", "break", "check", "realLife", "data", "maths", "code", "sources"]) assert.ok(c[k], `${f} missing ${k}`);
    assert.ok(Number.isInteger(c.check.answer) && c.check.answer >= 0 && c.check.answer < c.check.options.length, `${f} check answer out of range`);
  }
});
test("every lesson cites at least two sources with links", () => {
  for (const f of files) {
    const c = JSON.parse(fs.readFileSync(`content/lessons/${f}`, "utf8"));
    assert.ok(c.sources.length >= 2, `${f} needs two or more sources`);
    assert.ok(c.sources.every((s) => /^https:\/\//.test(s.url) && s.title && s.note), `${f} has a malformed source`);
  }
});
