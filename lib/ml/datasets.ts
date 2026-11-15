import { rng } from "./split.ts";
export type Kind = "blobs" | "moons" | "circles" | "xor";
export interface Data2D { X: number[][]; y: number[] }
const gauss = (r: () => number) => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());

/** Small two-class 2D datasets scaled to roughly [0, 1] x [0, 1]. Same seed, same points. */
export function makeData(kind: Kind, n = 160, noise = 0.08, seed = 4): Data2D {
  const r = rng(seed * 101 + kind.length), X: number[][] = [], y: number[] = [];
  for (let i = 0; i < n; i++) {
    const c = i % 2; let x = 0, z = 0;
    if (kind === "blobs") { x = (c ? 0.68 : 0.32) + gauss(r) * noise * 1.6; z = (c ? 0.62 : 0.38) + gauss(r) * noise * 1.6; }
    else if (kind === "moons") { const t = r() * Math.PI; x = c ? 0.5 - Math.cos(t) * 0.3 + 0.2 : 0.5 + Math.cos(t) * 0.3 - 0.2; z = c ? 0.5 - Math.sin(t) * 0.22 + 0.12 : 0.5 + Math.sin(t) * 0.22 - 0.12; x += gauss(r) * noise * 0.7; z += gauss(r) * noise * 0.7; }
    else if (kind === "circles") { const t = r() * 2 * Math.PI, rad = c ? 0.33 : 0.14; x = 0.5 + Math.cos(t) * rad + gauss(r) * noise * 0.5; z = 0.5 + Math.sin(t) * rad + gauss(r) * noise * 0.5; }
    else { const a = r() < 0.5 ? 0 : 1, b = r() < 0.5 ? 0 : 1; x = 0.28 + a * 0.44 + gauss(r) * noise * 1.3; z = 0.28 + b * 0.44 + gauss(r) * noise * 1.3; X.push([x, z]); y.push(a ^ b); continue; }
    X.push([x, z]); y.push(c);
  }
  return { X, y };
}
