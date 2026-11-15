"use client";
import { Canvas } from "@react-three/fiber";
import { RoundedBox } from "@react-three/drei";
import { useMemo } from "react";
import * as THREE from "three";

/** One small toy-like 3D object per lesson, rendered once to a PNG for the lesson cards. */
const INK = "#24301E", MAT = "#7FA556", MATD = "#5E8A3A", ROSE = "#D25579", PINK = "#F2B8C6", CREAM = "#FBEFEA", PAPER = "#FFF8F3", GOLD = "#E2B04A", BLUE = "#6C82C9", GREY = "#C9C6BE";
type V = [number, number, number];
const M = ({ c, r = 0.55 }: { c: string; r?: number }) => <meshStandardMaterial color={c} roughness={r} />;
const Tile = ({ w = 3, d = 3, c = "#FFFDFB" }: { w?: number; d?: number; c?: string }) => <group><RoundedBox args={[w, 0.22, d]} radius={0.07} position={[0, -0.11, 0]}><M c={c} r={0.9} /></RoundedBox><RoundedBox args={[w + 0.08, 0.06, d + 0.08]} radius={0.03} position={[0, -0.25, 0]}><M c={INK} /></RoundedBox></group>;
const Ball = ({ p, c = MAT, s = 0.16 }: { p: V; c?: string; s?: number }) => <mesh position={p}><sphereGeometry args={[s, 24, 24]} /><M c={c} r={0.35} /></mesh>;
const Box = ({ p, s, c = MAT }: { p: V; s: V; c?: string }) => <RoundedBox args={s} radius={Math.min(0.06, s[0] / 4, s[1] / 4, s[2] / 4)} position={p}><M c={c} /></RoundedBox>;
const Cyl = ({ p, h, r = 0.05, c = INK, rot = [0, 0, 0] as V }: { p: V; h: number; r?: number; c?: string; rot?: V }) => <mesh position={p} rotation={rot}><cylinderGeometry args={[r, r, h, 16]} /><M c={c} /></mesh>;
function Rod({ a, b, r = 0.04, c = INK }: { a: V; b: V; r?: number; c?: string }) {
  const { pos, q, len } = useMemo(() => { const A = new THREE.Vector3(...a), B = new THREE.Vector3(...b), d = B.clone().sub(A); return { pos: A.clone().add(B).multiplyScalar(0.5), len: d.length(), q: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize()) }; }, [a, b]);
  return <mesh position={pos} quaternion={q}><cylinderGeometry args={[r, r, len, 12]} /><M c={c} /></mesh>;
}
function Tube({ f, c = ROSE, r = 0.07, n = 80 }: { f: (t: number) => V; c?: string; r?: number; n?: number }) {
  const g = useMemo(() => new THREE.TubeGeometry(new THREE.CatmullRomCurve3(Array.from({ length: n + 1 }, (_, i) => new THREE.Vector3(...f(i / n)))), n * 2, r, 12, false), [f, r, n]);
  return <mesh geometry={g}><M c={c} r={0.35} /></mesh>;
}
function Surf({ f, w = 2.6, c1 = MAT, c2 = PINK }: { f: (x: number, z: number) => number; w?: number; c1?: string; c2?: string }) {
  const g = useMemo(() => { const g = new THREE.PlaneGeometry(w, w, 50, 50); g.rotateX(-Math.PI / 2); const p = g.attributes.position, col: number[] = [], A = new THREE.Color(c1), B = new THREE.Color(c2); let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < p.count; i++) { const y = f(p.getX(i), p.getZ(i)); p.setY(i, y); lo = Math.min(lo, y); hi = Math.max(hi, y); }
    for (let i = 0; i < p.count; i++) { const k = A.clone().lerp(B, (p.getY(i) - lo) / (hi - lo || 1)); col.push(k.r, k.g, k.b); }
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3)); g.computeVertexNormals(); return g; }, [f, w, c1, c2]);
  return <mesh geometry={g}><meshStandardMaterial vertexColors roughness={0.7} side={THREE.DoubleSide} /></mesh>;
}
const r = (i: number) => { const x = Math.sin(i * 12.9898) * 43758.5453; return x - Math.floor(x); };
const pts = <T,>(n: number, f: (i: number) => T): T[] => Array.from({ length: n }, (_, i) => f(i));
function Tree({ p, s = 1 }: { p: V; s?: number }) { return <group position={p} scale={s}><Cyl p={[0, 0.25, 0]} h={0.5} r={0.07} c="#8B6A4E" /><mesh position={[0, 0.75, 0]}><coneGeometry args={[0.35, 0.8, 16]} /><M c={MAT} /></mesh></group>; }
function Nodes({ layers, x0 = -1, gap = 1, drop = [] as number[], y = 0.7 }: { layers: number[]; x0?: number; gap?: number; drop?: number[]; y?: number }) {
  const P: V[][] = layers.map((n, l) => pts(n, (i) => [x0 + l * gap, y + (i - (n - 1) / 2) * 0.5, 0] as V)); let k = 0;
  return <group>{P.slice(1).map((L, l) => L.flatMap((b, i) => P[l].map((a, j) => <Rod key={`${l}${i}${j}`} a={a} b={b} r={0.015} c={GREY} />)))}{P.flat().map((p) => { const id = k++; return <Ball key={id} p={p} s={0.15} c={drop.includes(id) ? GREY : id < layers[0] ? MAT : id >= P.flat().length - layers.at(-1)! ? ROSE : BLUE} />; })}</group>;
}

const BUILD: Record<string, () => React.ReactElement> = {
  "linear-regression": () => <><Tile />{pts(8, (i) => [-1.1 + i * 0.32, 0.3 + i * 0.12 + (r(i) - 0.5) * 0.35, (r(i + 9) - 0.5) * 0.8] as V).map((p, i) => <Ball key={i} p={p} />)}<Rod a={[-1.3, 0.2, 0]} b={[1.3, 1.25, 0]} r={0.05} c={ROSE} /></>,
  "gradient-descent": () => <><Surf f={(x, z) => 0.18 * (x * x + z * z)} /><Ball p={[-0.9, 0.18 * 1.6 + 0.16, 0.9]} c={ROSE} s={0.18} /><Ball p={[0, 0.12, 0]} c={MATD} s={0.08} /></>,
  "learning-rate": () => <><Surf f={(x, z) => 0.18 * (x * x + z * z)} /><Tube f={(t) => { const k = 6 * t; return [Math.cos(k * Math.PI) * (0.4 + t), 0.5 + t * 0.6, -0.2] as V; }} r={0.04} c={INK} /><Ball p={[1.3, 1.2, -0.2]} c={ROSE} s={0.18} /></>,
  "feature-scaling": () => <><Tile /><Box p={[-0.6, 0.15, 0]} s={[0.5, 0.3, 0.5]} c={MAT} /><Box p={[0.6, 1.0, 0]} s={[0.5, 2, 0.5]} c={ROSE} /><Box p={[0, 0.03, 1]} s={[2.4, 0.06, 0.3]} c={GOLD} /></>,
  "polynomial-features": () => <><Tile /><Tube f={(t) => [-1.3 + 2.6 * t, 0.8 + 0.5 * Math.sin(t * Math.PI * 2), 0] as V} />{pts(7, (i) => [-1.2 + i * 0.4, 0.8 + 0.5 * Math.sin(((i * 0.4) / 2.6) * Math.PI * 2 + 0.2) + (r(i) - 0.5) * 0.2, 0.15] as V).map((p, i) => <Ball key={i} p={p} s={0.12} />)}</>,
  overfitting: () => <><Tile /><Tube f={(t) => [-1.3 + 2.6 * t, 0.8 + 0.35 * Math.sin(t * 20) * Math.sin(t * Math.PI), 0] as V} r={0.05} n={160} />{pts(8, (i) => [-1.15 + i * 0.33, 0.8 + (r(i) - 0.5) * 0.6, 0.15] as V).map((p, i) => <Ball key={i} p={p} s={0.11} />)}</>,
  regularisation: () => <><Tile /><Tube f={(t) => [0.4 * Math.cos(t * 30), 0.15 + t * 1.6, 0.4 * Math.sin(t * 30)] as V} c={GOLD} r={0.05} n={240} /><Box p={[0, 1.85, 0]} s={[1, 0.18, 1]} c={ROSE} /></>,
  "bias-and-variance": () => <>{[1.2, 0.85, 0.5, 0.18].map((rad, i) => <mesh key={i} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02 + i * 0.03, 0]}><circleGeometry args={[rad, 40]} /><M c={i % 2 ? PAPER : ROSE} /></mesh>)}<mesh position={[0, -0.05, 0]}><cylinderGeometry args={[1.25, 1.25, 0.1, 40]} /><M c={INK} /></mesh>{pts(5, (i) => [0.5 + (r(i) - 0.5) * 0.4, 0.25, -0.4 + (r(i + 3) - 0.5) * 0.4] as V).map((p, i) => <Ball key={i} p={p} c={MATD} s={0.09} />)}</>,
  "logistic-regression": () => <><Tile /><Tube f={(t) => [-1.3 + 2.6 * t, 0.2 + 1.3 / (1 + Math.exp(-(t - 0.5) * 14)), 0] as V} /></>,
  "decision-boundaries": () => <><Box p={[-0.75, 0, 0]} s={[1.5, 0.2, 2.6]} c={MAT} /><Box p={[0.75, 0, 0]} s={[1.5, 0.2, 2.6]} c={PINK} /><Box p={[0, 0.5, 0]} s={[0.06, 1, 2.6]} c={INK} />{pts(5, (i) => [-0.4 - r(i) * 0.9, 0.25, (r(i + 4) - 0.5) * 2] as V).map((p, i) => <Ball key={i} p={p} c={MATD} />)}{pts(5, (i) => [0.4 + r(i + 7) * 0.9, 0.25, (r(i + 2) - 0.5) * 2] as V).map((p, i) => <Ball key={`b${i}`} p={p} c={ROSE} />)}</>,
  "k-nearest-neighbours": () => <><Tile /><Ball p={[0, 0.3, 0]} c={GOLD} s={0.2} /><mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.03, 0]}><ringGeometry args={[0.85, 0.92, 48]} /><M c={INK} /></mesh>{pts(9, (i) => { const a = i * 0.7, d = 0.4 + r(i) * 0.9; return [Math.cos(a) * d, 0.2, Math.sin(a) * d] as V; }).map((p, i) => <Ball key={i} p={p} c={i % 3 ? MAT : ROSE} />)}</>,
  "naive-bayes": () => <><Tile />{([[-0.5, 0.3, 0], [0.5, 0.3, 0.1], [0, 0.9, 0.05]] as V[]).map((p, i) => <group key={i}><Box p={p} s={[0.6, 0.6, 0.6]} c={i === 2 ? ROSE : PAPER} />{[0, 1, 2].map((k) => <Ball key={k} p={[p[0] - 0.15 + k * 0.15, p[1] + 0.31, p[2] - 0.15 + k * 0.15]} s={0.05} c={INK} />)}</group>)}</>,
  "decision-trees": () => <><Tile /><Box p={[0, 1.4, 0]} s={[0.5, 0.3, 0.5]} c={INK} />{([[-0.7, 0.8, 0], [0.7, 0.8, 0]] as V[]).map((p, i) => <group key={i}><Rod a={[0, 1.3, 0]} b={p} r={0.03} /><Box p={p} s={[0.4, 0.26, 0.4]} c={GOLD} /><Rod a={p} b={[p[0] - 0.35, 0.2, 0]} r={0.025} /><Rod a={p} b={[p[0] + 0.35, 0.2, 0]} r={0.025} /><Box p={[p[0] - 0.35, 0.15, 0]} s={[0.3, 0.3, 0.3]} c={MAT} /><Box p={[p[0] + 0.35, 0.15, 0]} s={[0.3, 0.3, 0.3]} c={ROSE} /></group>)}</>,
  "random-forests": () => <><Tile /><Tree p={[-0.7, 0, 0.4]} /><Tree p={[0.7, 0, 0.3]} s={0.9} /><Tree p={[0, 0, -0.6]} s={1.15} /><Tree p={[0.4, 0, 0.9]} s={0.7} /></>,
  "gradient-boosting": () => <><Tile />{[0, 1, 2, 3].map((i) => <Box key={i} p={[-1 + i * 0.66, 0.2 + i * 0.25, 0]} s={[0.6, 0.4 + i * 0.5, 0.8]} c={[MAT, MAT, GOLD, ROSE][i]} />)}</>,
  "support-vector-machines": () => <><Tile />{pts(5, (i) => [-0.6 - r(i) * 0.6, 0.2, (r(i + 4) - 0.5) * 2] as V).map((p, i) => <Ball key={i} p={p} c={MATD} />)}{pts(5, (i) => [0.6 + r(i + 7) * 0.6, 0.2, (r(i + 2) - 0.5) * 2] as V).map((p, i) => <Ball key={`b${i}`} p={p} c={ROSE} />)}<mesh position={[0, 0.6, 0]}><boxGeometry args={[0.7, 1.2, 2.6]} /><meshStandardMaterial color={GOLD} transparent opacity={0.3} /></mesh><Box p={[0, 0.6, 0]} s={[0.04, 1.2, 2.6]} c={INK} /></>,
  "train-and-test": () => <><Box p={[-0.3, 0.4, 0]} s={[2, 0.8, 1.6]} c={MAT} /><Box p={[1.05, 0.4, 0]} s={[0.55, 0.8, 1.6]} c={ROSE} /></>,
  "cross-validation": () => <>{[0, 1, 2, 3, 4].map((i) => <Box key={i} p={[0, 0.12 + i * 0.3, 0]} s={[2.2, 0.24, 1.6]} c={i === 2 ? ROSE : i % 2 ? MAT : "#A9C57F"} />)}</>,
  "data-leakage": () => <><Tile /><mesh position={[0, 0.6, 0]}><cylinderGeometry args={[0.6, 0.5, 1.2, 32, 1, true]} /><meshStandardMaterial color={BLUE} side={THREE.DoubleSide} /></mesh><mesh position={[0, 0.02, 0]}><cylinderGeometry args={[0.5, 0.5, 0.04, 32]} /><M c={BLUE} /></mesh><Cyl p={[0.75, 0.25, 0]} h={0.5} r={0.08} c={GREY} rot={[0, 0, Math.PI / 2]} /><mesh position={[1.05, 0.05, 0]}><sphereGeometry args={[0.13, 20, 20]} /><M c={ROSE} r={0.2} /></mesh></>,
  metrics: () => <><Tile />{([[-0.45, -0.45, 1.2, MAT], [0.45, -0.45, 0.35, ROSE], [-0.45, 0.45, 0.25, ROSE], [0.45, 0.45, 1.0, MAT]] as [number, number, number, string][]).map(([x, z, h, c], i) => <Box key={i} p={[x, h / 2, z]} s={[0.8, h, 0.8]} c={c} />)}</>,
  "imbalanced-classes": () => <><Tile /><mesh position={[0, 0.3, 0]}><coneGeometry args={[0.3, 0.6, 3]} /><M c={INK} /></mesh><group position={[0, 0.62, 0]} rotation={[0, 0, -0.25]}><Box p={[0, 0, 0]} s={[2.6, 0.08, 0.5]} c={GOLD} />{pts(9, (i) => [0.7 + (i % 3) * 0.2, 0.15 + Math.floor(i / 3) * 0.2, (i % 2) * 0.1] as V).map((p, i) => <Ball key={i} p={p} c={MAT} s={0.1} />)}<Ball p={[-1, 0.15, 0]} c={ROSE} s={0.1} /></group></>,
  calibration: () => <><Tile /><mesh position={[0, 0.2, 0]} rotation={[Math.PI / 2, 0, 0]}><torusGeometry args={[1, 0.12, 12, 40, Math.PI]} /><M c={MAT} /></mesh><Rod a={[0, 0.2, 0.05]} b={[0.55, 0.9, 0.05]} r={0.05} c={ROSE} /><Ball p={[0, 0.2, 0.05]} c={INK} s={0.12} /></>,
  "hyperparameter-search": () => <><Tile />{pts(9, (i) => [-0.8 + (i % 3) * 0.8, 0, -0.8 + Math.floor(i / 3) * 0.8] as V).map((p, i) => <group key={i} position={p}><Cyl p={[0, 0.2, 0]} h={0.4} r={0.18} c={i === 5 ? ROSE : PAPER} /><Box p={[0, 0.42, 0.1]} s={[0.05, 0.05, 0.16]} c={INK} /></group>)}</>,
  "k-means": () => <><Tile />{([[-0.7, -0.5, MAT], [0.7, -0.4, ROSE], [0, 0.7, BLUE]] as [number, number, string][]).map(([x, z, c], k) => <group key={k}>{pts(6, (i) => [x + (r(i + k * 7) - 0.5) * 0.8, 0.15, z + (r(i + k * 3 + 1) - 0.5) * 0.8] as V).map((p, i) => <Ball key={i} p={p} c={c} s={0.13} />)}<Box p={[x, 0.55, z]} s={[0.25, 0.25, 0.25]} c={INK} /></group>)}</>,
  "clustering-methods": () => <><Tile />{[-0.9, -0.3, 0.3, 0.9].map((x, i) => <group key={i}><Cyl p={[x, 0.3, 0]} h={0.6} r={0.035} /><Ball p={[x, 0.12, 0]} c={i < 2 ? MAT : ROSE} s={0.14} /></group>)}<Rod a={[-0.9, 0.6, 0]} b={[-0.3, 0.6, 0]} r={0.035} /><Rod a={[0.3, 0.8, 0]} b={[0.9, 0.8, 0]} r={0.035} /><Cyl p={[0.3, 0.7, 0]} h={0.2} r={0.035} /><Cyl p={[0.9, 0.7, 0]} h={0.2} r={0.035} /><Cyl p={[-0.6, 0.85, 0]} h={0.5} r={0.035} /><Cyl p={[0.6, 0.95, 0]} h={0.3} r={0.035} /><Rod a={[-0.6, 1.1, 0]} b={[0.6, 1.1, 0]} r={0.035} c={GOLD} /></>,
  pca: () => <><Tile />{pts(22, (i) => { const t = (r(i) - 0.5) * 2.2; return [t, 0.6 + t * 0.35 + (r(i + 5) - 0.5) * 0.3, (r(i + 9) - 0.5) * 0.4] as V; }).map((p, i) => <Ball key={i} p={p} s={0.08} />)}<Rod a={[-1.2, 0.18, 0]} b={[1.2, 1.02, 0]} r={0.035} c={ROSE} /><mesh position={[1.2, 1.02, 0]} rotation={[0, 0, -1.23]}><coneGeometry args={[0.1, 0.25, 16]} /><M c={ROSE} /></mesh></>,
  "t-sne-and-umap": () => <><Tile />{[0, 1].map((k) => pts(12, (i) => { const a = (i / 12) * Math.PI * 2; return [(k ? 0.6 : -0.6) + Math.cos(a) * 0.45, 0.15, Math.sin(a) * 0.45] as V; }).map((p, i) => <Ball key={`${k}${i}`} p={p} c={k ? ROSE : MAT} s={0.1} />))}</>,
  "gaussian-mixtures": () => <Surf f={(x, z) => 1.1 * Math.exp(-((x + 0.5) ** 2 + z * z) / 0.3) + 0.7 * Math.exp(-((x - 0.7) ** 2 + (z - 0.3) ** 2) / 0.25)} c1={PAPER} c2={ROSE} />,
  "anomaly-detection": () => <><Tile />{pts(14, (i) => [-0.4 + (r(i) - 0.5) * 1, 0.15, (r(i + 4) - 0.5) * 1] as V).map((p, i) => <Ball key={i} p={p} c={MAT} s={0.11} />)}<Cyl p={[0.9, 0.4, 0.8]} h={0.8} r={0.02} /><Ball p={[0.9, 0.85, 0.8]} c={ROSE} s={0.18} /></>,
  perceptron: () => <><Tile />{[-0.6, 0, 0.6].map((z, i) => <group key={i}><Ball p={[-1, 0.5, z]} s={0.14} /><Rod a={[-1, 0.5, z]} b={[0.3, 0.7, 0]} r={0.025} c={GREY} /></group>)}<Ball p={[0.3, 0.7, 0]} c={ROSE} s={0.3} /><Rod a={[0.3, 0.7, 0]} b={[1.2, 0.7, 0]} r={0.03} /></>,
  "multilayer-networks": () => <Nodes layers={[3, 4, 4, 2]} x0={-1.3} gap={0.85} />,
  backpropagation: () => <><Nodes layers={[2, 3, 2]} x0={-1} gap={1} /><Tube f={(t) => [1 - 2 * t, 1.55 + 0.25 * Math.sin(t * Math.PI), 0] as V} c={ROSE} r={0.05} /><mesh position={[-1, 1.55, 0]} rotation={[0, 0, Math.PI / 2]}><coneGeometry args={[0.12, 0.28, 16]} /><M c={ROSE} /></mesh></>,
  activations: () => <><Tile /><Tube f={(t) => { const x = -1.2 + 2.4 * t; return [x, 0.3 + Math.max(0, x) * 0.8, 0] as V; }} c={ROSE} r={0.07} /><Tube f={(t) => { const x = -1.2 + 2.4 * t; return [x, 0.3 + 0.9 * Math.tanh(x * 2) * 0.5 + 0.45, -0.5] as V; }} c={MAT} r={0.05} /></>,
  optimisers: () => <><Surf f={(x, z) => 0.08 * x * x + 0.35 * z * z} /><Ball p={[-1, 0.29, 0.6]} c={ROSE} s={0.13} /><Ball p={[-0.5, 0.09, 0.3]} c={BLUE} s={0.13} /><Ball p={[0, 0.13, 0]} c={GOLD} s={0.13} /></>,
  "vanishing-gradients": () => <><Tile />{[0, 1, 2, 3, 4, 5].map((i) => <group key={i}><Ball p={[-1.25 + i * 0.5, 0.6, 0]} c={i === 0 ? GREY : MAT} s={0.2 - i * 0.0} />{i < 5 && <Rod a={[-1.25 + i * 0.5, 0.6, 0]} b={[-0.75 + i * 0.5, 0.6, 0]} r={0.01 + i * 0.012} c={ROSE} />}</group>)}</>,
  convolutions: () => <>{pts(25, (i) => [-1 + (i % 5) * 0.5, 0, -1 + Math.floor(i / 5) * 0.5] as V).map((p, i) => <Box key={i} p={p} s={[0.46, 0.15, 0.46]} c={(i % 5 === 2) ? INK : PAPER} />)}<group position={[-0.5, 0.7, -0.5]}>{pts(9, (i) => [-0.5 + (i % 3) * 0.5, 0, -0.5 + Math.floor(i / 3) * 0.5] as V).map((p, i) => <mesh key={i} position={p}><boxGeometry args={[0.46, 0.1, 0.46]} /><meshStandardMaterial color={ROSE} transparent opacity={0.75} /></mesh>)}</group></>,
  dropout: () => <Nodes layers={[3, 5, 5, 2]} x0={-1.3} gap={0.85} drop={[4, 6, 9, 11]} />,
  "time-series": () => <><Tile />{pts(10, (i) => i).map((i) => { const h = 0.3 + i * 0.07 + 0.3 * Math.sin(i * 1.2) + 0.3; return <Box key={i} p={[-1.2 + i * 0.27, h / 2, 0]} s={[0.2, h, 0.6]} c={i > 7 ? ROSE : MAT} />; })}</>,
  "hidden-markov-models": () => <><Tile /><Ball p={[-0.65, 0.45, 0]} c={MAT} s={0.4} /><Ball p={[0.65, 0.45, 0]} c={ROSE} s={0.4} /><mesh position={[0, 0.75, 0]} rotation={[0, 0, 0]}><torusGeometry args={[0.65, 0.04, 10, 40, Math.PI]} /><M c={INK} /></mesh><mesh position={[0, 0.2, 0]} rotation={[0, 0, Math.PI]}><torusGeometry args={[0.65, 0.04, 10, 40, Math.PI]} /><M c={INK} /></mesh></>,
  embeddings: () => <><Tile /><Rod a={[-1.2, 0.05, 1]} b={[1.2, 0.05, 1]} r={0.025} /><Rod a={[-1.2, 0.05, 1]} b={[-1.2, 1.8, 1]} r={0.025} /><Rod a={[-1.2, 0.05, 1]} b={[-1.2, 0.05, -1.2]} r={0.025} />{([[0.4, 1.2, -0.2, ROSE], [0.6, 1.0, -0.4, ROSE], [-0.4, 0.5, 0.2, MAT], [-0.2, 0.4, 0.4, MAT], [0.7, 0.3, 0.6, BLUE], [0.9, 0.5, 0.5, BLUE]] as [number, number, number, string][]).map(([x, y, z, c], i) => <Ball key={i} p={[x, y, z]} c={c} s={0.15} />)}</>,
  attention: () => <>{pts(25, (i) => i).map((i) => { const a = i % 5, b = Math.floor(i / 5), h = a === b ? 1.1 : 0.15 + r(i) * 0.35; return <Box key={i} p={[-1 + a * 0.5, h / 2, -1 + b * 0.5]} s={[0.42, h, 0.42]} c={a === b ? ROSE : "#E9D7CF"} />; })}</>,
  "bandits-and-q-learning": () => <><Tile />{[-0.8, 0, 0.8].map((x, i) => <group key={i}><Box p={[x, 0.55, 0]} s={[0.6, 1.1, 0.6]} c={[PAPER, PINK, PAPER][i]} /><Box p={[x, 0.75, 0.31]} s={[0.4, 0.3, 0.02]} c={INK} /><Cyl p={[x + 0.36, 0.9, 0]} h={0.5} r={0.03} /><Ball p={[x + 0.36, 1.18, 0]} c={i === 1 ? ROSE : MAT} s={0.08} /></group>)}</>,
  autoencoders: () => <><Tile /><mesh position={[-0.55, 0.6, 0]} rotation={[0, 0, Math.PI / 2]}><cylinderGeometry args={[0.08, 0.6, 1, 4]} /><M c={MAT} /></mesh><mesh position={[0.55, 0.6, 0]} rotation={[0, 0, -Math.PI / 2]}><cylinderGeometry args={[0.08, 0.6, 1, 4]} /><M c={PINK} /></mesh><Ball p={[0, 0.6, 0]} c={ROSE} s={0.13} /></>,
  diffusion: () => <><Tile />{pts(30, (i) => { const k = i % 6, row = Math.floor(i / 6), t = k / 5, x = -1.2 + k * 0.48, z = -1 + row * 0.5; return [x + (1 - t) * (r(i) - 0.5) * 0.5, 0.15 + (1 - t) * r(i + 2) * 0.6, z + (1 - t) * (r(i + 5) - 0.5) * 0.5] as V; }).map((p, i) => <Ball key={i} p={p} c={i % 6 > 3 ? ROSE : i % 6 > 1 ? PINK : GREY} s={0.1} />)}</>,
  interpretability: () => <><Tile />{[1.1, 0.7, 0.45, 0.2].map((w, i) => <Box key={i} p={[-1.1 + w / 2, 0.15, -0.8 + i * 0.45]} s={[w, 0.2, 0.32]} c={i === 0 ? ROSE : MAT} />)}<mesh position={[0.7, 0.55, 0.4]} rotation={[-0.6, 0, 0]}><torusGeometry args={[0.35, 0.06, 12, 32]} /><M c={INK} /></mesh><Rod a={[0.9, 0.35, 0.65]} b={[1.25, 0.1, 1.0]} r={0.06} /></>,
  fairness: () => <><Tile /><Cyl p={[0, 0.65, 0]} h={1.3} r={0.06} /><Box p={[0, 1.3, 0]} s={[2.2, 0.06, 0.12]} c={GOLD} />{[-1, 1].map((x) => <group key={x}><Rod a={[x, 1.3, 0]} b={[x, 0.75, 0]} r={0.015} /><mesh position={[x, 0.7, 0]}><cylinderGeometry args={[0.35, 0.25, 0.1, 24]} /><M c={x < 0 ? MAT : PINK} /></mesh><Ball p={[x, 0.85, 0]} c={x < 0 ? MATD : ROSE} s={0.12} /></group>)}</>,
};

export function ThumbScene({ slug }: { slug: string }) {
  const B = BUILD[slug] ?? BUILD["linear-regression"];
  return (<Canvas orthographic dpr={2} camera={{ position: [6, 5.2, 6], zoom: 92 }} gl={{ alpha: true, antialias: true, preserveDrawingBuffer: true }} style={{ width: 480, height: 360, background: "transparent" }} onCreated={({ camera }) => camera.lookAt(0, 0.45, 0)}>
    <hemisphereLight args={["#ffffff", "#d8c4bc", 1.1]} />
    <directionalLight position={[4, 8, 3]} intensity={1.7} />
    <directionalLight position={[-5, 3, -2]} intensity={0.4} />
    <group position={[0, -0.2, 0]}><B /></group>
  </Canvas>);
}
export const THUMB_SLUGS = Object.keys(BUILD);
