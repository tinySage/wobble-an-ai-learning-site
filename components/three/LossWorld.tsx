"use client";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { ContactShadows, OrbitControls, RoundedBox } from "@react-three/drei";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { OrbitControls as OC } from "three-stdlib";
import * as Ic from "@/components/shell/Icons";

/* ---------- the loss surface: a wide bowl, one tempting dip, gentle ripples ---------- */
const R = 3; // half-width of the terrain
export const loss = (x: number, z: number) =>
  0.16 * (x * x + 1.5 * z * z) - 0.75 * Math.exp(-((x - 0.8) ** 2 + (z - 0.35) ** 2) / 0.55) - 0.6 * Math.exp(-((x + 1.7) ** 2 + (z + 1.2) ** 2) / 0.4) + 0.12 * Math.sin(1.9 * x) * Math.cos(1.6 * z);
const H = (x: number, z: number) => { const v = 0.48 * loss(x, z); return v > 0 ? 1.15 * Math.tanh(v / 1.15) : v; };
const grad = (x: number, z: number) => { const e = 1e-4; return [(loss(x + e, z) - loss(x - e, z)) / (2 * e), (loss(x, z + e) - loss(x, z - e)) / (2 * e)]; };
const BASE = -0.7;
const MIN = (() => { let b = [0, 0], v = Infinity; for (let i = 0; i <= 240; i++) for (let j = 0; j <= 240; j++) { const x = -R + (2 * R * i) / 240, z = -R + (2 * R * j) / 240, l = loss(x, z); if (l < v) { v = l; b = [x, z]; } } return b; })();

const C_LOW = new THREE.Color("#7FA556"), C_MID = new THREE.Color("#DCE8BC"), C_HIGH = new THREE.Color("#F5C9D3"), C_TOP = new THREE.Color("#E58AA2");
function Terrain() {
  const geo = useMemo(() => {
    const n = 140, g = new THREE.PlaneGeometry(2 * R, 2 * R, n, n); g.rotateX(-Math.PI / 2);
    const p = g.attributes.position, col: number[] = [];
    let lo = Infinity, hi = -Infinity; for (let i = 0; i < p.count; i++) { const y = H(p.getX(i), p.getZ(i)); lo = Math.min(lo, y); hi = Math.max(hi, y); }
    for (let i = 0; i < p.count; i++) {
      const y = H(p.getX(i), p.getZ(i)); p.setY(i, y);
      const t = (y - lo) / (hi - lo), c = t < 0.35 ? C_LOW.clone().lerp(C_MID, t / 0.35) : t < 0.7 ? C_MID.clone().lerp(C_HIGH, (t - 0.35) / 0.35) : C_HIGH.clone().lerp(C_TOP, (t - 0.7) / 0.3);
      col.push(c.r, c.g, c.b);
    }
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3)); g.computeVertexNormals(); return g;
  }, []);
  // contour-ish grid lines drawn on the surface
  const lines = useMemo(() => {
    const pts: number[] = [], k = 12, m = 90;
    for (let a = 0; a <= k; a++) { const u = -R + (2 * R * a) / k;
      for (let b = 0; b < m; b++) { const v0 = -R + (2 * R * b) / m, v1 = -R + (2 * R * (b + 1)) / m;
        pts.push(u, H(u, v0) + 0.012, v0, u, H(u, v1) + 0.012, v1); pts.push(v0, H(v0, u) + 0.012, u, v1, H(v1, u) + 0.012, u); } }
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3)); return g;
  }, []);
  // skirt: walls from the terrain edge down to the base, so the surface reads as a solid object
  const skirt = useMemo(() => {
    const pos: number[] = [], col: number[] = [], m = 80, edge = (t: number, s: number): [number, number] => [[-R + 2 * R * t, -R], [R, -R + 2 * R * t], [R - 2 * R * t, R], [-R, R - 2 * R * t]][s] as [number, number];
    const top = new THREE.Color("#F3C3CF"), bot = new THREE.Color("#FBEFEA"), c = (y: number) => { const t = Math.max(0, Math.min(1, (y - BASE) / 1.9)); const k = bot.clone().lerp(top, t); return [k.r, k.g, k.b]; };
    for (let s = 0; s < 4; s++) for (let i = 0; i < m; i++) {
      const [x0, z0] = edge(i / m, s), [x1, z1] = edge((i + 1) / m, s), y0 = H(x0, z0), y1 = H(x1, z1);
      pos.push(x0, y0, z0, x0, BASE, z0, x1, y1, z1, x1, y1, z1, x0, BASE, z0, x1, BASE, z1);
      col.push(...c(y0), ...c(BASE), ...c(y1), ...c(y1), ...c(BASE), ...c(BASE));
    }
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3)); g.computeVertexNormals(); return g;
  }, []);
  // strata lines on the walls, like layers in a cut cake
  const strata = useMemo(() => {
    const pts: number[] = [], m = 80, edge = (t: number, s: number): [number, number] => [[-R + 2 * R * t, -R], [R, -R + 2 * R * t], [R - 2 * R * t, R], [-R, R - 2 * R * t]][s] as [number, number];
    for (let lv = BASE + 0.25; lv < 1.2; lv += 0.25) for (let s = 0; s < 4; s++) for (let i = 0; i < m; i++) {
      const [x0, z0] = edge(i / m, s), [x1, z1] = edge((i + 1) / m, s); if (H(x0, z0) < lv || H(x1, z1) < lv) continue;
      const o = 0.004, nx = s === 1 ? o : s === 3 ? -o : 0, nz = s === 0 ? -o : s === 2 ? o : 0; pts.push(x0 + nx, lv, z0 + nz, x1 + nx, lv, z1 + nz);
    }
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3)); return g;
  }, []);
  return (<group>
    <mesh geometry={geo} receiveShadow castShadow><meshStandardMaterial vertexColors roughness={0.82} metalness={0} flatShading={false} side={THREE.DoubleSide} /></mesh>
    <lineSegments geometry={lines}><lineBasicMaterial color="#24301E" transparent opacity={0.16} /></lineSegments>
    <mesh geometry={skirt} castShadow><meshStandardMaterial vertexColors roughness={0.9} side={THREE.DoubleSide} /></mesh>
    <lineSegments geometry={strata}><lineBasicMaterial color="#9E2E52" transparent opacity={0.22} /></lineSegments>
    <RoundedBox args={[2 * R + 0.3, 0.32, 2 * R + 0.3]} radius={0.08} position={[0, BASE - 0.15, 0]} receiveShadow castShadow><meshStandardMaterial color="#EFD9CF" roughness={0.95} /></RoundedBox>
    <RoundedBox args={[2 * R + 0.42, 0.08, 2 * R + 0.42]} radius={0.03} position={[0, BASE - 0.34, 0]}><meshStandardMaterial color="#24301E" roughness={0.9} /></RoundedBox>
  </group>);
}

/* ---------- the ball doing gradient descent ---------- */
export type Anchors = Record<string, THREE.Vector3>;
const START = [-2.45, 2.25];
function Descent({ lr, running, seed, anchors, onStep }: { lr: number; running: boolean; seed: number; anchors: React.MutableRefObject<Anchors>; onStep: (s: { step: number; loss: number; state: string }) => void }) {
  const ball = useRef<THREE.Mesh>(null), arrow = useRef<THREE.Group>(null), trail = useRef<THREE.BufferGeometry>(null);
  const st = useRef({ x: START[0], z: START[1], y: 0, vy: 0, falling: false, step: 0, acc: 0, path: [] as number[], done: false });
  useEffect(() => { st.current = { x: START[0], z: START[1], y: H(START[0], START[1]), vy: 0, falling: false, step: 0, acc: 0, path: [], done: false }; }, [seed, lr]);
  useFrame((_, dt) => {
    const s = st.current;
    if (running && !s.done) {
      s.acc += dt;
      if (s.falling) { s.vy -= 9.8 * dt; s.y += s.vy * dt; if (s.y < -6) { s.done = true; onStep({ step: s.step, loss: Infinity, state: "flew off" }); } }
      else if (s.acc > 0.16) {
        s.acc = 0; const [gx, gz] = grad(s.x, s.z); s.x -= lr * gx; s.z -= lr * gz; s.step++;
        s.path.push(s.x, H(Math.max(-R, Math.min(R, s.x)), Math.max(-R, Math.min(R, s.z))) + 0.03, s.z);
        if (Math.abs(s.x) > R || Math.abs(s.z) > R) { s.falling = true; s.y = H(Math.max(-R, Math.min(R, s.x)), Math.max(-R, Math.min(R, s.z))) + 0.4; s.vy = 2; }
        const l = loss(s.x, s.z), moved = Math.hypot(gx, gz) * lr;
        const state = s.falling ? "flew off" : moved < 2e-3 ? "settled" : s.step > 160 ? "never settled" : moved > 0.25 ? "bouncing" : "rolling";
        if (state === "never settled") s.done = true;
        if (state === "settled") s.done = true;
        onStep({ step: s.step, loss: l, state });
      }
    }
    const y = s.falling ? s.y : H(s.x, s.z) + 0.17;
    if (ball.current) ball.current.position.set(s.x, y, s.z);
    anchors.current.ball.set(s.x, y + 0.1, s.z);
    if (trail.current) { trail.current.setAttribute("position", new THREE.Float32BufferAttribute(s.path.length ? s.path : [s.x, y, s.z], 3)); trail.current.computeBoundingSphere(); }
    if (arrow.current && !s.falling) {
      const [gx, gz] = grad(s.x, s.z), d = new THREE.Vector3(-gx, 0, -gz); const len = d.length();
      arrow.current.visible = len > 0.02; if (len > 0.02) { d.normalize(); arrow.current.position.set(s.x, H(s.x, s.z) + 0.05, s.z); arrow.current.rotation.set(0, Math.atan2(d.x, d.z), 0); arrow.current.scale.setScalar(Math.min(1.2, 0.5 + len)); }
      anchors.current.grad.set(s.x + d.x * 0.7, H(s.x + d.x * 0.7, s.z + d.z * 0.7) + 0.1, s.z + d.z * 0.7);
    } else if (arrow.current) arrow.current.visible = false;
  });
  return (<group>
    <mesh ref={ball} castShadow><sphereGeometry args={[0.17, 40, 40]} /><meshStandardMaterial color="#C4416B" roughness={0.25} metalness={0.05} /></mesh>
    <line><bufferGeometry ref={trail} /><lineBasicMaterial color="#24301E" transparent opacity={0.75} /></line>
    <group ref={arrow}>
      <mesh position={[0, 0.06, 0.35]} rotation={[Math.PI / 2, 0, 0]}><cylinderGeometry args={[0.025, 0.025, 0.55, 10]} /><meshStandardMaterial color="#24301E" /></mesh>
      <mesh position={[0, 0.06, 0.68]} rotation={[Math.PI / 2, 0, 0]}><coneGeometry args={[0.08, 0.2, 14]} /><meshStandardMaterial color="#24301E" /></mesh>
    </group>
  </group>);
}

function Flag() {
  const [x, z] = MIN, y = H(x, z);
  return (<group position={[x, y, z]}>
    <mesh position={[0, 0.35, 0]} castShadow><cylinderGeometry args={[0.018, 0.018, 0.7, 8]} /><meshStandardMaterial color="#24301E" /></mesh>
    <mesh position={[0.15, 0.6, 0]} castShadow><boxGeometry args={[0.3, 0.18, 0.02]} /><meshStandardMaterial color="#5E8A3A" /></mesh>
    <mesh position={[0, 0.01, 0]} rotation={[-Math.PI / 2, 0, 0]}><ringGeometry args={[0.12, 0.17, 32]} /><meshBasicMaterial color="#24301E" /></mesh>
  </group>);
}

/* projects named 3D anchors into stage pixels every frame, and drives the leader lines */
function Projector({ anchors, out, offY = 0 }: { anchors: React.MutableRefObject<Anchors>; out: React.MutableRefObject<(p: Record<string, [number, number]>) => void>; offY?: number }) {
  const { camera, size } = useThree(), v = useMemo(() => new THREE.Vector3(), []);
  useFrame(() => { const r: Record<string, [number, number]> = {}; for (const k in anchors.current) { v.copy(anchors.current[k]); v.y += offY; v.project(camera); r[k] = [((v.x + 1) / 2) * size.width, ((1 - v.y) / 2) * size.height]; } out.current(r); });
  return null;
}

function CameraRig({ cmd }: { cmd: { n: number; kind: string } }) {
  const { camera, size } = useThree(), ctl = useThree((s) => s.controls) as unknown as OC | null;
  useEffect(() => { if (!ctl) return; const a = size.width / size.height; if (a < 1.05) { const off = camera.position.clone().sub(ctl.target).multiplyScalar(Math.min(1.9, 1.15 / a)); camera.position.copy(ctl.target.clone().add(off)); ctl.update(); } }, [ctl]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!cmd.n || !ctl) return;
    const off = camera.position.clone().sub(ctl.target);
    if (cmd.kind === "left" || cmd.kind === "right") off.applyAxisAngle(new THREE.Vector3(0, 1, 0), cmd.kind === "left" ? 0.4 : -0.4);
    if (cmd.kind === "in") off.multiplyScalar(0.85); if (cmd.kind === "out") off.multiplyScalar(1.18);
    if (cmd.kind === "reset") { off.set(13, 9.6, 13); ctl.target.set(0, -0.2, 0); }
    camera.position.copy(ctl.target.clone().add(off)); ctl.update();
  }, [cmd, camera, ctl]);
  useEffect(() => {
    if (!ctl) return;
    const key = (e: KeyboardEvent) => { const t = e.target as HTMLElement; if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return; const off = camera.position.clone().sub(ctl.target);
      if (e.key === "q" || e.key === "Q") off.applyAxisAngle(new THREE.Vector3(0, 1, 0), 0.25); else if (e.key === "e" || e.key === "E") off.applyAxisAngle(new THREE.Vector3(0, 1, 0), -0.25); else return;
      camera.position.copy(ctl.target.clone().add(off)); ctl.update(); };
    window.addEventListener("keydown", key); return () => window.removeEventListener("keydown", key);
  }, [camera, ctl]);
  return null;
}

/* ---------- the stage: canvas + callout cards + toolbar ---------- */
type Callout = { id: string; title: string; text: string; icon: React.ReactNode; cls: string };
const CALLOUTS: Callout[] = [
  { id: "surface", title: "Loss surface", text: "Every possible model, scored. Low ground means small error.", icon: <Ic.Mountain />, cls: "c-tl" },
  { id: "ball", title: "Your model", text: "The ball is your current weights. Training moves it.", icon: <Ic.Ball />, cls: "c-tc" },
  { id: "grad", title: "Gradient", text: "The arrow points downhill: the direction to nudge the weights.", icon: <Ic.Arrow />, cls: "c-tr" },
  { id: "min", title: "Best fit", text: "The lowest point on the map. The goal of training.", icon: <Ic.Flag />, cls: "c-br" },
  { id: "dip", title: "Local dip", text: "A false bottom. Gradient descent can get stuck here.", icon: <Ic.Target />, cls: "c-bl" },
];

export default function LossWorld({ onStatus, lr, running, seed, setRunning }: { onStatus?: (s: { step: number; loss: number; state: string }) => void; lr: number; running: boolean; seed: number; setRunning: (b: boolean) => void }) {
  const stage = useRef<HTMLDivElement>(null), svg = useRef<SVGSVGElement>(null), cards = useRef<Record<string, HTMLDivElement | null>>({});
  const anchors = useRef<Anchors>({ surface: new THREE.Vector3(-2.2, H(-2.2, -2.6) + 0.05, -2.6), ball: new THREE.Vector3(), grad: new THREE.Vector3(), min: new THREE.Vector3(MIN[0] + 0.15, H(MIN[0], MIN[1]) + 0.62, MIN[1]), dip: new THREE.Vector3(-1.7, H(-1.7, -1.2) + 0.04, -1.2) });
  const [cmd, setCmd] = useState({ n: 0, kind: "" });
  const draw = useRef<(p: Record<string, [number, number]>) => void>(() => {});
  draw.current = (p) => {
    const s = svg.current, st = stage.current; if (!s || !st) return; const sb = st.getBoundingClientRect();
    CALLOUTS.forEach((c) => {
      const el = cards.current[c.id], line = s.querySelector<SVGPolylineElement>(`[data-l="${c.id}"]`), dot = s.querySelector<SVGCircleElement>(`[data-d="${c.id}"]`), dot2 = s.querySelector<SVGCircleElement>(`[data-e="${c.id}"]`);
      if (!el || !line || !dot || !dot2 || !p[c.id]) return;
      const b = el.getBoundingClientRect(), [tx, ty] = p[c.id];
      const top = b.top - sb.top, bottom = b.bottom - sb.top, left = b.left - sb.left, right = b.right - sb.left;
      const ax = Math.max(left, Math.min(right, tx)), ay = Math.max(top, Math.min(bottom, ty));
      const ex = ax === tx && ay === ty ? ax : ax, ey = ay;
      line.setAttribute("points", `${ex},${ey} ${tx},${ty}`);
      dot.setAttribute("cx", String(tx)); dot.setAttribute("cy", String(ty)); dot2.setAttribute("cx", String(ex)); dot2.setAttribute("cy", String(ey));
    });
  };
  return (
    <div className="stage3d" ref={stage}>
      <Canvas shadows dpr={[1, 2]} camera={{ position: [13, 9.6, 13], fov: 24 }} gl={{ antialias: true, preserveDrawingBuffer: true }} aria-label="A 3D loss surface with a ball rolling downhill by gradient descent">
        <hemisphereLight args={["#fff7f2", "#d9c3bb", 0.9]} />
        <directionalLight position={[5, 9, 4]} intensity={1.6} castShadow shadow-mapSize={[2048, 2048]} shadow-camera-left={-6} shadow-camera-right={6} shadow-camera-top={6} shadow-camera-bottom={-6} shadow-bias={-0.0004} />
        <group position={[0, -0.2, 0]}>
          <Terrain /><Flag />
          <Descent lr={lr} running={running} seed={seed} anchors={anchors} onStep={(s) => onStatus?.(s)} />
        </group>
        <ContactShadows position={[0, BASE - 0.6, 0]} opacity={0.35} scale={14} blur={2.4} far={3} color="#5A2236" />
        <OrbitControls makeDefault enableDamping dampingFactor={0.08} minDistance={10} maxDistance={30} maxPolarAngle={1.35} target={[0, -0.2, 0]} />
        <CameraRig cmd={cmd} />
        <Projector anchors={anchors} out={draw} offY={-0.2} />
      </Canvas>
      <svg className="leaders" ref={svg} aria-hidden="true">
        {CALLOUTS.map((c) => <g key={c.id}><polyline data-l={c.id} fill="none" stroke="#24301E" strokeWidth={1.3} strokeDasharray="4 4" /><circle data-e={c.id} r={3.5} fill="#24301E" /><circle data-d={c.id} r={6} fill="#5E8A3A" stroke="#FFFAF6" strokeWidth={2.5} className="hot" /></g>)}
      </svg>
      {CALLOUTS.map((c) => <div key={c.id} ref={(el) => { cards.current[c.id] = el; }} className={`callout ${c.cls}`}><span className="callout-ic">{c.icon}</span><div><b>{c.title}</b><p>{c.text}</p></div></div>)}
      <div className="stage-hint">Drag to orbit · Q / E to rotate · Scroll to zoom</div>
      <div className="stage-tools" role="toolbar" aria-label="View and training controls">
        <button onClick={() => setCmd((c) => ({ n: c.n + 1, kind: "left" }))} aria-label="Rotate left"><Ic.Rotl size={18} /></button>
        <button onClick={() => setCmd((c) => ({ n: c.n + 1, kind: "right" }))} aria-label="Rotate right"><Ic.Rotr size={18} /></button>
        <span className="sep" />
        <button onClick={() => setCmd((c) => ({ n: c.n + 1, kind: "out" }))} aria-label="Zoom out"><Ic.Minus size={18} /></button>
        <button onClick={() => setCmd((c) => ({ n: c.n + 1, kind: "in" }))} aria-label="Zoom in"><Ic.Plus size={18} /></button>
        <button onClick={() => setCmd((c) => ({ n: c.n + 1, kind: "reset" }))} aria-label="Reset view"><Ic.Target size={18} /></button>
        <span className="sep" />
        <button className={running ? "on" : ""} onClick={() => setRunning(!running)} aria-label={running ? "Pause training" : "Play training"}>{running ? <Ic.Pause size={18} /> : <Ic.Run size={18} />}</button>
      </div>
    </div>
  );
}
