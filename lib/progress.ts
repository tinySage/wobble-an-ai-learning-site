const KEY = "wobble-progress";
let memory: string[] = [];
/** Finished lessons live in this browser only. Storage can be blocked, so every call falls back to memory. */
export function readDone(): string[] {
  try { const v = JSON.parse(window.localStorage.getItem(KEY) ?? "[]"); if (Array.isArray(v)) { memory = v.filter((s) => typeof s === "string"); } } catch { /* keep memory */ }
  return memory;
}
export function markDone(slug: string) {
  const next = [...new Set([...readDone(), slug])]; memory = next;
  try { window.localStorage.setItem(KEY, JSON.stringify(next)); } catch { /* private mode: progress lasts for this visit */ }
  window.dispatchEvent(new Event("wobble-progress"));
}
