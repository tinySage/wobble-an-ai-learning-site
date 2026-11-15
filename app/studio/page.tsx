"use client";
import dynamic from "next/dynamic";
import { useEffect, useState } from "react";

/** Renders one lesson thumbnail for scripts/make_thumbs.py. Not linked from the site. */
const ThumbScene = dynamic(() => import("@/components/three/Thumb").then((m) => m.ThumbScene), { ssr: false });
export default function Studio() {
  const [slug, setSlug] = useState<string | null>(null);
  useEffect(() => { setSlug(new URLSearchParams(window.location.search).get("s") ?? "linear-regression"); document.body.style.background = "transparent"; }, []);
  return <div id="thumb" style={{ width: 480, height: 360 }}>{slug && <ThumbScene slug={slug} />}</div>;
}
