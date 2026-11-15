import type { ReactNode } from "react";
import Link from "next/link";
import { firstLive } from "@/lib/lessons";

export default function Chrome({ current, children }: { current?: "lessons" | "about"; children: ReactNode }) {
  return (
    <div className="wrap">
      <a href="#main" className="skip">Skip to content</a>
      <header className="site">
        <Link href="/" className="brand">Wobble</Link>
        <nav className="main" aria-label="Main">
          <Link href="/lessons/" aria-current={current === "lessons" ? "page" : undefined}>Lessons</Link>
          <Link href="/about/" aria-current={current === "about" ? "page" : undefined}>About</Link>
          <Link href={`/lessons/${firstLive()}/`} className="btn primary" style={{ minHeight: 44, marginLeft: 8 }}>Start learning</Link>
        </nav>
      </header>
      <main id="main" tabIndex={-1}>{children}</main>
      <footer className="site">Wobble · MIT licence · Educational, not professional advice.</footer>
    </div>
  );
}
