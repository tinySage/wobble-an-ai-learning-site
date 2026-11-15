import type { ReactNode } from "react";
import Link from "next/link";
import Mascot from "@/components/Mascot";
import { firstLive } from "@/lib/lessons";

export default function Chrome({ current, children }: { current?: "lessons" | "about"; children: ReactNode }) {
  return (
    <>
      <a href="#main" className="skip">Skip to content</a>
      <div className="topbar"><div className="wrap topbar-in"><span>MACHINE LEARNING · LEARN IT BY BREAKING IT</span><span>made by <a href="https://github.com/tinySage">Momo</a> ♥</span></div></div>
      <div className="wrap">
        <header className="site">
          <Link href="/" className="brand" aria-label="Wobble home"><Mascot size={40} still label="" /><span className="wordmark">WOBBLE</span></Link>
          <nav className="main" aria-label="Main">
            <Link href="/lessons/" aria-current={current === "lessons" ? "page" : undefined}>Map</Link>
            <Link href="/about/" aria-current={current === "about" ? "page" : undefined}>About</Link>
            <Link href={`/lessons/${firstLive()}/`} className="px-btn primary small">Play ▸</Link>
          </nav>
        </header>
        <main id="main" tabIndex={-1}>{children}</main>
      </div>
      <footer className="site">
        <div className="wrap foot-in">
          <div className="foot-brand"><Mascot size={56} mood="happy" label="Wobble, waving" /><div><b>WOBBLE</b><span>Built by Momo, a grad student who broke a lot of models to make this. Open source, MIT.</span></div></div>
          <div className="foot-links"><Link href="/lessons/">Map</Link><Link href="/about/">About</Link><a href="https://github.com/tinySage/wobble-an-ai-learning-site">GitHub</a></div>
          <p className="foot-fine">Runs in your browser. No accounts, no tracking. Your stars are saved on this device only.</p>
        </div>
      </footer>
    </>
  );
}
