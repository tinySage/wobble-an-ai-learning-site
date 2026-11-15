import Link from "next/link";
import type { ReactNode } from "react";
import * as Ic from "./Icons";
import { Clock, ProgressCard } from "./Live";
import Backdrop from "./Backdrop";

export function Logo() {
  return (<Link href="/" className="logo" aria-label="Wobble home">
    <svg viewBox="0 0 64 64" width={58} height={58} aria-hidden="true"><rect x="4" y="4" width="56" height="56" rx="16" fill="#24301E" /><path d="M14 22c4 22 12 28 18 28s14-6 18-28" fill="none" stroke="#C9DB9E" strokeWidth="4.5" strokeLinecap="round" /><circle cx="21" cy="33" r="6.5" fill="#E8789A" /><circle cx="19" cy="31" r="2" fill="#FFD9E1" /></svg>
    <span className="logo-t"><b>WOBBLE</b><small>MAKE <i>·</i> BREAK <i>·</i> UNDERSTAND</small></span>
  </Link>);
}

export function Rail({ current }: { current?: string }) {
  const it = (href: string, label: string, icon: ReactNode, key: string) => <Link href={href} aria-label={label} title={label} aria-current={current === key ? "page" : undefined}>{icon}</Link>;
  return (<nav className="rail" aria-label="Main">
    <div className="rail-top">
      {it("/", "Home", <Ic.Home />, "home")}
      {it("/lessons/", "Map of all lessons", <Ic.Map />, "map")}
      {it("/lessons/linear-regression/", "Start learning", <Ic.Play />, "play")}
      {it("/about/", "About", <Ic.Info />, "about")}
    </div>
    <div className="rail-bottom">
      <span className="made">DESIGNED BY <b>MOMO</b></span>
      <a href="https://github.com/tinySage/wobble-an-ai-learning-site" aria-label="Source code on GitHub" title="GitHub"><Ic.Git /></a>
    </div>
  </nav>);
}

export default function Shell({ current, children, progress }: { current?: string; children: ReactNode; progress: { label: string; slugs: string[] } }) {
  return (<div className="shell">
    <Backdrop />
    <a href="#main" className="skip">Skip to content</a>
    <Rail current={current} />
    <div className="page">
      <header className="top">
        <Logo />
        <div className="top-right"><ProgressCard label={progress.label} slugs={progress.slugs} /><Clock /></div>
      </header>
      <main id="main" tabIndex={-1}>{children}</main>
      <footer className="foot">
        <div className="foot-l"><b>© 2026 Muktanan Mohapatra (Momo)</b><span>Wobble · learn machine learning by breaking it · MIT licence</span></div>
        <div className="foot-r">
          <a href="mailto:muktamohapatra672@gmail.com"><Ic.Mail size={18} />muktamohapatra672@gmail.com</a>
          <a href="https://github.com/tinySage"><Ic.Git size={18} />github.com/tinySage</a>
        </div>
      </footer>
    </div>
  </div>);
}
