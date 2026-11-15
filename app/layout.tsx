import type { Metadata } from "next";
import type { ReactNode } from "react";
import "@fontsource/barlow/400.css";
import "@fontsource/barlow/500.css";
import "@fontsource/barlow/600.css";
import "@fontsource/barlow-condensed/500.css";
import "@fontsource/barlow-condensed/600.css";
import "@fontsource-variable/jetbrains-mono";
import "./globals.css";

export const metadata: Metadata = {
  authors: [{ name: "Muktanan Mohapatra", url: "https://github.com/tinySage" }],
  creator: "Muktanan Mohapatra",
  openGraph: { title: "Wobble · learn machine learning by breaking it", description: "46 interactive lessons and a 3D loss landscape. Make the model work, then press the red button.", images: ["/og.png"], type: "website" },
  title: "Wobble · learn machine learning by breaking it",
  description: "46 tiny playable levels, from linear regression to diffusion. Make the model work, then press the red button.",
};
export default function RootLayout({ children }: { children: ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
