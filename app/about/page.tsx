import Chrome from "@/components/Chrome";

export const metadata = { title: "About · Wobble" };
export default function About() {
  return (
    <Chrome current="about">
      <article style={{ maxWidth: 680, padding: "24px 0 60px", display: "flex", flexDirection: "column", gap: 16 }}>
        <h1 style={{ fontSize: 64 }}>About</h1>
        <p>Wobble teaches machine learning by letting you break it. Every lesson has a model you can drag and a button that makes it fail, because the failures are where the ideas become clear.</p>
        <p>Everything runs in your browser. There are no accounts and no tracking.</p>
      </article>
    </Chrome>
  );
}
