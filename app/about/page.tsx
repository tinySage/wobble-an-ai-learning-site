import Chrome from "@/components/Chrome";

export const metadata = { title: "About · Wobble" };
export default function About() {
  return (
    <Chrome current="about">
      <article style={{ maxWidth: 680, padding: "24px 0 60px", display: "flex", flexDirection: "column", gap: 16 }}>
        <h1 style={{ fontSize: 64 }}>About</h1>
        <p>Wobble teaches machine learning by letting you break it. Every lesson has a model you can drag and a button that makes it fail, because the failures are where the ideas become clear.</p>
        <p>Everything runs in your browser. There are no accounts and no tracking. Your finished lessons are remembered in this browser only.</p>
        <h2 style={{ fontSize: 36, marginTop: 12 }}>How the lessons are made</h2>
        <p>Each lesson is checked against more than one trusted source before it is written, such as the scikit-learn documentation, Stanford CS229 notes, Google's Machine Learning Crash Course and the original papers. Where sources disagree, the Sources tab says so.</p>
        <p>The models are written in TypeScript so they can run in your browser, and the maths is tested against scikit-learn.</p>
        <h2 style={{ fontSize: 36, marginTop: 12 }}>What to keep in mind</h2>
        <p>The datasets are small and made up so a lesson stays fast and readable. Results such as where overfitting starts will shift with other data. Use the lessons to build intuition, then test on your own problem.</p>
        <p>Wobble is open source under the MIT licence: <a href="https://github.com/tinySage/wobble-an-ai-learning-site" style={{ textDecoration: "underline" }}>github.com/tinySage/wobble-an-ai-learning-site</a>.</p>
      </article>
    </Chrome>
  );
}
