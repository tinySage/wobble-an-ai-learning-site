import Shell from "@/components/shell/Shell";
import { liveInOrder } from "@/lib/lessons";

export const metadata = { title: "About · Wobble" };
export default function About() {
  return (
    <Shell current="about" progress={{ label: "ALL LESSONS", slugs: liveInOrder() }}>
      <article className="card about prose">
        <h1>About Wobble</h1>
        <p>Most machine learning tutorials show a model working. Working models all look alike. Broken ones are specific: the line that chases one outlier, the score that only looked good because the test set leaked, the network whose gradients quietly reach zero. So every lesson here is a small model you can handle, plus a red button that makes it fail on purpose.</p>
        <h2>How the lessons are made</h2>
        <p>Each lesson is checked against several trusted sources before it is written: library documentation, textbooks, course notes and the original papers. When they disagree, the Sources tab says so.</p>
        <p>The models are written from scratch in TypeScript so they run in your browser, and tests check them against scikit-learn, NumPy and SciPy. The datasets are small and made up so each lesson stays fast. Numbers will shift on real data: use the lessons for intuition, then test on your own problem.</p>
        <h2>Who made it</h2>
        <p>Wobble was designed and built by Muktanan Mohapatra (Momo), a computer science grad student. Questions, corrections or ideas are welcome at <a href="mailto:muktamohapatra672@gmail.com">muktamohapatra672@gmail.com</a>, or on GitHub at <a href="https://github.com/tinySage">github.com/tinySage</a>.</p>
        <p>The code is open source under the MIT licence: <a href="https://github.com/tinySage/wobble-an-ai-learning-site">github.com/tinySage/wobble-an-ai-learning-site</a>. © 2026 Muktanan Mohapatra.</p>
      </article>
    </Shell>
  );
}
