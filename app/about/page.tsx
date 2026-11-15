import Chrome from "@/components/Chrome";
import Mascot from "@/components/Mascot";

export const metadata = { title: "About · Wobble" };
export default function About() {
  return (
    <Chrome current="about">
      <article className="about">
        <div className="about-hero"><Mascot size={140} talk label="Wobble" /><div><span className="kicker">ABOUT</span><h1>Why break things?</h1></div></div>
        <div className="window"><div className="win-bar"><span className="dots"><i /><i /><i /></span><span>README.TXT</span></div><div className="win-body prose">
          <p>Most machine learning tutorials show a model working. That teaches you very little, because working models all look the same. Broken ones are specific: the line that chases one outlier, the score that only looked good because the test set leaked, the network whose gradients quietly hit zero.</p>
          <p>So every level here is a small toy with a red button. You make it work, you break it on purpose, and the way it breaks is the lesson.</p>
          <h2>How the levels are made</h2>
          <p>Before a level is written, it is checked against several trusted sources: the scikit-learn docs, textbooks, course notes and the original papers. When they disagree, the Sources tab says so instead of picking a winner quietly.</p>
          <p>The models are written from scratch in TypeScript so they run in your browser, and the tests check them against scikit-learn, NumPy and SciPy. The datasets are small and made up so each toy stays fast. Numbers will shift on real data: use the levels for intuition, then test on your own problem.</p>
          <h2>Who made it</h2>
          <p>Wobble was built by Momo (<a href="https://github.com/tinySage">tinySage</a>), a computer science grad student, as a learning site people can actually use. The code is open under the MIT licence at <a href="https://github.com/tinySage/wobble-an-ai-learning-site">github.com/tinySage/wobble-an-ai-learning-site</a>.</p>
        </div></div>
      </article>
    </Chrome>
  );
}
