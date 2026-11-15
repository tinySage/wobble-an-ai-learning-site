import katex from "katex";
import "katex/dist/katex.min.css";

/** A line wrapped in $$ ... $$ is typeset as a formula. Anything else is plain text. */
export default function Maths({ line }: { line: string }) {
  const m = line.match(/^\$\$([\s\S]+)\$\$$/);
  if (!m) return <p style={{ margin: "0 0 10px" }}>{line}</p>;
  const html = katex.renderToString(m[1], { displayMode: true, throwOnError: false, output: "htmlAndMathml" });
  return <div style={{ overflowX: "auto", padding: "4px 0" }} dangerouslySetInnerHTML={{ __html: html }} />;
}
