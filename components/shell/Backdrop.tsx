/** Faint circuit traces and pixel clusters behind everything, like a blueprint. */
export default function Backdrop() {
  const traces = ["M-20 330 H160 L240 250 H520 L600 170", "M1500 90 H1200 L1130 160 H900", "M-20 880 H90 L150 940 V1200", "M1500 640 H1380 L1320 700 V980 L1260 1040", "M380 1300 V1180 L440 1120 H760", "M980 -10 V60 L1040 120 H1180"];
  const ends = [[600, 170], [900, 160], [150, 1200], [1260, 1040], [760, 1120], [1180, 120], [240, 250], [1130, 160]];
  const pix = [[1240, 210], [70, 1080], [1380, 1210], [520, 70]];
  return (<svg className="backdrop" viewBox="0 0 1440 1300" preserveAspectRatio="xMidYMin slice" aria-hidden="true">
    <defs><pattern id="iso" width="64" height="37" patternUnits="userSpaceOnUse" patternTransform="skewX(-30)"><path d="M0 0H64M0 0V37" stroke="#24301E" strokeOpacity=".045" /></pattern></defs>
    <rect width="1440" height="1300" fill="url(#iso)" />
    {traces.map((d, i) => <path key={i} d={d} fill="none" stroke="#24301E" strokeOpacity=".13" strokeWidth="1.4" />)}
    {ends.map(([x, y], i) => <circle key={i} cx={x} cy={y} r="6" fill="none" stroke="#24301E" strokeOpacity=".2" strokeWidth="1.4" />)}
    {pix.map(([x, y], i) => <g key={i} fill="#24301E" fillOpacity=".09">{[[0, 0], [14, 0], [14, 14], [28, 14], [0, 28], [28, -14]].map(([a, b], k) => <rect key={k} x={x + a} y={y + b} width="12" height="12" />)}</g>)}
  </svg>);
}
