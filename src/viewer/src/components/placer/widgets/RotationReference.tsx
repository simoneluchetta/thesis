/**
 * Three small drawings of yaw, pitch and roll: Z-up world, Euler order ZXY,
 * camera looking down its local +Z.
 */
// https://developer.mozilla.org/en-US/docs/Web/SVG/Reference/Attribute/viewBox
// https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Paths#arcs
// https://en.wikipedia.org/wiki/Euler_angles

// viewBox of each of the three drawings.
const BOX = 70;
// Amber for the rotation arrow.
const ARROW = "#fc6";
// Grey for the dashed axes.
const AXIS = "#9d9d9d";
// Light grey for the camera icon.
const CAM = "#cccccc";

interface PanelProps {
  title: string;
  axisLabel: string;
  /** SVG fragment for the axes, cam icon and rotation arrow, in a BOX x BOX viewBox. */
  body: React.ReactNode;
}

/** Title, drawing and axis caption for one of the three rotations. */
function Panel({ title, axisLabel, body }: PanelProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", flex: 1 }}>
      <div style={{ fontSize: 11, color: "#9d9d9d" }}>{title}</div>
      <svg viewBox={`0 0 ${BOX} ${BOX}`} style={{ width: "100%", aspectRatio: "1 / 1", maxWidth: 80 }}>
        {body}
      </svg>
      <div style={{ fontSize: 10, color: "#6d6d6d" }}>around {axisLabel}</div>
    </div>
  );
}

/** Cam icon: a small triangular footprint with the apex (lens) at (cx, cy). */
function CamFootprint({ cx, cy, dir, size = 10 }: { cx: number; cy: number; dir: "up" | "right" | "out"; size?: number }) {
  // "out" is looking into the lens: a square, not a triangle.
  if (dir === "out") {
    return <rect x={cx - size / 2} y={cy - size / 2} width={size} height={size} fill={CAM} stroke="#000" strokeWidth={0.5} />;
  }
  const pts = dir === "up"
    ? `${cx},${cy - size} ${cx - size * 0.7},${cy + size * 0.5} ${cx + size * 0.7},${cy + size * 0.5}`
    : `${cx + size},${cy} ${cx - size * 0.5},${cy - size * 0.7} ${cx - size * 0.5},${cy + size * 0.7}`;
  return <polygon points={pts} fill={CAM} stroke="#000" strokeWidth={0.5} />;
}

/** The yaw, pitch and roll diagrams shown under the placer controls. */
export default function RotationReference() {
  const c = BOX / 2;
  return (
    <div style={{ display: "flex", gap: 4, padding: 4 }}>
      {/* Yaw: around Z, seen from above. The cam triangle points +Y. */}
      <Panel title="Yaw" axisLabel="Z (height)" body={
        <g>
          {/* Floor cross */}
          <line x1={5} y1={c} x2={BOX - 5} y2={c} stroke={AXIS} strokeWidth={0.5} strokeDasharray="2 2" />
          <line x1={c} y1={5} x2={c} y2={BOX - 5} stroke={AXIS} strokeWidth={0.5} strokeDasharray="2 2" />
          {/* Yaw arrow (counter-clockwise around center, seen from above) */}
          <path d={`M ${c + 18} ${c} A 18 18 0 1 0 ${c - 18} ${c}`} fill="none" stroke={ARROW} strokeWidth={1.5} />
          <polygon points={`${c - 22},${c} ${c - 14},${c - 4} ${c - 14},${c + 4}`} fill={ARROW} />
          {/* Cam looking up the page (+Y world) */}
          <CamFootprint cx={c} cy={c} dir="up" />
          {/* Axis label */}
          <text x={c + 2} y={9} fontSize={8} fill={AXIS}>+Y</text>
          <text x={BOX - 11} y={c - 2} fontSize={8} fill={AXIS}>+X</text>
        </g>
      } />

      {/* Pitch: around X, seen from the side (+X out of page), cam along +Y. */}
      <Panel title="Pitch" axisLabel="X" body={
        <g>
          <line x1={5} y1={c} x2={BOX - 5} y2={c} stroke={AXIS} strokeWidth={0.5} strokeDasharray="2 2" />
          <line x1={c} y1={5} x2={c} y2={BOX - 5} stroke={AXIS} strokeWidth={0.5} strokeDasharray="2 2" />
          {/* Pitch arrow (in the YZ plane, around X-out-of-page) */}
          <path d={`M ${c + 18} ${c} A 18 18 0 0 0 ${c} ${c - 18}`} fill="none" stroke={ARROW} strokeWidth={1.5} />
          <polygon points={`${c},${c - 22} ${c - 4},${c - 14} ${c + 4},${c - 14}`} fill={ARROW} />
          {/* Cam looking right (+Y world is to the right in this side view) */}
          <CamFootprint cx={c} cy={c} dir="right" />
          <text x={c + 2} y={9} fontSize={8} fill={AXIS}>+Z</text>
          <text x={BOX - 11} y={c - 2} fontSize={8} fill={AXIS}>+Y</text>
        </g>
      } />

      {/* Roll: around Y, the third stored angle, seen from behind the cam. */}
      <Panel title="Roll" axisLabel="Y (forward)" body={
        <g>
          <line x1={5} y1={c} x2={BOX - 5} y2={c} stroke={AXIS} strokeWidth={0.5} strokeDasharray="2 2" />
          <line x1={c} y1={5} x2={c} y2={BOX - 5} stroke={AXIS} strokeWidth={0.5} strokeDasharray="2 2" />
          {/* Roll arrow (in XZ plane, around Y-out-of-page) */}
          <path d={`M ${c + 18} ${c} A 18 18 0 1 1 ${c - 18} ${c}`} fill="none" stroke={ARROW} strokeWidth={1.5} />
          <polygon points={`${c - 22},${c} ${c - 14},${c - 4} ${c - 14},${c + 4}`} fill={ARROW} />
          {/* Cam seen from behind (looking along +Y away from us) */}
          <CamFootprint cx={c} cy={c} dir="out" />
          <text x={c + 2} y={9} fontSize={8} fill={AXIS}>+Z</text>
          <text x={BOX - 11} y={c - 2} fontSize={8} fill={AXIS}>+X</text>
        </g>
      } />
    </div>
  );
}
