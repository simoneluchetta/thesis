/**
 * A small top-down court diagram in plain SVG: world axes, corner names, and
 * optionally where a camera stands or where a keypoint sits.
 */
// https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/viewBox
// https://developer.mozilla.org/en-US/docs/Web/SVG/Element/polygon
import type React from "react";

/** Where a camera stands on the compass and which way it looks. */
export interface CameraPoseHint {
  /** Camera position in metres. Only X and Y are used. */
  position: [number, number];
  /** Forward direction projected onto X-Y. (0,0) just skips the arrow. */
  forward: [number, number];
  /** Short caption for the dot, usually the cam name. */
  label?: string;
}

interface CompassProps {
  size?: number;
  /** Which court silhouette to draw. The axis labels are the same either way. */
  variant?: "basketball" | "volleyball";
  /** Caption below the SVG. */
  caption?: string;
  /** Wrap the compass in a translucent card so it can sit on top of an image. */
  panel?: boolean;
  /** Draws an amber dot and forward arrow for the camera. */
  cameraPose?: CameraPoseHint | null;
  /** Draws every keypoint as a dot at its world (x, y). */
  keypoints?: { id: string; world: [number, number, number] }[];
  /** Id from `keypoints` to highlight in amber. */
  highlightKeypoint?: string | null;
}

// The SVG is VBOX x VBOX, so every coordinate below is in these units.
const VBOX = 100;
// Basketball orange, the same value as placer/CourtOverlay.
const COURT_COLOR_BB = "#e7a06a";
// Volleyball green, likewise.
const COURT_COLOR_VB = "#9bd16a";
// Same axis colours as the axesHelper in SceneCanvas: X red, Y green, Z blue.
const X_COLOR = "#ff6464";
const Y_COLOR = "#64ff80";
const Z_COLOR = "#7aa2ff";

/** Top-down court: long axis along +X, short along +Y, Z out of the page.
 *  Static: it shows the world convention, not the court anchor and yaw. */
export default function Compass({
  size = 88, variant = "basketball", caption, panel, cameraPose,
  keypoints, highlightKeypoint,
}: CompassProps) {
  const cx = VBOX / 2;
  const cy = VBOX / 2;

  // Basketball 28x15, volleyball 18x9.
  const ratio = variant === "volleyball" ? 18 / 9 : 28 / 15;
  // Sized to leave room around the edges for the arrows and labels.
  const courtW = 46;
  const courtH = courtW / ratio;
  const courtX = cx - courtW / 2;
  const courtY = cy - courtH / 2;
  const courtColor = variant === "volleyball" ? COURT_COLOR_VB : COURT_COLOR_BB;

  // Half-length of the court in metres, used to map world (x, y) into SVG coords.
  const courtHalfWorldX = variant === "volleyball" ? 9 : 14;
  const svgPerMeter = (courtW / 2) / courtHalfWorldX;

  // Arrow lengths past the court edge.
  const ext = 12;
  const xTip = courtX + courtW + ext;
  const yTip = courtY - ext;

  // Cameras often sit outside the court, so clamp the dot into the viewBox.
  const cam = cameraPose ? (function () {
    const [wx, wy] = cameraPose.position;
    const [fx, fy] = cameraPose.forward;
    const rawX = cx + wx * svgPerMeter;
    const rawY = cy - wy * svgPerMeter;
    const sx = Math.max(3, Math.min(VBOX - 3, rawX));
    const sy = Math.max(3, Math.min(VBOX - 3, rawY));
    const clamped = sx !== rawX || sy !== rawY;
    const fwdLen = Math.hypot(fx, fy);
    const arrowLen = 7;
    const tipX = fwdLen > 1e-6 ? sx + (fx / fwdLen) * arrowLen : sx;
    const tipY = fwdLen > 1e-6 ? sy - (fy / fwdLen) * arrowLen : sy;
    return { sx, sy, tipX, tipY, hasArrow: fwdLen > 1e-6, clamped };
  })() : null;

  const body = (
    <div style={{ display: "inline-flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
      <svg viewBox={`0 0 ${VBOX} ${VBOX}`} style={{ width: size, height: size, flexShrink: 0 }}>
        {/* Court rectangle. */}
        <rect x={courtX} y={courtY} width={courtW} height={courtH}
          fill="none" stroke={courtColor} strokeWidth={1.2} />
        {/* Centre line. */}
        <line x1={cx} y1={courtY} x2={cx} y2={courtY + courtH}
          stroke={courtColor} strokeWidth={0.6} opacity={0.7} />

        {/* +X arrow, the long axis. */}
        <line x1={cx} y1={cy} x2={xTip} y2={cy}
          stroke={X_COLOR} strokeWidth={1.6} strokeLinecap="round" />
        <polygon
          points={`${xTip},${cy} ${xTip - 5},${cy - 3.5} ${xTip - 5},${cy + 3.5}`}
          fill={X_COLOR}
        />
        <text x={xTip + 1} y={cy + 4} fontSize="10" fontWeight="700" fill={X_COLOR}>+X</text>

        {/* +Y arrow, the short axis. SVG y grows downward, so up is -y. */}
        <line x1={cx} y1={cy} x2={cx} y2={yTip}
          stroke={Y_COLOR} strokeWidth={1.6} strokeLinecap="round" />
        <polygon
          points={`${cx},${yTip} ${cx - 3.5},${yTip + 5} ${cx + 3.5},${yTip + 5}`}
          fill={Y_COLOR}
        />
        <text x={cx + 4} y={yTip - 1} fontSize="10" fontWeight="700" fill={Y_COLOR}>+Y</text>

        {/* Corner names, matching the keypoint catalog. */}
        <text x={courtX + courtW + 1.5} y={courtY - 1.5}
          fontSize="6.5" fontWeight="700" fill="#cccccc">NE</text>
        <text x={courtX - 1.5} y={courtY - 1.5}
          fontSize="6.5" fontWeight="700" fill="#cccccc" textAnchor="end">NW</text>
        <text x={courtX + courtW + 1.5} y={courtY + courtH + 6}
          fontSize="6.5" fontWeight="700" fill="#cccccc">SE</text>
        <text x={courtX - 1.5} y={courtY + courtH + 6}
          fontSize="6.5" fontWeight="700" fill="#cccccc" textAnchor="end">SW</text>

        {/* Z out of the page: the usual dot-in-circle symbol. */}
        <circle cx={cx} cy={cy} r={3} fill="none" stroke={Z_COLOR} strokeWidth={1.2} />
        <circle cx={cx} cy={cy} r={1.1} fill={Z_COLOR} />
        <text x={cx + 5.5} y={cy + 3} fontSize="7.5" fontWeight="700" fill={Z_COLOR}>Z</text>

        {/* Keypoint dots, drawn before the cam so the cam stays on top. */}
        {keypoints && keypoints.length > 0 && keypoints.map(function (kp) {
          const sx = cx + kp.world[0] * svgPerMeter;
          const sy = cy - kp.world[1] * svgPerMeter;
          if (sx < 2 || sx > VBOX - 2 || sy < 2 || sy > VBOX - 2) return null;
          const sel = highlightKeypoint === kp.id;
          return (
            <g key={kp.id} pointerEvents="none">
              {sel && (
                <circle cx={sx} cy={sy} r={5}
                  fill="none" stroke="#fc6" strokeWidth={1.2} opacity={0.85} />
              )}
              <circle cx={sx} cy={sy} r={sel ? 2.2 : 1.4}
                fill={sel ? "#fc6" : "#9d9d9d"}
                stroke={sel ? "#1f1f1f" : "none"} strokeWidth={0.4} />
            </g>
          );
        })}

        {/* Camera dot and forward arrow, drawn last so it sits on top. */}
        {cam && (
          <g pointerEvents="none">
            {cam.hasArrow && (
              <>
                <line x1={cam.sx} y1={cam.sy} x2={cam.tipX} y2={cam.tipY}
                  stroke="#fc6" strokeWidth={1.4} strokeLinecap="round" />
                <polygon
                  points={`${cam.tipX},${cam.tipY} ${cam.tipX - 2 * (cam.tipX - cam.sx) / 7 - 1.5 * (cam.tipY - cam.sy) / 7},${cam.tipY - 2 * (cam.tipY - cam.sy) / 7 + 1.5 * (cam.tipX - cam.sx) / 7} ${cam.tipX - 2 * (cam.tipX - cam.sx) / 7 + 1.5 * (cam.tipY - cam.sy) / 7},${cam.tipY - 2 * (cam.tipY - cam.sy) / 7 - 1.5 * (cam.tipX - cam.sx) / 7}`}
                  fill="#fc6"
                />
              </>
            )}
            <circle cx={cam.sx} cy={cam.sy} r={cam.clamped ? 2.2 : 2.8}
              fill="#fc6" stroke="#1f1f1f" strokeWidth={0.8} />
            {cameraPose?.label && (
              <text x={cam.sx + 4} y={cam.sy - 4}
                fontSize="7" fontWeight="700" fill="#fc6">{cameraPose.label}</text>
            )}
          </g>
        )}
      </svg>
      {caption && (
        <div style={captionStyle}>{caption}</div>
      )}
    </div>
  );
  if (!panel) return body;
  return <div style={panelStyle}>{body}</div>;
}

// Small grey caption under the drawing.
const captionStyle: React.CSSProperties = {
  fontSize: 9,
  color: "#9d9d9d",
  textTransform: "uppercase",
  letterSpacing: "0.08em",
  fontWeight: 600,
};

// Same translucent card as the TopDownMap legend.
const panelStyle: React.CSSProperties = {
  background: "rgba(24, 24, 24, 0.82)",
  backdropFilter: "blur(8px)",
  WebkitBackdropFilter: "blur(8px)",
  border: "1px solid #3c3c3c",
  borderRadius: 8,
  boxShadow: "0 6px 22px rgba(0, 0, 0, 0.5)",
  padding: "8px 10px 6px",
  pointerEvents: "none",
  userSelect: "none",
};
