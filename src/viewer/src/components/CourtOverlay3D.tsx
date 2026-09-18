/**
 * The court markings, drawn in the 3-D scene at floor level.
 * Same geometry as placer/CourtOverlay.tsx. Drei <Line> width is in screen px.
 */
// https://github.com/pmndrs/drei#line
// https://en.wikipedia.org/wiki/Basketball_court
import { useMemo } from "react";
import { Line } from "@react-three/drei";
import { BASKETBALL_COLOR, VOLLEYBALL_COLOR } from "./placer/CourtOverlay";

/** One point of a court line, in world metres. */
type Pt = [number, number, number];

interface Props {
  basketball: boolean;
  volleyball: boolean;
  /** Translation in world coords, before worldQuat. */
  anchorX?: number;
  anchorY?: number;
  anchorZ?: number;
  /** Rotation around the world up axis (Z), in radians. */
  yawRad?: number;
}

/** Court markings as 3-D line strips in the world X-Y plane at Z=0. */
export default function CourtOverlay3D({
  basketball, volleyball,
  anchorX = 0, anchorY = 0, anchorZ = 0, yawRad = 0,
}: Props) {
  const bb = useMemo(function () { return basketballSegments(); }, []);
  const vb = useMemo(function () { return volleyballSegments(); }, []);
  return (
    // Yaw is around local Z = world Z, so only the court moves.
    <group position={[anchorX, anchorY, anchorZ]} rotation={[0, 0, yawRad]}>
      {basketball && bb.map(function (pts, i) {
        return (
          <Line key={`b-${i}`} points={pts} color={BASKETBALL_COLOR} lineWidth={2} />
        );
      })}
      {volleyball && vb.map(function (pts, i) {
        return (
          <Line key={`v-${i}`} points={pts} color={VOLLEYBALL_COLOR} lineWidth={2} />
        );
      })}
    </group>
  );
}

/** n+1 points along an arc from a0 to a1 (radians). No short-way rule. */
function arcPoints(cx: number, cy: number, r: number, a0: number, a1: number, n: number): Pt[] {
  const pts: Pt[] = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const a = a0 + (a1 - a0) * t;
    pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a), 0]);
  }
  return pts;
}

/** FIBA basketball markings as line strips, 28x15 m centred on the origin. */
function basketballSegments(): Pt[][] {
  const HX = 14, HY = 7.5;
  const BX = HX - 1.575;           // basket center inset from baseline
  const KEY_W = 4.9, KEY_D = 5.8;
  const FT_LINE_X = HX - KEY_D;    // 8.2
  const ARC_R = 6.75;
  const ARC_TAN_Y = HY - 0.9;       // 6.6
  const arcDx = Math.sqrt(ARC_R * ARC_R - ARC_TAN_Y * ARC_TAN_Y);  // ~1.4151
  const lTanX = -BX + arcDx;
  const rTanX = BX - arcDx;

  // Tangent angles measured from each basket center.
  const ang0L = Math.atan2(ARC_TAN_Y, arcDx);     // ~+78 deg, court-interior side
  const ang0R = Math.atan2(ARC_TAN_Y, -arcDx);    // ~+102 deg, opposite side

  return [
    // Court outline (rectangle).
    [[-HX, -HY, 0], [HX, -HY, 0], [HX, HY, 0], [-HX, HY, 0], [-HX, -HY, 0]],
    // Halfway line.
    [[0, -HY, 0], [0, HY, 0]],
    // Center circle.
    arcPoints(0, 0, 1.8, 0, 2 * Math.PI, 48),
    // Left and right keys, 3-sided: the baseline is already the court outline.
    [[-HX, -KEY_W / 2, 0], [-FT_LINE_X, -KEY_W / 2, 0], [-FT_LINE_X, KEY_W / 2, 0], [-HX, KEY_W / 2, 0]],
    [[HX, -KEY_W / 2, 0], [FT_LINE_X, -KEY_W / 2, 0], [FT_LINE_X, KEY_W / 2, 0], [HX, KEY_W / 2, 0]],
    // Free-throw circles.
    arcPoints(-FT_LINE_X, 0, 1.8, 0, 2 * Math.PI, 48),
    arcPoints(FT_LINE_X, 0, 1.8, 0, 2 * Math.PI, 48),
    // Left 3-point line: baseline, straight at y=+6.6, arc bulging toward +x, back to baseline.
    [
      [-HX, ARC_TAN_Y, 0],
      [lTanX, ARC_TAN_Y, 0],
      ...arcPoints(-BX, 0, ARC_R, ang0L, -ang0L, 32),
      [-HX, -ARC_TAN_Y, 0],
    ],
    // Right 3-point line: the arc goes the long way so its midpoint faces the court.
    [
      [HX, ARC_TAN_Y, 0],
      [rTanX, ARC_TAN_Y, 0],
      ...arcPoints(BX, 0, ARC_R, ang0R, 2 * Math.PI - ang0R, 32),
      [HX, -ARC_TAN_Y, 0],
    ],
  ];
}

/** FIVB volleyball markings as line strips, 18x9 m centred on the origin. */
function volleyballSegments(): Pt[][] {
  const HX = 9, HY = 4.5;
  const ATK = 3;
  return [
    [[-HX, -HY, 0], [HX, -HY, 0], [HX, HY, 0], [-HX, HY, 0], [-HX, -HY, 0]],
    [[0, -HY, 0], [0, HY, 0]],
    [[-ATK, -HY, 0], [-ATK, HY, 0]],
    [[ATK, -HY, 0], [ATK, HY, 0]],
  ];
}
