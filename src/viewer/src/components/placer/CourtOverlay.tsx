/**
 * SVG overlay of the FIBA basketball and FIVB volleyball court markings.
 * The court lies in the world X-Y plane, long axis along +X, centred on the origin.
 */
// https://developer.mozilla.org/en-US/docs/Web/SVG/Element/path
// https://www.w3.org/TR/SVG11/paths.html#PathDataEllipticalArcCommands
// https://en.wikipedia.org/wiki/Basketball_court

// Orange for the basketball lines, shared with the 3-D overlay.
export const BASKETBALL_COLOR = "#e7a06a";
// Green for the volleyball lines, shared with the 3-D overlay.
export const VOLLEYBALL_COLOR = "#9bd16a";

// The parent's world -> SVG mapping. `readonly` so callers cannot write the tuple.
// https://www.typescriptlang.org/docs/handbook/2/objects.html#readonly-properties
type W2S = (wx: number, wy: number) => readonly [number, number];

interface Props {
  basketball: boolean;
  volleyball: boolean;
  worldToSvg: W2S;
  metersToSvg: number;
}

/** The court markings as one SVG group, drawn inside the placer's top-down map. */
export default function CourtOverlay({ basketball, volleyball, worldToSvg, metersToSvg }: Props) {
  return (
    <g pointerEvents="none">
      {volleyball && <VolleyballCourt worldToSvg={worldToSvg} />}
      {basketball && <BasketballCourt worldToSvg={worldToSvg} metersToSvg={metersToSvg} />}
    </g>
  );
}

/** SVG x, y, width and height for the world rectangle between two corners. */
function rectFromWorld(w2s: W2S, x0: number, y0: number, x1: number, y1: number) {
  // worldToSvg flips Y, so the top-left SVG corner is (min wx, max wy).
  const [sx0, sy0] = w2s(Math.min(x0, x1), Math.max(y0, y1));
  const [sx1, sy1] = w2s(Math.max(x0, x1), Math.min(y0, y1));
  return { x: sx0, y: sy0, width: sx1 - sx0, height: sy1 - sy0 };
}

/** Volleyball markings: the 18x9 m outline, the centre line and the attack lines. */
function VolleyballCourt({ worldToSvg }: { worldToSvg: W2S }) {
  const HX = 9, HY = 4.5;
  const ATK = 3;
  const court = rectFromWorld(worldToSvg, -HX, -HY, HX, HY);
  const [c0x, c0y] = worldToSvg(0, -HY);
  const [c1x, c1y] = worldToSvg(0, HY);
  const [aL0x, aL0y] = worldToSvg(-ATK, -HY);
  const [aL1x, aL1y] = worldToSvg(-ATK, HY);
  const [aR0x, aR0y] = worldToSvg(ATK, -HY);
  const [aR1x, aR1y] = worldToSvg(ATK, HY);
  return (
    <g stroke={VOLLEYBALL_COLOR} fill="none" strokeWidth={1.5}>
      <rect {...court} />
      <line x1={c0x} y1={c0y} x2={c1x} y2={c1y} />
      <line x1={aL0x} y1={aL0y} x2={aL1x} y2={aL1y} />
      <line x1={aR0x} y1={aR0y} x2={aR1x} y2={aR1y} />
    </g>
  );
}

/** Basketball markings: the 28x15 m outline, keys, circles and 3-point lines. */
function BasketballCourt({ worldToSvg, metersToSvg }: { worldToSvg: W2S; metersToSvg: number }) {
  const HX = 14, HY = 7.5;
  const BX = HX - 1.575;            // basket centre, 1.575 m inside the baseline
  const KEY_W = 4.9, KEY_D = 5.8;   // the key
  const FT_LINE_X = HX - KEY_D;     // 8.2
  const ARC_R = 6.75;               // 3-point arc radius
  const ARC_TAN_Y = HY - 0.9;       // 6.6, the straight parts sit 0.9 m from the sideline
  // Where the straight part meets the arc, measured from the basket.
  const arcDx = Math.sqrt(ARC_R * ARC_R - ARC_TAN_Y * ARC_TAN_Y);  // ~1.4151

  const court = rectFromWorld(worldToSvg, -HX, -HY, HX, HY);
  const leftKey = rectFromWorld(worldToSvg, -HX, -KEY_W / 2, -FT_LINE_X, KEY_W / 2);
  const rightKey = rectFromWorld(worldToSvg, FT_LINE_X, -KEY_W / 2, HX, KEY_W / 2);

  const [centerLineTopX, centerLineTopY] = worldToSvg(0, HY);
  const [centerLineBotX, centerLineBotY] = worldToSvg(0, -HY);

  const [centerCircleX, centerCircleY] = worldToSvg(0, 0);
  const [leftFTCircleX, leftFTCircleY] = worldToSvg(-FT_LINE_X, 0);
  const [rightFTCircleX, rightFTCircleY] = worldToSvg(FT_LINE_X, 0);
  const r18 = 1.8 * metersToSvg;
  const rArc = ARC_R * metersToSvg;

  // Left 3-point line. The arc bulges into the court: a clockwise SVG sweep, flag 1.
  const lTanX = -BX + arcDx;
  const [lB0x, lB0y] = worldToSvg(-HX, ARC_TAN_Y);
  const [lT0x, lT0y] = worldToSvg(lTanX, ARC_TAN_Y);
  const [lT1x, lT1y] = worldToSvg(lTanX, -ARC_TAN_Y);
  const [lB1x, lB1y] = worldToSvg(-HX, -ARC_TAN_Y);
  const leftThree =
    `M ${lB0x} ${lB0y} L ${lT0x} ${lT0y}` +
    ` A ${rArc} ${rArc} 0 0 1 ${lT1x} ${lT1y}` +
    ` L ${lB1x} ${lB1y}`;

  // Right 3-point line: bulges toward -x, so the sweep flag is 0.
  const rTanX = BX - arcDx;
  const [rB0x, rB0y] = worldToSvg(HX, ARC_TAN_Y);
  const [rT0x, rT0y] = worldToSvg(rTanX, ARC_TAN_Y);
  const [rT1x, rT1y] = worldToSvg(rTanX, -ARC_TAN_Y);
  const [rB1x, rB1y] = worldToSvg(HX, -ARC_TAN_Y);
  const rightThree =
    `M ${rB0x} ${rB0y} L ${rT0x} ${rT0y}` +
    ` A ${rArc} ${rArc} 0 0 0 ${rT1x} ${rT1y}` +
    ` L ${rB1x} ${rB1y}`;

  return (
    <g stroke={BASKETBALL_COLOR} fill="none" strokeWidth={1.5}>
      <rect {...court} />
      <line x1={centerLineTopX} y1={centerLineTopY} x2={centerLineBotX} y2={centerLineBotY} />
      <circle cx={centerCircleX} cy={centerCircleY} r={r18} />
      <rect {...leftKey} />
      <rect {...rightKey} />
      <circle cx={leftFTCircleX} cy={leftFTCircleY} r={r18} />
      <circle cx={rightFTCircleX} cy={rightFTCircleY} r={r18} />
      <path d={leftThree} />
      <path d={rightThree} />
    </g>
  );
}
