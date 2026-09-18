/**
 * Court lines on the floor plane (z=0), world coords: X long, Y short, origin centred.
 * Straight edges are sampled into many points so they can bend once distorted.
 */
// https://en.wikipedia.org/wiki/Basketball_court
// https://en.wikipedia.org/wiki/Volleyball
import { type Pt } from "./homography";

/** Court measurements in metres, as half-lengths from the centre. */
interface CourtDims {
  basketball: { Lhalf: number; Whalf: number; keyDepth: number; keyHW: number; ftR: number; ccR: number };
  volleyball: { Lhalf: number; Whalf: number; attackX: number };
}
/** One court line, already sampled into points. */
export interface CourtPolyline { court: "basketball" | "volleyball"; pts: Pt[]; }

/** Dimensions from the catalog's `courts` block, falling back to FIBA/FIVB.
 *  https://www.typescriptlang.org/docs/handbook/utility-types.html#recordkeys-type */
export function courtDimsFromCatalog(courts: Record<string, any> | undefined): CourtDims {
  const bb = courts?.basketball_fiba ?? {};
  const vb = courts?.volleyball_fivb ?? {};
  return {
    basketball: {
      Lhalf: bb.long_half_m ?? 14.0,
      Whalf: bb.short_half_m ?? 7.5,
      keyDepth: bb.key_depth_m ?? 5.8,
      keyHW: bb.key_half_width_m ?? 2.45,
      ftR: bb.free_throw_circle_radius_m ?? 1.8,
      ccR: bb.center_circle_radius_m ?? 1.8,
    },
    volleyball: {
      Lhalf: vb.long_half_m ?? 9.0,
      Whalf: vb.short_half_m ?? 4.5,
      attackX: vb.attack_line_x_m ?? 3.0,
    },
  };
}

// Straight segment a to b, sampled into n+1 points.
function seg(a: Pt, b: Pt, n = 16): Pt[] {
  const out: Pt[] = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
  }
  return out;
}
// Arc of radius r about (cx,cy) from angle a0 to a1, sampled into n+1 points.
function arc(cx: number, cy: number, r: number, a0: number, a1: number, n = 48): Pt[] {
  const out: Pt[] = [];
  for (let i = 0; i <= n; i++) {
    const a = a0 + (a1 - a0) * (i / n);
    out.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  return out;
}

/** Build all floor polylines in world (X,Y). */
export function buildCourtModel(d: CourtDims): CourtPolyline[] {
  const out: CourtPolyline[] = [];
  function bbL(pts: Pt[]) { return out.push({ court: "basketball", pts }); }
  function vbL(pts: Pt[]) { return out.push({ court: "volleyball", pts }); }

  const { Lhalf: L, Whalf: W, keyDepth, keyHW: kw, ftR, ccR } = d.basketball;
  const ftX = L - keyDepth; // free-throw line X

  // Outer rectangle (4 edges).
  bbL(seg([-L, -W], [L, -W], 40));
  bbL(seg([L, -W], [L, W], 24));
  bbL(seg([L, W], [-L, W], 40));
  bbL(seg([-L, W], [-L, -W], 24));
  // Center line + center circle.
  bbL(seg([0, -W], [0, W], 24));
  bbL(arc(0, 0, ccR, 0, 2 * Math.PI, 64));
  // Keys at both ends, free-throw lines, and semicircles facing the centre.
  for (const s of [1, -1]) {
    const base = s * L, ft = s * ftX;
    bbL(seg([base, kw], [ft, kw], 16));   // near lane line
    bbL(seg([base, -kw], [ft, -kw], 16)); // far lane line
    bbL(seg([ft, -kw], [ft, kw], 12));    // FT line
    // semicircle bulging toward the centre, apex at s*(ftX - ftR)
    const a0 = s === 1 ? Math.PI / 2 : -Math.PI / 2;
    const a1 = s === 1 ? (3 * Math.PI) / 2 : Math.PI / 2;
    bbL(arc(ft, 0, ftR, a0, a1, 32));
  }

  const { Lhalf: vL, Whalf: vW, attackX } = d.volleyball;
  // Volleyball outer rectangle.
  vbL(seg([-vL, -vW], [vL, -vW], 32));
  vbL(seg([vL, -vW], [vL, vW], 16));
  vbL(seg([vL, vW], [-vL, vW], 32));
  vbL(seg([-vL, vW], [-vL, -vW], 16));
  // Net/center line + attack lines.
  vbL(seg([0, -vW], [0, vW], 16));
  vbL(seg([attackX, -vW], [attackX, vW], 16));
  vbL(seg([-attackX, -vW], [-attackX, vW], 16));

  return out;
}
