/**
 * Snap a click onto the nearest court-line crossing, matching on line colour.
 * Foerstner junction operator: N x = b, N = sum g g^T, b = sum (g g^T) p.
 */
// https://en.wikipedia.org/wiki/Corner_detection
// https://en.wikipedia.org/wiki/Structure_tensor
// https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/getImageData
// https://developer.mozilla.org/en-US/docs/Web/API/ImageData/data

/** Which court line colour the click should snap to. */
export type LineColor = "white" | "blue" | "any";

/** Where the click ended up after snapping, and how sure the snap is. */
interface SnapResult {
  /** Snapped point, in the same pixel space as the input click. */
  xy: [number, number];
  /** 0 to 1. Higher means a clearer two-line junction. */
  roundness: number;
  /** How far the point moved from the raw click, px. */
  delta: number;
  snapped: boolean;
}

/** Per-pixel line response, roughly 0 to 255, for the target line colour.
 *  white: bright and unsaturated. blue: blue over red and green. any: plain luma. */
function lineResponse(r: number, g: number, b: number, mode: LineColor): number {
  if (mode === "white") {
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    const bright = (r + g + b) / 3;
    const sat = mx > 1e-3 ? (mx - mn) / mx : 0;
    return bright * (1 - sat); // the wood is orange, so saturated; the white lines pop
  }
  if (mode === "blue") {
    return Math.max(0, b - Math.max(r, g));
  }
  // Rec. 601 luma: https://en.wikipedia.org/wiki/Luma_(video)
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

/**
 * Snap the click (cx, cy) in the w*h RGBA thumbnail. radius is a half-window in px,
 * and the snap only happens when junction roundness exceeds minRoundness.
 */
export function snapToIntersection(
  rgba: Uint8ClampedArray,
  w: number,
  h: number,
  cx: number,
  cy: number,
  mode: LineColor = "any",
  radius = 14,
  minRoundness = 0.18,
): SnapResult {
  const x0 = Math.max(1, Math.floor(cx - radius));
  const x1 = Math.min(w - 2, Math.ceil(cx + radius));
  const y0 = Math.max(1, Math.floor(cy - radius));
  const y1 = Math.min(h - 2, Math.ceil(cy + radius));
  const none: SnapResult = { xy: [cx, cy], roundness: 0, delta: 0, snapped: false };
  if (x1 - x0 < 4 || y1 - y0 < 4) return none;

  function resp(x: number, y: number): number {
    const i = (y * w + x) * 4;
    return lineResponse(rgba[i], rgba[i + 1], rgba[i + 2], mode);
  }

  // Structure tensor N and N-weighted positions b, Gaussian-weighted by radius.
  let Nxx = 0, Nxy = 0, Nyy = 0, bx = 0, by = 0;
  const sigma2 = (radius * 0.7) ** 2;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const gx = (resp(x + 1, y) - resp(x - 1, y)) * 0.5;
      const gy = (resp(x, y + 1) - resp(x, y - 1)) * 0.5;
      const mag2 = gx * gx + gy * gy;
      if (mag2 < 4) continue; // flat: bare wood, or a line of the other colour
      const rw = Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma2));
      const axx = gx * gx * rw, axy = gx * gy * rw, ayy = gy * gy * rw;
      Nxx += axx; Nxy += axy; Nyy += ayy;
      bx += axx * x + axy * y;
      by += axy * x + ayy * y;
    }
  }
  const det = Nxx * Nyy - Nxy * Nxy;
  const tr = Nxx + Nyy;
  if (tr < 1e-3 || det <= 0) return none;
  // 4*det/tr^2 is 1 at a junction, 0 at a single straight edge.
  const roundness = (4 * det) / (tr * tr);
  const sx = (Nyy * bx - Nxy * by) / det;
  const sy = (-Nxy * bx + Nxx * by) / det;
  const delta = Math.hypot(sx - cx, sy - cy);
  const ok = roundness >= minRoundness && delta <= radius &&
             sx >= x0 && sx <= x1 && sy >= y0 && sy <= y1;
  return ok
    ? { xy: [sx, sy], roundness, delta, snapped: true }
    : { ...none, roundness };
}
