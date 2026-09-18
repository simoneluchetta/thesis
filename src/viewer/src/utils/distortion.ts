/**
 * One-parameter division-model radial distortion, for the court overlay and PnP.
 * Points are normalized by the sensor size, matching the Python solver.
 */
// https://en.wikipedia.org/wiki/Distortion_(optics)
// division model: https://www.robots.ox.ac.uk/~vgg/publications/2001/Fitzgibbon01b/fitzgibbon01b.pdf
// https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
// https://en.wikipedia.org/wiki/Pinhole_camera_model
import { type Pt, type Mat3, applyH, ransacHomography } from "./homography";

/** Principal point and scale in px, in the same pixel space as the points. */
export interface DivNorm { cx: number; cy: number; s: number; }

/** Observed px -> ideal pinhole px. r_u = r_d / (1 + k1 r_d^2). */
export function divisionUndistort(p: Pt, n: DivNorm, k1: number): Pt {
  if (k1 === 0) return p;
  const x = (p[0] - n.cx) / n.s, y = (p[1] - n.cy) / n.s;
  const f = 1 / (1 + k1 * (x * x + y * y));
  return [x * f * n.s + n.cx, y * f * n.s + n.cy];
}

/** Ideal pinhole px -> observed px: r_d = (1 - sqrt(1 - 4 k1 r_u^2)) / (2 k1 r_u).
 *  null past the model's max radius. Not fixed-point iteration: that diverges here. */
export function divisionDistort(p: Pt, n: DivNorm, k1: number): Pt | null {
  if (k1 === 0) return p;
  const xu = (p[0] - n.cx) / n.s, yu = (p[1] - n.cy) / n.s;
  const ru = Math.hypot(xu, yu);
  if (ru < 1e-9) return p;
  const disc = 1 - 4 * k1 * ru * ru;
  if (disc < 0) return null;                 // off-model: break the polyline
  const rd = (1 - Math.sqrt(disc)) / (2 * k1 * ru);
  const scale = rd / ru;
  return [xu * scale * n.s + n.cx, yu * scale * n.s + n.cy];
}

/** The best k1 found, with the homography and RANSAC score that go with it. */
interface K1Fit { k1: number; rms: number; nInliers: number; H: Mat3 | null; }

/** Estimate k1 by a coarse then fine sweep, keeping the most RANSAC inliers.
 *  src is world floor (X,Y); dst and threshPx are in sensor px.
 *  https://en.wikipedia.org/wiki/Random_sample_consensus
 */
export function estimateDivisionK1(
  src: Pt[],
  dst: Pt[],
  norm: DivNorm,
  threshPx: number,
  opts?: { lo?: number; hi?: number; step?: number; refine?: number; iters?: number; warmStart?: number },
): K1Fit {
  // Barrel distortion needs k1 < 0.
  // https://en.wikipedia.org/wiki/Distortion_(optics)
  // https://stackoverflow.com/questions/51895602/opencv-are-lens-distortion-coefficients-inverted-for-projectpoints
  const lo = opts?.lo ?? -0.60, hi = opts?.hi ?? 0.10, step = opts?.step ?? 0.025;
  const refine = opts?.refine ?? 0.005, iters = opts?.iters ?? 200;
  const minInliers = Math.max(4, Math.ceil(0.6 * src.length));

  function evalK1(k1: number): K1Fit {
    const dstU = dst.map(function (p) { return divisionUndistort(p, norm, k1); });
    const fit = ransacHomography(src, dstU, threshPx, iters);
    if (!fit || fit.nInliers < minInliers) return { k1, rms: Infinity, nInliers: fit?.nInliers ?? 0, H: null };
    return { k1, rms: fit.rms, nInliers: fit.nInliers, H: fit.H };
  }

  let best: K1Fit = { k1: 0, rms: Infinity, nInliers: 0, H: null };
  function consider(c: K1Fit) {
    if (c.nInliers > best.nInliers || (c.nInliers === best.nInliers && c.rms < best.rms)) best = c;
  }

  for (let k = lo; k <= hi + 1e-9; k += step) consider(evalK1(+k.toFixed(4)));
  // fine sweep around the coarse best
  const c0 = best.k1;
  for (let k = c0 - step; k <= c0 + step + 1e-9; k += refine) {
    if (k < lo - 1e-9 || k > hi + 1e-9) continue;
    consider(evalK1(+k.toFixed(4)));
  }
  consider(evalK1(0));                                   // always allow no distortion
  if (opts?.warmStart !== undefined) consider(evalK1(opts.warmStart));
  if (best.H === null) return { k1: 0, rms: Infinity, nInliers: 0, H: null };
  return best;
}

/** A K1Fit plus the distortion centre that won. */
interface CenterK1Fit { norm: DivNorm; k1: number; rms: number; nInliers: number; H: Mat3 | null; }

/** Estimate the distortion centre with k1, for the ceiling eyes: half of a shared
 *  optic, so the centre is not the image centre. Keeps the centred fit unless beaten. */
export function estimateDivisionCenterK1(
  src: Pt[],
  dst: Pt[],
  norm0: DivNorm,
  threshPx: number,
  opts?: { maxShiftXFrac?: number; maxShiftYFrac?: number; gridX?: number; gridY?: number; acceptFrac?: number; iters?: number },
): CenterK1Fit {
  const maxX = (opts?.maxShiftXFrac ?? 0.40) * norm0.s;
  const maxY = (opts?.maxShiftYFrac ?? 0.22) * norm0.s;
  const gx = opts?.gridX ?? 3, gy = opts?.gridY ?? 2;
  const acceptFrac = opts?.acceptFrac ?? 0.85;
  const iters = opts?.iters ?? 200;

  const base = estimateDivisionK1(src, dst, norm0, threshPx, { iters });
  if (!base.H) return { norm: norm0, k1: base.k1, rms: base.rms, nInliers: base.nInliers, H: base.H };

  // Coarse centre search, one homography per grid cell.
  // https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
  let bestNorm = norm0, bestRms = base.rms, bestInl = base.nInliers, moved = false;
  for (let i = -gx; i <= gx; i++) {
    for (let j = -gy; j <= gy; j++) {
      if (i === 0 && j === 0) continue;
      const norm: DivNorm = { cx: norm0.cx + (i / gx) * maxX, cy: norm0.cy + (j / gy) * maxY, s: norm0.s };
      const dstU = dst.map(function (p) { return divisionUndistort(p, norm, base.k1); });
      const fit = ransacHomography(src, dstU, threshPx, iters);
      if (!fit) continue;
      if (fit.nInliers > bestInl || (fit.nInliers === bestInl && fit.rms < bestRms)) {
        bestInl = fit.nInliers; bestRms = fit.rms; bestNorm = norm; moved = true;
      }
    }
  }
  if (!moved) return { norm: norm0, k1: base.k1, rms: base.rms, nInliers: base.nInliers, H: base.H };

  // Re-fit k1 at the winning centre, then compare against the centred fit.
  const refit = estimateDivisionK1(src, dst, bestNorm, threshPx, { iters, warmStart: base.k1 });
  const accept = refit.H && (refit.nInliers > base.nInliers || refit.rms <= acceptFrac * base.rms);
  if (accept) return { norm: bestNorm, k1: refit.k1, rms: refit.rms, nInliers: refit.nInliers, H: refit.H };
  return { norm: norm0, k1: base.k1, rms: base.rms, nInliers: base.nInliers, H: base.H };
}

/** World floor point -> observed sensor px: H maps world to ideal px, then distort. */
export function projectWorld(world: Pt, H: Mat3, norm: DivNorm, k1: number): Pt | null {
  return divisionDistort(applyH(H, world), norm, k1);
}
