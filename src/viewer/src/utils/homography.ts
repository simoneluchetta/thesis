/**
 * Planar homography fit and reprojection, for live court-calibration feedback.
 * The keypoints lie on the floor plane, so floor-to-image is exactly a homography.
 */
// https://en.wikipedia.org/wiki/Homography_(computer_vision)
// https://en.wikipedia.org/wiki/Random_sample_consensus
// https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html

/** A 2D point [x,y], in world metres or image px depending on the caller. */
export type Pt = [number, number];
export type Mat3 = number[]; // row-major length 9

/** Solve A x = b for small square A, Gaussian elimination with partial pivoting.
 *  https://en.wikipedia.org/wiki/Gaussian_elimination
 *  https://en.wikipedia.org/wiki/Pivot_element */
function solveLinear(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  // augmented copy
  const M = A.map(function (row, i) { return [...row, b[i]]; });
  for (let col = 0; col < n; col++) {
    // pivot
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-12) return null;
    [M[col], M[piv]] = [M[piv], M[col]];
    const d = M[col][col];
    for (let c = col; c <= n; c++) M[col][c] /= d;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      if (f === 0) continue;
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map(function (row) { return row[n]; });
}

/** Hartley normalization: centroid to the origin, mean distance to sqrt(2).
 *  https://en.wikipedia.org/wiki/Eight-point_algorithm#Normalized_algorithm */
function normalization(pts: Pt[]): { T: Mat3; Tinv: Mat3; norm: Pt[] } {
  const n = pts.length;
  let cx = 0, cy = 0;
  for (const [x, y] of pts) { cx += x; cy += y; }
  cx /= n; cy /= n;
  let meanDist = 0;
  for (const [x, y] of pts) meanDist += Math.hypot(x - cx, y - cy);
  meanDist /= n;
  const s = meanDist > 1e-9 ? Math.SQRT2 / meanDist : 1;
  // T = [[s,0,-s*cx],[0,s,-s*cy],[0,0,1]]
  const T: Mat3 = [s, 0, -s * cx, 0, s, -s * cy, 0, 0, 1];
  const Tinv: Mat3 = [1 / s, 0, cx, 0, 1 / s, cy, 0, 0, 1];
  const norm = pts.map(function ([x, y]) { return [s * x - s * cx, s * y - s * cy] as Pt; });
  return { T, Tinv, norm };
}

// 3x3 matrix product, row-major.
function matMul3(A: Mat3, B: Mat3): Mat3 {
  const C = new Array(9).fill(0);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++) C[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
  return C;
}

/** Apply H to a point. */
export function applyH(H: Mat3, p: Pt): Pt {
  const w = H[6] * p[0] + H[7] * p[1] + H[8];
  return [
    (H[0] * p[0] + H[1] * p[1] + H[2]) / w,
    (H[3] * p[0] + H[4] * p[1] + H[5]) / w,
  ];
}

/** DLT homography src -> dst, 4 points or more, h33 fixed to 1, Hartley-normalized.
 *  https://en.wikipedia.org/wiki/Direct_linear_transformation */
function solveHomography(src: Pt[], dst: Pt[]): Mat3 | null {
  if (src.length < 4) return null;
  const sN = normalization(src);
  const dN = normalization(dst);
  const A: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < src.length; i++) {
    const [X, Y] = sN.norm[i];
    const [u, v] = dN.norm[i];
    A.push([X, Y, 1, 0, 0, 0, -u * X, -u * Y]); b.push(u);
    A.push([0, 0, 0, X, Y, 1, -v * X, -v * Y]); b.push(v);
  }
  let h: number[] | null;
  if (src.length === 4) {
    h = solveLinear(A, b);
  } else {
    // Normal equations A^T A h = A^T b (8x8).
    // https://en.wikipedia.org/wiki/Linear_least_squares
    const AtA: number[][] = Array.from({ length: 8 }, function () { return new Array(8).fill(0); });
    const Atb: number[] = new Array(8).fill(0);
    for (let r = 0; r < A.length; r++) {
      for (let i = 0; i < 8; i++) {
        Atb[i] += A[r][i] * b[r];
        for (let j = 0; j < 8; j++) AtA[i][j] += A[r][i] * A[r][j];
      }
    }
    h = solveLinear(AtA, Atb);
  }
  if (!h) return null;
  const Hn: Mat3 = [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
  // denormalize: H = Tdst^-1 * Hn * Tsrc
  return matMul3(matMul3(dN.Tinv, Hn), sN.T);
}

/** A fitted homography with its per-point residuals and its inlier set. */
interface HomographyFit {
  H: Mat3;
  /** Reprojection residual per input point, in dst px. */
  residuals: number[];
  inliers: boolean[];
  /** Over the inliers only. */
  rms: number;
  nInliers: number;
}

/** RANSAC homography src -> dst, robust to wrong-line clicks. threshPx is in dst px.
 *  https://en.wikipedia.org/wiki/Random_sample_consensus */
export function ransacHomography(
  src: Pt[],
  dst: Pt[],
  threshPx: number,
  iters = 800,
): HomographyFit | null {
  const n = src.length;
  if (n < 4) return null;
  function reproj(H: Mat3): number[] {
    return src.map(function (p, i) {
      const q = applyH(H, p);
      return Math.hypot(q[0] - dst[i][0], q[1] - dst[i][1]);
    });
  }

  // Fixed-seed PRNG, so the feedback doesn't jitter between renders.
  // https://en.wikipedia.org/wiki/Linear_congruential_generator
  let seed = 12345;
  function rand() {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  }

  let best: boolean[] | null = null;
  let bestCount = 3;
  if (n === 4) {
    const H = solveHomography(src, dst);
    if (H) return finalize(src, dst, H, threshPx);
    return null;
  }
  for (let it = 0; it < iters; it++) {
    // pick 4 distinct
    const idx = new Set<number>();
    while (idx.size < 4) idx.add(Math.floor(rand() * n));
    const ix = [...idx];
    const srcPick = ix.map(function (i) { return src[i]; });
    const dstPick = ix.map(function (i) { return dst[i]; });
    const H = solveHomography(srcPick, dstPick);
    if (!H) continue;
    const res = reproj(H);
    const inl = res.map(function (r) { return r < threshPx; });
    const c = inl.reduce(function (a, x) { return a + (x ? 1 : 0); }, 0);
    if (c > bestCount) { bestCount = c; best = inl; }
  }
  if (!best) {
    // fall back: least-squares on all points
    const H = solveHomography(src, dst);
    return H ? finalize(src, dst, H, threshPx) : null;
  }
  // refit on inliers
  const inSrc = src.filter(function (_, i) { return best![i]; });
  const inDst = dst.filter(function (_, i) { return best![i]; });
  const H = solveHomography(inSrc, inDst) ?? solveHomography(src, dst);
  if (!H) return null;
  return finalize(src, dst, H, threshPx);
}

/** Convex hull (Andrew's monotone chain), counter-clockwise, endpoint not repeated.
 *  Bounds the court overlay to the region the user clicked.
 *  https://en.wikipedia.org/wiki/Convex_hull_algorithms */
export function convexHull(pts: Pt[]): Pt[] {
  const p = pts.slice().sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
  if (p.length < 3) return p;
  function cross(o: Pt, a: Pt, b: Pt) {
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  }
  const lower: Pt[] = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
    lower.push(q);
  }
  const upper: Pt[] = [];
  for (let i = p.length - 1; i >= 0; i--) {
    const q = p[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
    upper.push(q);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

/** Grow a convex polygon from its centroid: each vertex moves out by `factor`
 *  (1 leaves it alone) plus `pad` in the polygon's own units. */
export function expandPolygon(poly: Pt[], factor: number, pad = 0): Pt[] {
  if (poly.length === 0) return poly;
  let cx = 0, cy = 0;
  for (const [x, y] of poly) { cx += x; cy += y; }
  cx /= poly.length; cy /= poly.length;
  return poly.map(function ([x, y]) {
    const dx = x - cx, dy = y - cy;
    const d = Math.hypot(dx, dy) || 1;
    const k = factor + pad / d;
    return [cx + dx * k, cy + dy * k] as Pt;
  });
}

/** Is p inside or on the CCW convex polygon `poly`? Interior points are left of
 *  every edge. https://en.wikipedia.org/wiki/Cross_product */
export function pointInConvexPoly(p: Pt, poly: Pt[]): boolean {
  if (poly.length < 3) return false;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    const cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    if (cross < -1e-9) return false;
  }
  return true;
}

// Packs H with its residuals, inliers and inlier rms.
function finalize(src: Pt[], dst: Pt[], H: Mat3, threshPx: number): HomographyFit {
  const residuals = src.map(function (p, i) {
    const q = applyH(H, p);
    return Math.hypot(q[0] - dst[i][0], q[1] - dst[i][1]);
  });
  const inliers = residuals.map(function (r) { return r < threshPx; });
  const inRes = residuals.filter(function (_, i) { return inliers[i]; });
  const rms = inRes.length
    ? Math.sqrt(inRes.reduce(function (a, r) { return a + r * r; }, 0) / inRes.length)
    : NaN;
  return { H, residuals, inliers, rms, nInliers: inliers.filter(Boolean).length };
}
