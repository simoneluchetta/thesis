"""Checks the floor geometry with a round trip, and holds the projection helpers.

The check: take a floor point whose position is known, project it into a camera, then take
that pixel back to the floor. The distance between start and end is the combined error of
the calibration, the distortion model and its inversion.

Most scripts in the project import the helpers here: the projection, the pixel-to-floor
inverse, and the fold detection below.

Run from src/:  uv run python _floor_homography_sanity.py
"""
from __future__ import annotations
import json
from pathlib import Path

import cv2
import numpy as np

from camera_frames import _placement_to_cam_from_world  # placer entry -> (R_w2c, t_w2c)

ROOT = Path(__file__).resolve().parent
PLACER = ROOT.parent / "placements" / "placed_cameras_redfox_repaired.json"
PNP = ROOT.parent / "diagnostics" / "court_pnp_cameras_redfox_repaired.json"
KPS = ROOT / "viewer" / "public" / "court_keypoints.json"


def full_opencv_to_KD(params):
    """Split the twelve-parameter camera blob into a matrix and a distortion vector.

    COLMAP and OpenCV happen to order the distortion terms identically, so the tail can be
    handed straight over without rearranging.
    """
    fx, fy, cx, cy = params[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=np.float64)
    dist = np.asarray(params[4:12], dtype=np.float64).reshape(1, 8)
    return K, dist


# The fold.
#
# A radial model maps a true radius to a distorted one, and should keep growing. Nothing in
# the fit enforces that, so a model fitted from clicks near the middle can turn over further
# out. Past the turning point a pixel has no true radius at all, and undistortPoints still
# returns a number for it. On the second session cam07, cam12 and cam13 fold at about
# 1900 px, with the corners at 2250. The repaired calibration in diagnostics/ fixes it.
#
# see: https://en.wikipedia.org/wiki/Distortion_(optics)
# and: https://stackoverflow.com/questions/22925306/undistortpoints-cannot-handle-lens-distortions
_FOLD_CACHE: dict = {}
_FOLD_WARNED: set = set()


def radial_max_radius_px(K, dist, _search=8.0, _n=200_001) -> float:
    """How far out this distortion model stays invertible. Infinite if it always does.

    Found by walking the radial curve outwards and looking for where it stops rising. A pole
    in the rational denominator bounds it too. Cached, since it depends only on the
    parameters.
    """
    K = np.asarray(K, dtype=np.float64)
    d = np.asarray(dist, dtype=np.float64).ravel()
    key = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), tuple(np.round(d, 12)))
    hit = _FOLD_CACHE.get(key)
    if hit is not None:
        return hit
    k1, k2 = (d[0], d[1]) if d.size > 1 else (0.0, 0.0)
    k3 = d[4] if d.size > 4 else 0.0
    k4, k5, k6 = (d[5], d[6], d[7]) if d.size >= 8 else (0.0, 0.0, 0.0)
    r = np.linspace(0.0, _search, _n)
    den = 1.0 + k4 * r**2 + k5 * r**4 + k6 * r**6
    with np.errstate(divide="ignore", invalid="ignore"):
        rd = r * (1.0 + k1 * r**2 + k2 * r**4 + k3 * r**6) / den
    bad = ~np.isfinite(rd)
    if bad.any():                      # the denominator has a root here, which bounds it too
        rd = rd[: int(np.argmax(bad))]
    turn = np.flatnonzero(np.diff(rd) <= 0.0)
    out = float(rd[turn[0]] * K[0, 0]) if turn.size else float(np.inf)
    _FOLD_CACHE[key] = out
    return out


# strong distortion folds back near the corners
# see: https://stackoverflow.com/questions/22925306/undistortpoints-cannot-handle-lens-distortions
def undistort_recoverable(uv, K, dist) -> np.ndarray:
    """Which of these pixels can be undistorted, and which are past the fold."""
    lim = radial_max_radius_px(K, dist)
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(lim):
        return np.ones(len(uv), dtype=bool)
    K = np.asarray(K, dtype=np.float64)
    return np.hypot(uv[:, 0] - K[0, 2], uv[:, 1] - K[1, 2]) <= lim


def warn_if_folding(name, K, dist, W=None, H=None) -> float:
    """Say something, once, if a camera's distortion folds within its own frame.

    A model that folds outside the sensor is ignored, since no pixel reaches that radius.
    When it folds inside the image the message gives the share of the frame affected.
    """
    lim = radial_max_radius_px(K, dist)
    if not np.isfinite(lim):
        return lim
    K = np.asarray(K, dtype=np.float64)
    if W and H:
        corner = max(np.hypot(u - K[0, 2], v - K[1, 2])
                     for u, v in [(0, 0), (W, 0), (0, H), (W, H)])
        if lim >= corner:
            return np.inf                      # folds beyond the sensor, so nothing is affected
        ys, xs = np.mgrid[0:H:8, 0:W:8]
        frac = float((np.hypot(xs - K[0, 2], ys - K[1, 2]) > lim).mean() * 100.0)
        msg = (f"[distortion] {name}: radial model FOLDS at r={lim:.0f} px (image corner "
               f"{corner:.0f} px) - {frac:.2f} % of the frame has no undistorted preimage. "
               f"Observations there are DROPPED. The repaired calibration in diagnostics/ avoids this.")
    else:
        msg = (f"[distortion] {name}: radial model FOLDS at r={lim:.0f} px; observations beyond "
               f"that radius are DROPPED. The repaired calibration in diagnostics/ avoids this.")
    if name not in _FOLD_WARNED:
        _FOLD_WARNED.add(name)
        print(msg)
    return lim


def project_world_to_pixel(W, R_w2c, t_w2c, K, dist):
    """Project a world point into the image."""
    W = np.asarray(W, dtype=np.float64).reshape(-1, 3)
    rvec, _ = cv2.Rodrigues(R_w2c)
    # projectPoints and the distortion model
    # see: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
    uv, _ = cv2.projectPoints(W, rvec, t_w2c.reshape(3, 1), K, dist)
    return uv.reshape(-1, 2)


def pixel_to_floor(uv, R_w2c, t_w2c, K, dist):
    """The floor point a pixel sees.

    Works because the floor is a known plane: undo the distortion, turn the pixel into a
    ray, rotate it into the world, and find where it crosses zero height.
    see: https://en.wikipedia.org/wiki/Line-plane_intersection
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    # Undistortion has no closed form, so it is solved iteratively. OpenCV stops after five
    # steps, too few for these wide lenses; 100 steps brings every camera under 0.02 px.
    crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-8)
    # see: https://stackoverflow.com/questions/44024133/opencv-undistortpoints-not-giving-the-exact-inverse-of-distortion-model
    norm = cv2.undistortPointsIter(uv, K, dist, None, None, crit).reshape(-1, 2)
    rays_cam = np.concatenate([norm, np.ones((len(norm), 1))], axis=1)  # (N,3)
    R_c2w = R_w2c.T
    C = -R_c2w @ t_w2c                       # camera center in world
    dirs = rays_cam @ R_c2w.T                 # world-frame ray directions (N,3)
    # walk along the ray until its height reaches zero
    s = -C[2] / dirs[:, 2]
    P = C[None, :] + s[:, None] * dirs
    P[:, 2] = 0.0
    in_front = s > 0                          # plane hit must be in front of cam
    # Past the fold there is no answer, whatever the iteration returned. Marking those invalid
    # here means every caller handles it already, since they all check this flag.
    valid = in_front & undistort_recoverable(uv.reshape(-1, 2), K, dist)
    return P, valid


def main():
    placer = json.loads(PLACER.read_text(encoding="utf-8"))
    pnp = json.loads(PNP.read_text(encoding="utf-8"))
    kdoc = json.loads(KPS.read_text(encoding="utf-8"))
    floor = np.array([k["world"] for k in kdoc["keypoints"]
                      if abs(k["world"][2]) < 1e-6], dtype=np.float64)
    print(f"{len(floor)} floor keypoints; cams in both placer&pnp:")

    print(f"\n{'cam':<8}{'n_inb':>6}{'mean_m':>9}{'med_m':>9}{'max_m':>9}{'fov_y':>7}")
    for name in sorted(set(placer) & set(pnp)):
        R_w2c, t_w2c = _placement_to_cam_from_world(placer[name])
        K, dist = full_opencv_to_KD(pnp[name]["params"])
        W, H = pnp[name]["width"], pnp[name]["height"]
        fov_y = np.degrees(2 * np.arctan2(0.5 * H, K[1, 1]))

        uv = project_world_to_pixel(floor, R_w2c, t_w2c, K, dist)
        # a point that projects outside the frame was never observed, so it proves nothing
        inb = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        Pb, front = pixel_to_floor(uv, R_w2c, t_w2c, K, dist)
        good = inb & front
        if good.sum() == 0:
            print(f"{name:<8}{0:>6}{'--':>9}{'--':>9}{'--':>9}{fov_y:>7.1f}")
            continue
        err = np.linalg.norm(Pb[good] - floor[good], axis=1)
        print(f"{name:<8}{int(good.sum()):>6}{err.mean():>9.4f}{np.median(err):>9.4f}"
              f"{err.max():>9.4f}{fov_y:>7.1f}")


if __name__ == "__main__":
    main()
