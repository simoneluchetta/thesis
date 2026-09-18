"""Turns each person's 2D joints into a 3D skeleton, in metres, using the calibrated cameras.

Every COCO joint is triangulated on its own across the cameras that saw it: try every pair
of views, keep the pair whose 3D point most other views agree with (reprojection within
sel_px pixels), then refit on the agreeing views and drop the rest. The scale comes from the
calibrated rig. Joints with fewer than min_views supporting views come back as None, and
every joint carries its reprojection rms (pixels) and the number of cameras behind it.

Reads the tracked people (adapter.DEFAULT_TRACKED or --tracked) and the calibration. Writes
a JSON with a 17-joint skeleton per person and timestamp. run_bodies.py runs it, and
fit_smplx.py calls triangulate_person_frame directly.

Run from src/:
  uv run python people/3D_People/triangulate.py --selftest
  uv run python people/3D_People/triangulate.py --out ../work/bodies/skeletons.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent
for _p in (str(SRC), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adapter import DEFAULT_TRACKED, iter_person_frames, load_cameras
from _floor_homography_sanity import project_world_to_pixel, undistort_recoverable

# COCO-17 skeleton bones (index pairs) used for the anthropometric sanity check.
BONES = {
    "shoulder_w": (5, 6), "hip_w": (11, 12),
    "upperarm_L": (5, 7), "upperarm_R": (6, 8),
    "forearm_L": (7, 9), "forearm_R": (8, 10),
    "thigh_L": (11, 13), "thigh_R": (12, 14),
    "shin_L": (13, 15), "shin_R": (14, 16),
}
# Rough adult ranges (m) for a plausibility readout (not hard gates).
BONE_PLAUSIBLE = {
    "shoulder_w": (0.30, 0.50), "hip_w": (0.20, 0.40),
    "upperarm_L": (0.22, 0.36), "upperarm_R": (0.22, 0.36),
    "forearm_L": (0.20, 0.34), "forearm_R": (0.20, 0.34),
    "thigh_L": (0.32, 0.52), "thigh_R": (0.32, 0.52),
    "shin_L": (0.32, 0.50), "shin_R": (0.32, 0.50),
}
_UNDISTORT_CRIT = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-8)


def _undistort_norm(uv, K, dist) -> np.ndarray:
    """Distorted pixel -> normalized camera coords (x', y') s.t. the cam-frame ray is (x',y',1)."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPointsIter(uv, K, dist, None, None, _UNDISTORT_CRIT).reshape(-1, 2)[0]


def _dlt(norms, Rs, ts) -> np.ndarray:
    """Linear triangulation from normalized observations (P = [R|t], K = I). Returns world XYZ."""
    A = []
    for (x, y), R, t in zip(norms, Rs, ts):
        P = np.hstack([R, t.reshape(3, 1)])          # 3x4 (world->cam, normalized image)
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    # direct linear transformation
    # see: https://en.wikipedia.org/wiki/Direct_linear_transformation
    # and: https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
    _, _, Vt = np.linalg.svd(np.asarray(A))
    X = Vt[-1]
    return X[:3] / X[3]


def _reproj_err(X, o) -> float:
    uv = project_world_to_pixel(np.asarray(X).reshape(1, 3), o["R"], o["t"], o["K"], o["dist"])[0]
    return float(np.hypot(uv[0] - o["uv"][0], uv[1] - o["uv"][1]))


def triangulate_joint(obs, sel_px: float, min_views: int):
    """Robust triangulation of one joint from a list of per-view obs dicts
    {uv,K,dist,R,t,cam}. RANSAC pair seed -> IRLS refit dropping >sel_px views.
    Returns {X, n_views, rms_px, cams} or None if < min_views support it."""
    # A pixel outside the camera's invertible distortion region cannot be undistorted, and cv2
    # returns junk instead of raising. Those views are dropped before counting support.
    obs = [o for o in obs if bool(undistort_recoverable(o["uv"], o["K"], o["dist"])[0])]
    n = len(obs)
    if n < min_views:
        return None
    for o in obs:
        o["norm"] = _undistort_norm(o["uv"], o["K"], o["dist"])

    def dlt_of(idx):
        return _dlt([obs[i]["norm"] for i in idx], [obs[i]["R"] for i in idx], [obs[i]["t"] for i in idx])

    # seed
    if n == 2:
        active = [0, 1]
        X = dlt_of(active)
    else:
        best = None
        for i, j in itertools.combinations(range(n), 2):
            X = dlt_of([i, j])
            inl = [k for k in range(n) if _reproj_err(X, obs[k]) <= sel_px]
            if best is None or len(inl) > len(best[1]):
                best = (X, inl)
        X, active = best
        if len(active) < min_views:                  # weak frame: keep the best min_views
            active = sorted(range(n), key=lambda k: _reproj_err(X, obs[k]))[:min_views]
        X = dlt_of(active)

    # IRLS refit
    for _ in range(10):
        new = [k for k in range(n) if _reproj_err(X, obs[k]) <= sel_px]
        if len(new) < min_views:
            new = sorted(range(n), key=lambda k: _reproj_err(X, obs[k]))[:min_views]
        if set(new) == set(active):
            break
        active = new
        X = dlt_of(active)

    res = [_reproj_err(X, obs[k]) for k in active]
    return {"X": [float(v) for v in X], "n_views": len(active),
            "rms_px": float(np.sqrt(np.mean(np.square(res)))),
            "cams": [obs[k]["cam"] for k in active]}


def triangulate_person_frame(pf, kp_thr: float, sel_px: float, min_views: int):
    """Triangulate all 17 COCO joints for one PersonFrame. Returns a dict of per-joint results."""
    joints = {}
    for j in range(17):
        obs = []
        for v in pf.views:
            u, vv, s = v.keypoints[j]
            if s < kp_thr:
                continue
            cam = v.camera
            obs.append({"uv": np.array([u, vv]), "K": cam.K, "dist": cam.dist,
                        "R": cam.R, "t": cam.t, "cam": v.cam})
        joints[j] = triangulate_joint(obs, sel_px, min_views)
    return joints


def _joint_xyz(joints, j):
    r = joints.get(j)
    return np.asarray(r["X"]) if r else None


def run(tracked_path, kp_thr, sel_px, min_views):
    cameras = load_cameras()
    tracked = json.loads(Path(tracked_path).read_text(encoding="utf-8"))
    out_frames = {}
    all_rms, bone_lens = [], {b: [] for b in BONES}
    ankle_z, n_joint_ok, n_joint_tot = [], 0, 0
    pf_count = 0
    for pf in iter_person_frames(tracked, cameras):
        pf_count += 1
        joints = triangulate_person_frame(pf, kp_thr, sel_px, min_views)
        skel = []
        for j in range(17):
            r = joints[j]
            n_joint_tot += 1
            if r:
                n_joint_ok += 1
                all_rms.append(r["rms_px"])
                skel.append({"j": j, "xyz": r["X"], "n_views": r["n_views"],
                             "rms_px": round(r["rms_px"], 2)})
            else:
                skel.append({"j": j, "xyz": None, "n_views": 0, "rms_px": None})
        out_frames.setdefault(str(pf.t), []).append(
            {"track_id": pf.track_id, "joints": skel})
        # stats
        for b, (a, c) in BONES.items():
            pa, pc = _joint_xyz(joints, a), _joint_xyz(joints, c)
            if pa is not None and pc is not None:
                bone_lens[b].append(float(np.linalg.norm(pa - pc)))
        for aj in (15, 16):
            p = _joint_xyz(joints, aj)
            if p is not None:
                ankle_z.append(float(p[2]))
    result = {"timestamps": sorted(int(t) for t in out_frames),
              "frames": out_frames,
              "params": {"kp_thr": kp_thr, "sel_px": sel_px, "min_views": min_views}}
    stats = {"person_frames": pf_count, "joint_fill": (n_joint_ok, n_joint_tot),
             "rms_px": all_rms, "bone_lens": bone_lens, "ankle_z": ankle_z}
    return result, stats


def _report(stats):
    rms = np.asarray(stats["rms_px"])
    ok, tot = stats["joint_fill"]
    print(f"[triangulate] person-frames: {stats['person_frames']}")
    print(f"[triangulate] joints triangulated: {ok}/{tot} ({100*ok/max(tot,1):.0f}%)")
    if rms.size:
        print(f"[triangulate] per-joint reprojection rms: median {np.median(rms):.1f}px  "
              f"p90 {np.percentile(rms,90):.1f}px")
    az = np.asarray(stats["ankle_z"])
    if az.size:
        print(f"[triangulate] ankle height z: median {np.median(az):.3f} m  "
              f"p10 {np.percentile(az,10):.3f}  p90 {np.percentile(az,90):.3f}  (expect ~0)")
    print("[triangulate] bone lengths (median m, plausible range):")
    for b in BONES:
        v = np.asarray(stats["bone_lens"][b])
        if not v.size:
            print(f"    {b:12s}: (none)"); continue
        lo, hi = BONE_PLAUSIBLE[b]
        flag = "ok" if lo <= np.median(v) <= hi else "**"
        print(f"    {b:12s}: {np.median(v):.3f}  (n={v.size}, std {v.std():.3f}) [{lo:.2f}-{hi:.2f}] {flag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracked", default=str(DEFAULT_TRACKED))
    ap.add_argument("--out", default=None, help="write the 3D skeletons JSON here")
    ap.add_argument("--kp_thr", type=float, default=2.0, help="min KeypointRCNN joint score to use a view")
    ap.add_argument("--sel_px", type=float, default=25.0, help="reprojection inlier gate (px) -- mirrors the tie-point solve")
    ap.add_argument("--min_views", type=int, default=2, help="min supporting views to accept a joint")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    result, stats = run(args.tracked, args.kp_thr, args.sel_px, args.min_views)
    _report(stats)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result), encoding="utf-8")
        print(f"[triangulate] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
