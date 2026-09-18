"""Puts the detections of all cameras together and decides how many people are on the court.

Each camera has already placed the feet it found on the floor (detect.py). This script
proposes a set of people and keeps the set that explains the detections best: a person gains
score for every camera that saw them and loses score for every camera that covers their spot
but reported nobody. This is the probabilistic occupancy map (Fleuret 2008) applied to foot
points on a known floor. Duplicates and ghosts lose; two detections in a corner covered by
only two cameras still win, so the minimum camera count follows from the coverage.

Each detection has an error ellipse, not a circle: a pixel of error moves the floor point
mostly along the camera ray, and further when the view is shallow. The model parameters
(sigma0, p_det, lam_fa) are fitted to the clip and printed. Foot points are computed as
in fuse.py (ankle-height correction, weights, excluded cameras). The foreground term is
optional and off by default; the pipeline does not use it.

Run from src/ (run_people.py does this):
    uv run python -m people.fuse_global --in <foot_detections...> --out <people_fused.json> \\
        --placer <placed_cameras.json> --pnp <court_pnp_cameras.json>

In : one or more foot_detections*.json chunks (work/people/).
Out: fuse.py's schema plus per-person `n_cov` (cameras covering the spot) and `ll`
     (log-likelihood gain), both ignored downstream.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent           # src/people/
SRC = ROOT.parent                                 # src/
MT = SRC.parent                                   # repository root
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Shared with fuse.py, so both scripts measure foot points the same way.
from people.fuse import (EXCLUDE_CAMS, _appearance, _pool_template,  # noqa: E402
                         height_correct)
from people.reid import cosine_distance                              # noqa: E402

PERIMETER = ["cam01", "cam02", "cam03", "cam04", "cam05",
             "cam06", "cam07", "cam08", "cam12", "cam13"]


# Which cameras can see a person standing at a given floor spot.
class Coverage:
    """For each camera, the floor cells where a standing person would be visible to it.

    A camera covers a cell when the foot point (z=0) and the head point (z=body_h) both
    project inside the image, the cell is in front of the camera, and the foot ray rises at
    least `elev_min` degrees. This is the same gate detect.py uses, so a camera is never
    blamed for a view it was not allowed to report.
    """

    def __init__(self, poses, pnp, arena_xy, res, elev_min, body_h, margin_px):
        from camera_frames import _placement_to_cam_from_world
        from _floor_homography_sanity import full_opencv_to_KD

        ax, ay = arena_xy
        self.res = res
        self.xs = np.arange(-ax, ax + res, res)
        self.ys = np.arange(-ay, ay + res, res)
        gx, gy = np.meshgrid(self.xs, self.ys, indexing="ij")
        self.shape = gx.shape
        foot = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
        head = foot + np.array([0.0, 0.0, body_h])

        self.mask = {}
        self.cam_C = {}
        self.proj = {}          # cam -> (K, dist, rvec, twc, W, H), for silhouette rendering
        for cam in sorted(set(poses) & set(pnp)):
            if cam in EXCLUDE_CAMS:
                continue
            W, H = pnp[cam]["width"], pnp[cam]["height"]
            K, dist = full_opencv_to_KD(pnp[cam]["params"])
            Rwc, twc = _placement_to_cam_from_world(poses[cam])
            rvec, _ = cv2.Rodrigues(Rwc)
            ok = np.ones(len(foot), bool)
            for pts in (foot, head):
                # see: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
                # and: https://stackoverflow.com/questions/51895602/opencv-are-lens-distortion-coefficients-inverted-for-projectpoints
                uv = cv2.projectPoints(pts, rvec, twc.reshape(3, 1), K, dist)[0].reshape(-1, 2)
                zc = (Rwc @ pts.T + twc[:, None]).T[:, 2]
                ok &= (zc > 0)
                ok &= (uv[:, 0] >= -margin_px) & (uv[:, 0] < W + margin_px)
                ok &= (uv[:, 1] >= -margin_px) & (uv[:, 1] < H + margin_px)
            C = -Rwc.T @ twc
            self.cam_C[cam] = C
            self.proj[cam] = (K, dist, rvec, twc, W, H)
            horiz = np.linalg.norm(foot[:, :2] - C[None, :2], axis=1)
            elev = np.degrees(np.arctan2(C[2], np.maximum(horiz, 1e-6)))
            ok &= elev >= elev_min
            self.mask[cam] = ok.reshape(self.shape)
        self.cams = sorted(self.mask)

    def covering(self, xy):
        """Which cameras cover this spot."""
        i = int(round((xy[0] - self.xs[0]) / self.res))
        j = int(round((xy[1] - self.ys[0]) / self.res))
        if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
            return []
        return [c for c in self.cams if self.mask[c][i, j]]

    def count_grid(self):
        return np.sum([m for m in self.mask.values()], axis=0)


class Foreground:
    """Optional extra evidence from background subtraction: is something standing here?

    For a floor point y, a standing person (a cylinder of radius `body_r` and height `body_h`)
    is projected into each covering camera. The mean background distance inside that image box
    is high where a body stands and low over empty floor. It does not use the person detector,
    so it can support a person the detector missed.

    `calibrate` measures this value at known people (occupied) and at floor points far from
    anybody (empty) and stores log P(phi | occupied) - log P(phi | empty) per bin as a lookup
    table, so no foreground threshold is needed. Reads fg_<cam>.npz files from a folder.
    """

    def __init__(self, path, cov, body_r=0.30, body_h=1.80, n_bins=16, w_fg=1.0, max_llr=3.0):
        self.dist, self.tmap = {}, {}
        self.scale = None
        d = Path(path)
        for f in sorted(d.glob("fg_*.npz")):
            cam = f.stem[3:]
            if cam not in cov.proj:
                continue
            z = np.load(f)
            self.dist[cam] = z["dist"]
            self.tmap[cam] = {int(t): i for i, t in enumerate(z["timestamps"])}
            self.scale = int(z["scale"])
        self.cov = cov
        self.body_r, self.body_h = body_r, body_h
        self.n_bins, self.w_fg, self.max_llr = n_bins, w_fg, max_llr
        self.edges = None
        self.llr = None
        self.boxes = None
        self.cams = sorted(self.dist)

    def ready(self):
        return self.llr is not None and bool(self.cams)

    def calibrate(self, frames_people, arena_xy, clear_m, rng_seed=0):
        """Learn the occupied and empty distributions of phi from this clip.

        Samples phi at the given people (occupied) and at random floor points at least
        `clear_m` metres from everybody (empty), then stores the log ratio per bin.
        """
        occ, emp = [], []
        rng = np.random.default_rng(rng_seed)
        ax, ay = arena_xy
        for t, people in frames_people:
            P = [np.asarray(q["pos"][:2], float) for q in people]
            for y in P:
                for cam in self.cov.covering(y):
                    v = self.phi(cam, t, y)
                    if v is not None:
                        occ.append(v)
            for _ in range(max(1, len(P))):
                z = np.array([rng.uniform(-ax, ax), rng.uniform(-ay, ay)])
                if P and min(float(np.linalg.norm(z - q)) for q in P) < clear_m:
                    continue
                for cam in self.cov.covering(z):
                    v = self.phi(cam, t, z)
                    if v is not None:
                        emp.append(v)
        occ, emp = np.asarray(occ), np.asarray(emp)
        if occ.size < 50 or emp.size < 50:
            print(f"[fuse_global] foreground calibration too thin "
                  f"(occ {occ.size}, empty {emp.size}) - channel DISABLED")
            return False
        lo, hi = float(min(occ.min(), emp.min())), float(max(occ.max(), emp.max()))
        self.edges = np.linspace(lo, hi + 1e-6, self.n_bins + 1)
        ho = np.histogram(occ, self.edges)[0].astype(float) + 1.0
        he = np.histogram(emp, self.edges)[0].astype(float) + 1.0
        self.llr = np.clip(np.log(ho / ho.sum()) - np.log(he / he.sum()),
                           -self.max_llr, self.max_llr) * self.w_fg
        print(f"[fuse_global] foreground calibrated on {occ.size} occupied / {emp.size} empty "
              f"samples: median phi {np.median(occ):.1f} vs {np.median(emp):.1f}, "
              f"llr range [{self.llr.min():.2f}, {self.llr.max():.2f}] nats/camera")
        return True

    def build_grid(self, arena_xy, res):
        """Precompute, for each floor cell and camera, the image box of a standing person.

        The boxes never move, so they are computed once and reused for every frame.
        """
        ax, ay = arena_xy
        self.gx = np.arange(-ax, ax + res, res)
        self.gy = np.arange(-ay, ay + res, res)
        GX, GY = np.meshgrid(self.gx, self.gy, indexing="ij")
        self.gshape = GX.shape
        flat = np.stack([GX.ravel(), GY.ravel()], axis=1)
        r, hgt = self.body_r, self.body_h
        self.boxes, self.gcov = {}, {}
        for cam in self.cams:
            K, dist, rvec, twc, W, H = self.cov.proj[cam]
            corners = []
            for dx, dy, dz in ((-r, -r, 0), (r, -r, 0), (-r, r, 0), (r, r, 0),
                               (-r, -r, hgt), (r, -r, hgt), (-r, r, hgt), (r, r, hgt)):
                pts = np.stack([flat[:, 0] + dx, flat[:, 1] + dy,
                                np.full(len(flat), float(dz))], axis=1)
                corners.append(cv2.projectPoints(pts, rvec, twc.reshape(3, 1), K,
                                                 dist)[0].reshape(-1, 2))
            C = np.stack(corners, axis=1) / self.scale                       # (N, 8, 2)
            h, w = self.dist[cam].shape[1], self.dist[cam].shape[2]
            a = np.clip(np.floor(C[:, :, 0].min(1)), 0, w - 1).astype(np.int32)
            b = np.clip(np.ceil(C[:, :, 0].max(1)), 1, w).astype(np.int32)
            c = np.clip(np.floor(C[:, :, 1].min(1)), 0, h - 1).astype(np.int32)
            e = np.clip(np.ceil(C[:, :, 1].max(1)), 1, h).astype(np.int32)
            b = np.maximum(b, a + 1)
            e = np.maximum(e, c + 1)
            self.boxes[cam] = (a, b, c, e)
            gi = np.clip(((flat[:, 0] - self.cov.xs[0]) / self.cov.res).round().astype(int),
                         0, self.cov.shape[0] - 1)
            gj = np.clip(((flat[:, 1] - self.cov.ys[0]) / self.cov.res).round().astype(int),
                         0, self.cov.shape[1] - 1)
            self.gcov[cam] = self.cov.mask[cam][gi, gj]

    def occupancy(self, t):
        """Floor map of summed foreground evidence at time t, from the background plates alone.

        Peaks of this map can propose a person the detector never reported.
        """
        if self.boxes is None or not self.ready():
            return None
        acc = np.zeros(len(self.gx) * len(self.gy), np.float64)
        for cam in self.cams:
            i = self.tmap[cam].get(int(t))
            if i is None:
                continue
            I = cv2.integral(self.dist[cam][i].astype(np.float32)).astype(np.float64)
            a, b, c, e = self.boxes[cam]
            area = (b - a).astype(np.float64) * (e - c).astype(np.float64)
            mean = (I[e, b] - I[c, b] - I[e, a] + I[c, a]) / np.maximum(area, 1.0)
            bins = np.clip(np.searchsorted(self.edges, mean) - 1, 0, self.n_bins - 1)
            acc += np.where(self.gcov[cam], self.llr[bins], 0.0)
        return acc.reshape(self.gshape)

    def peaks(self, t, nms_r, min_ll):
        """Local maxima of the occupancy map above `min_ll`, used as extra person candidates."""
        occ = self.occupancy(t)
        if occ is None:
            return []
        res = float(self.gx[1] - self.gx[0])
        k = max(1, int(round(nms_r / res)))
        mx = cv2.dilate(occ.astype(np.float32), np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        sel = np.argwhere((occ >= mx) & (occ > min_ll))
        return [np.array([self.gx[i], self.gy[j]]) for i, j in sel]

    def _box(self, cam, t, y):
        """The box a standing person at this spot would occupy in this camera."""
        arr = self.dist.get(cam)
        if arr is None:
            return None
        i = self.tmap[cam].get(int(t))
        if i is None:
            return None
        K, dist, rvec, twc, W, H = self.cov.proj[cam]
        r, hgt = self.body_r, self.body_h
        box = np.array([[y[0] + dx, y[1] + dy, dz]
                        for dx, dy, dz in ((-r, -r, 0), (r, -r, 0), (-r, r, 0), (r, r, 0),
                                           (-r, -r, hgt), (r, -r, hgt), (-r, r, hgt), (r, r, hgt))])
        uv = cv2.projectPoints(box, rvec, twc.reshape(3, 1), K, dist)[0].reshape(-1, 2) / self.scale
        h, w = arr.shape[1], arr.shape[2]
        a = int(np.clip(np.floor(uv[:, 0].min()), 0, w - 1))
        b = int(np.clip(np.ceil(uv[:, 0].max()), 1, w))
        c = int(np.clip(np.floor(uv[:, 1].min()), 0, h - 1))
        e = int(np.clip(np.ceil(uv[:, 1].max()), 1, h))
        return (i, a, max(b, a + 1), c, max(e, c + 1))

    def phi(self, cam, t, y, claimed=None):
        """Mean background distance in the person's box, counting only pixels not yet claimed.

        Once a person is opened, `claim` marks its pixels as taken. A duplicate on top of it then
        finds nothing left to claim, the same explaining-away used for detections.
        """
        bx = self._box(cam, t, y)
        if bx is None:
            return None
        i, a, b, c, e = bx
        patch = self.dist[cam][i, c:e, a:b]
        if claimed is not None:
            free = ~claimed[cam][c:e, a:b]
            if free.sum() < max(4, 0.15 * free.size):
                return 0.0          # fully explained by somebody already there
            return float(patch[free].mean())
        return float(patch.mean())

    def claim(self, t, y, cams, claimed):
        for cam in cams:
            bx = self._box(cam, t, y)
            if bx is None:
                continue
            i, a, b, c, e = bx
            claimed[cam][c:e, a:b] = True

    def new_claims(self):
        return {cam: np.zeros(self.dist[cam].shape[1:], bool) for cam in self.cams}

    def bonus(self, t, y, cams, claimed=None):
        """Summed foreground log-likelihood ratio for a person at y over the given cameras."""
        if not self.ready():
            return 0.0
        tot = 0.0
        for cam in cams:
            v = self.phi(cam, t, y, claimed)
            if v is None:
                continue
            b = int(np.clip(np.searchsorted(self.edges, v) - 1, 0, self.n_bins - 1))
            tot += float(self.llr[b])
        return tot


# the probabilistic occupancy map of Fleuret et al. 2008, see https://doi.org/10.1109/TPAMI.2007.1174
class Model:
    """The error model and its three parameters: sigma0, p_det and lam_fa.

    A person at floor position y is seen by each covering camera with probability p_det, giving
    a foot point x ~ N(y, Sigma_i). False alarms arrive with density lam_fa per m^2 per camera.
    The gain of opening a person at y that claims the detections A (at most one per camera) is

        G(y, A) = sum_{i in A} [ log p_det - log lam_fa + 0.5 log|Lam_i| - log 2pi
                                 - 0.5 (x_i - y)' Lam_i (x_i - y) ]
                + sum_{c covers y, c not in A} log(1 - p_det)

    where Lam_i is the inverse covariance of detection i. A person is opened only if G > 0.

    The covariance is an ellipse, not a circle. A foot pixel error moves the floor point mostly
    along the camera ray, and the ray-to-floor intersection stretches it by 1/sin(elevation):

        sigma_perp  = sigma0 * (range / range_ref)            (across the ray, metres)
        sigma_along = sigma_perp / sin(elevation)             (along the ray)

    Box-bottom feet (no ankle keypoint) get their sigma multiplied by `box_inflate`.
    """

    def __init__(self, sigma0, p_det, lam_fa, sigma_lo=0.06, sigma_hi=2.50,
                 w_app=0.0, app_ref=0.35, range_ref=12.0, aniso=True, box_inflate=2.5):
        self.sigma0 = sigma0
        self.p_det = min(max(p_det, 0.05), 0.98)
        self.lam_fa = max(lam_fa, 1e-6)
        self.sigma_lo, self.sigma_hi = sigma_lo, sigma_hi
        self.w_app = w_app
        self.app_ref = app_ref
        self.range_ref = range_ref
        self.aniso = aniso
        self.box_inflate = box_inflate

    def axes(self, obs):
        """Standard deviation of each detection along the camera ray and across it, in metres."""
        rng = np.array([o["rng"] for o in obs])
        sel = np.sin(np.radians(np.array([o["elev"] for o in obs])))
        box = np.array([self.box_inflate if o["box"] else 1.0 for o in obs])
        perp = np.clip(self.sigma0 * (rng / self.range_ref) * box, self.sigma_lo, self.sigma_hi)
        along = np.clip(perp / np.maximum(sel, 0.05), self.sigma_lo, self.sigma_hi)             if self.aniso else perp
        return along, perp

    def info(self, obs):
        """Inverse covariance of each detection, oriented along its camera ray, and 0.5 log|Lam|."""
        along, perp = self.axes(obs)
        U = np.array([o["u"] for o in obs])                   # unit vector camera -> point
        V = np.stack([-U[:, 1], U[:, 0]], axis=1)             # perpendicular
        ia, ip = 1.0 / along ** 2, 1.0 / perp ** 2
        Lam = (ia[:, None, None] * U[:, :, None] * U[:, None, :]
               + ip[:, None, None] * V[:, :, None] * V[:, None, :])
        half_logdet = 0.5 * (np.log(ia) + np.log(ip))
        return Lam, half_logdet

    @property
    def miss(self):
        """Cost of a covering camera that did not see the person: log(1 - p_det)."""
        return float(np.log1p(-self.p_det))

    @property
    def base(self):
        """Constant part of a detection's gain: log p_det - log lam_fa - log 2pi."""
        return float(np.log(self.p_det) - np.log(self.lam_fa) - np.log(2.0 * np.pi))

    def gain_maha(self, maha, half_logdet, app_d=None):
        """Gain of assigning detections with the given squared Mahalanobis distances to a person."""
        g = self.base + half_logdet - 0.5 * maha
        if app_d is not None and self.w_app > 0:
            g = g - self.w_app * (app_d - self.app_ref)
        return g

    def __repr__(self):
        return (f"Model(sigma0={self.sigma0:.3f} m @ {self.range_ref:.0f} m, "
                f"p_det={self.p_det:.3f}, lam_fa={self.lam_fa:.4f}/m2, "
                f"miss={self.miss:.2f} nats, aniso={self.aniso}, w_app={self.w_app:.2f})")


def prepare(rows, arena_xy):
    """Turn raw detection rows into weighted floor points, the same way fuse.py does."""
    ax, ay = arena_xy
    obs = []
    for r in rows:
        if r["cam"] in EXCLUDE_CAMS:
            continue
        p = height_correct(r["world"][:2], r["cam_xy"], r["elev_deg"], r["foot_src"])
        if abs(p[0]) > ax or abs(p[1]) > ay:
            continue
        w = np.sin(np.radians(r["elev_deg"]))          # steeper ray -> lower ground error
        w *= 0.4 if r["foot_src"] == "box" else 1.0     # down-weight box-bottom fallback
        vec, backend = _appearance(r)
        d = np.asarray(p, float) - np.asarray(r["cam_xy"], float)
        rng = float(np.linalg.norm(d))
        u = d / rng if rng > 1e-6 else np.array([1.0, 0.0])
        obs.append({"xy": np.asarray(p, float), "cam": r["cam"], "w": float(max(w, 1e-3)),
                    "score": r.get("det_score", 1.0), "app": vec, "backend": backend, "row": r,
                    "u": u, "rng": rng, "elev": float(r["elev_deg"]),
                    "box": r["foot_src"] == "box"})
    return obs


def candidates(obs, arena_xy, res, blur, nms_r, seeds=None):
    """Floor positions where a person may be opened.

    Every detection adds a Gaussian of width `blur` metres to a floor grid, weighted by its
    ray precision. Local maxima at least `nms_r` metres apart become candidates.
    """
    ax, ay = arena_xy
    xs = np.arange(-ax, ax + res, res)
    ys = np.arange(-ay, ay + res, res)
    acc = np.zeros((len(xs), len(ys)), np.float32)
    if obs:
        pts = np.array([o["xy"] for o in obs])
        ws = np.array([o["w"] for o in obs])
        rad = int(np.ceil(3 * blur / res))
        for (px, py), w in zip(pts, ws):
            i0 = int(round((px + ax) / res))
            j0 = int(round((py + ay) / res))
            ia, ib = max(0, i0 - rad), min(len(xs), i0 + rad + 1)
            ja, jb = max(0, j0 - rad), min(len(ys), j0 + rad + 1)
            if ia >= ib or ja >= jb:
                continue
            dx = xs[ia:ib, None] - px
            dy = ys[None, ja:jb] - py
            acc[ia:ib, ja:jb] += w * np.exp(-(dx ** 2 + dy ** 2) / (2 * blur ** 2))
    k = int(np.ceil(nms_r / res))
    mx = cv2.dilate(acc, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    peaks = np.argwhere((acc >= mx) & (acc > 0))
    cand = [np.array([xs[i], ys[j]]) for i, j in peaks]
    # Every detection is also a candidate, and so is every person from the previous frame.
    cand += [o["xy"] for o in obs]
    if seeds is not None:
        cand += [np.asarray(s, float) for s in seeds]
    return cand


def _maha(obs_idx, Y, Lam, X):
    """Squared Mahalanobis distance between each listed detection and each candidate position."""
    D = X[obs_idx][:, None, :] - Y[None, :, :]                     # (n, d, 2)
    L = Lam[obs_idx]                                               # (n, 2, 2)
    return np.einsum("ndi,nij,ndj->nd", D, L, D)


def assign(obs, ys, cov_sets, model, cache, templates=None):
    """Given the people, decide which detection belongs to whom.

    Each person takes at most one detection per camera, so the problem splits by camera into
    one linear assignment each (Hungarian algorithm). Returns owner[i] (person index, or -1 for
    a false alarm) and the total assigned gain.
    """
    X, Lam, hld, by_cam = cache
    n = len(obs)
    owner = np.full(n, -1, int)
    if not ys or n == 0:
        return owner, 0.0
    Y = np.stack(ys)
    total = 0.0
    for cam, idxs in by_cam.items():
        cols = [s for s in range(len(ys)) if cam in cov_sets[s]]
        if not cols:
            continue
        idxs = np.asarray(idxs)
        maha = _maha(idxs, Y[cols], Lam, X)
        app_d = None
        if templates is not None and model.w_app > 0:
            app_d = np.full(maha.shape, model.app_ref, np.float64)
            for ai, i in enumerate(idxs):
                if obs[i]["app"] is None:
                    continue
                for bi, sidx in enumerate(cols):
                    t = templates[sidx]
                    if t is not None and t[1] == obs[i]["backend"]:
                        app_d[ai, bi] = cosine_distance(obs[i]["app"], t[0])
        g = model.gain_maha(maha, hld[idxs][:, None], app_d)
        # Zero-cost dummy columns let any detection stay unassigned (a false alarm).
        cost = np.concatenate([-g, np.zeros((len(idxs), len(idxs)))], axis=1)
        # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        rr, cc = linear_sum_assignment(cost)
        for ai, bi in zip(rr, cc):
            if bi < len(cols) and g[ai, bi] > 0:
                owner[idxs[ai]] = cols[bi]
                total += float(g[ai, bi])
    return owner, total


def fuse_position(members, cache):
    """Fuse the member detections into one position by inverse-covariance weighted least squares."""
    X, Lam, _, _ = cache
    idx = np.asarray(members)
    A = Lam[idx].sum(0)
    b = np.einsum("nij,nj->i", Lam[idx], X[idx])
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return X[idx].mean(0)


def person_ll(members, y, cov, model, obs, cache, template=None, fg=None, t=None):
    """Log-likelihood gain of one person: the gain of its detections, minus a miss cost for each
    covering camera that did not see it, plus the foreground bonus when that channel is on."""
    extra = fg.bonus(t, y, cov) if (fg is not None and fg.ready()) else 0.0
    if len(members) == 0:
        return float(len(cov)) * model.miss + extra
    X, Lam, hld, _ = cache
    idx = np.asarray(members)
    maha = _maha(idx, np.asarray(y)[None, :], Lam, X).ravel()
    app_d = None
    if template is not None and model.w_app > 0:
        app_d = np.array([cosine_distance(obs[i]["app"], template[0])
                          if (obs[i]["app"] is not None and obs[i]["backend"] == template[1])
                          else model.app_ref for i in members])
    g = float(model.gain_maha(maha, hld[idx], app_d).sum())
    seen = {obs[i]["cam"] for i in members}
    return g + float(len({c for c in cov if c not in seen})) * model.miss + extra


def _total_ll(obs, ys, cov_sets, model, cache, owner, templates=None, fg=None, t=None):
    return sum(person_ll(np.flatnonzero(owner == s).tolist(), ys[s], cov_sets[s], model, obs,
                         cache, None if templates is None else templates[s], fg, t)
               for s in range(len(ys)))


def split_move(obs, ys, cov_sets, model, args, cov, cache, owner, fg=None, t=None):
    """Try to split one person into two (off by default, --max_splits 0).

    For each person, its members are cut at the widest gap along their principal axis. The
    split is accepted only if the total objective improves after a full re-assignment.

    Returns (ys, cov_sets, owner, n_accepted).
    """
    X = cache[0]
    accepted = 0
    for _ in range(args.max_splits):
        base = _total_ll(obs, ys, cov_sets, model, cache, owner, None, fg, t)
        best = None
        for s in range(len(ys)):
            mem = np.flatnonzero(owner == s)
            if mem.size < 2 * args.min_cams:
                continue
            P = X[mem]
            u = P - P.mean(0)
            w, vec = np.linalg.eigh(u.T @ u)
            proj = u @ vec[:, int(np.argmax(w))]
            order = np.argsort(proj)
            gaps = np.diff(proj[order])
            if not gaps.size:
                continue
            cut = int(np.argmax(gaps)) + 1
            lo, hi = mem[order[:cut]].tolist(), mem[order[cut:]].tolist()
            if len({obs[i]["cam"] for i in lo}) < args.min_cams:
                continue
            if len({obs[i]["cam"] for i in hi}) < args.min_cams:
                continue
            y_lo, y_hi = fuse_position(lo, cache), fuse_position(hi, cache)
            if float(np.linalg.norm(y_lo - y_hi)) < args.min_sep:
                continue
            trial = [y for k, y in enumerate(ys) if k != s] + [y_lo, y_hi]
            tcov = [c for k, c in enumerate(cov_sets) if k != s] + \
                   [cov.covering(y_lo), cov.covering(y_hi)]
            towner, _ = assign(obs, trial, tcov, model, cache)
            gain = _total_ll(obs, trial, tcov, model, cache, towner, None, fg, t) - base
            if gain > args.split_ll and (best is None or gain > best[0]):
                best = (gain, trial, tcov, towner)
        if best is None:
            break
        _, ys, cov_sets, owner = best
        accepted += 1
    return ys, cov_sets, owner, accepted


def open_move(obs, ys, cov_sets, model, args, cov, cache, owner, fg, t):
    """Open extra people at peaks of the foreground occupancy map (only with --foreground).

    A peak is only a proposal. It is accepted if the detection-based objective improves after
    a full re-assignment, so the foreground evidence is not counted twice.

    Returns (ys, cov_sets, owner, n_opened).
    """
    if fg is None or not fg.ready() or fg.boxes is None:
        return ys, cov_sets, owner, 0
    peaks = fg.peaks(t, args.nms_r, args.fg_peak_ll)
    if not peaks:
        return ys, cov_sets, owner, 0
    opened = 0
    for _ in range(args.fg_max_open):
        base = _total_ll(obs, ys, cov_sets, model, cache, owner)
        best = None
        for y in peaks:
            if ys and min(float(np.linalg.norm(y - q)) for q in ys) < args.min_sep:
                continue
            cvs = cov.covering(y)
            if len(cvs) < args.min_cams:
                continue
            trial = list(ys) + [np.asarray(y, float)]
            tcov = list(cov_sets) + [cvs]
            towner, _ = assign(obs, trial, tcov, model, cache)
            if len({obs[i]["cam"] for i in np.flatnonzero(towner == len(ys))}) < args.min_cams:
                continue
            gain = _total_ll(obs, trial, tcov, model, cache, towner) - base
            if gain > args.fg_open_ll and (best is None or gain > best[0]):
                best = (gain, trial, tcov, towner)
        if best is None:
            break
        _, ys, cov_sets, owner = best
        opened += 1
    return ys, cov_sets, owner, opened


def fuse_timestamp_global(rows, cov, model, args, seeds=None, fg=None, t=None):
    """Fuse one timestamp: candidates -> greedy opening -> EM -> closing, opening and split moves."""
    obs = prepare(rows, (args.arena_x, args.arena_y))
    if not obs:
        return [], []
    X = np.stack([o["xy"] for o in obs])
    Lam, hld = model.info(obs)
    by_cam = {}
    for i, o in enumerate(obs):
        by_cam.setdefault(o["cam"], []).append(i)
    cache = (X, Lam, hld, by_cam)

    fg_seeds = (fg.peaks(t, args.nms_r, args.fg_peak_ll)
                if (fg is not None and fg.ready() and fg.boxes is not None) else [])
    cand = candidates(obs, (args.arena_x, args.arena_y), args.grid_res,
                      args.blur, args.nms_r, list(seeds or []) + fg_seeds)
    if not cand:
        return [], []
    cand_cov = [cov.covering(c) for c in cand]

    # Greedy opening: the best candidate takes its detections, so a duplicate on top of a
    # real person finds nothing left to claim.
    ys, cov_sets = [], []
    used = np.zeros(len(obs), bool)
    claims = fg.new_claims() if (fg is not None and fg.ready()) else None
    for _ in range(args.max_people):
        best, best_ll, best_members = None, args.open_ll, None
        for c, cvs in zip(cand, cand_cov):
            if len(cvs) < args.min_cams:
                continue
            if ys and min(np.linalg.norm(c - y) for y in ys) < args.min_sep:
                continue
            members = []
            for camn in cvs:
                pool = [i for i in by_cam.get(camn, []) if not used[i]]
                if not pool:
                    continue
                pool = np.asarray(pool)
                g = model.gain_maha(_maha(pool, c[None, :], Lam, X).ravel(), hld[pool])
                bi = int(np.argmax(g))
                if g[bi] > 0:
                    members.append(int(pool[bi]))
            if len({obs[i]["cam"] for i in members}) < args.min_cams:
                continue
            ll = person_ll(members, c, cvs, model, obs, cache)
            if claims is not None:
                ll += fg.bonus(t, c, cvs, claims)
            if ll > best_ll:
                best, best_ll, best_members = c, ll, members
        if best is None:
            break
        pos = fuse_position(best_members, cache)
        ys.append(pos)
        cov_sets.append(cov.covering(pos))
        used[np.asarray(best_members)] = True
        if claims is not None:
            fg.claim(t, pos, cov_sets[-1], claims)

    if not ys:
        return [], []

    # EM: re-assign the detections exactly, then refit each position
    owner = None
    for _ in range(args.em_iters):
        templates = None
        if model.w_app > 0 and owner is not None:
            templates = [_tpl(np.flatnonzero(owner == s).tolist(), obs) for s in range(len(ys))]
        owner, _ = assign(obs, ys, cov_sets, model, cache, templates)
        moved = 0.0
        for s in range(len(ys)):
            mem = np.flatnonzero(owner == s)
            if not mem.size:
                continue
            new = fuse_position(mem.tolist(), cache)
            moved = max(moved, float(np.linalg.norm(new - ys[s])))
            ys[s] = new
            cov_sets[s] = cov.covering(new)
        if moved < args.em_tol:
            break

    # Closing: drop the person with the lowest gain while it is at or below keep_ll, or one seen
    # by too few cameras. Each round re-assigns the freed detections. Appearance templates are
    # recomputed every round when w_app > 0.
    def _templates(own):
        if model.w_app <= 0:
            return None
        return [_tpl(np.flatnonzero(own == s).tolist(), obs) for s in range(len(ys))]

    for _ in range(args.max_people):
        owner, _ = assign(obs, ys, cov_sets, model, cache, _templates(owner))
        lls, ncams = [], []
        tps = _templates(owner)
        for s in range(len(ys)):
            mem = np.flatnonzero(owner == s).tolist()
            lls.append(person_ll(mem, ys[s], cov_sets[s], model, obs, cache,
                                 None if tps is None else tps[s], fg, t))
            ncams.append(len({obs[i]["cam"] for i in mem}))
        drop = None
        if min(lls) <= args.keep_ll:
            drop = int(np.argmin(lls))
        elif min(ncams) < args.min_cams:
            drop = int(np.argmin(ncams))
        if drop is None:
            break
        ys.pop(drop)
        cov_sets.pop(drop)
        if not ys:
            return [], []

    owner, _ = assign(obs, ys, cov_sets, model, cache, _templates(owner))
    if args.fg_max_open:
        ys, cov_sets, owner, _ = open_move(obs, ys, cov_sets, model, args, cov, cache,
                                           owner, fg, t)
    if args.max_splits:
        ys, cov_sets, owner, _ = split_move(obs, ys, cov_sets, model, args, cov, cache,
                                            owner, fg, t)
    people = []
    for s in range(len(ys)):
        mem = np.flatnonzero(owner == s).tolist()
        cams = {obs[i]["cam"] for i in mem}
        if len(cams) < args.min_cams:
            continue
        pts = X[np.asarray(mem)]
        spread = float(np.linalg.norm(pts - ys[s], axis=1).max())
        views = []
        for i in mem:
            rr = obs[i]["row"]
            views.append({"cam": rr["cam"], "bbox": rr.get("bbox"),
                          "keypoints": rr.get("keypoints"), "foot_uv": rr.get("foot_uv"),
                          "det_score": rr.get("det_score")})
        pobj = {"pos": [float(ys[s][0]), float(ys[s][1]), 0.0],
                "n_cams": len(cams), "spread_m": round(spread, 3),
                "cams": sorted(cams), "views": views,
                "n_cov": len(cov_sets[s]),
                "ll": round(person_ll(mem, ys[s], cov_sets[s], model, obs, cache,
                                     None, fg, t), 2)}
        tpl, backend, dim = _pool_template(mem, obs)
        if tpl is not None:
            pobj["appearance"] = [round(float(x), 4) for x in tpl]
            pobj["appearance_backend"] = backend
            pobj["appearance_dim"] = dim
        people.append(pobj)
    people.sort(key=lambda q: (-q["n_cams"], q["pos"][0]))
    for n, q in enumerate(people):
        q["id"] = n
    return people, _stats(obs, ys, cov_sets, owner, model, cache)


def _tpl(mem, obs):
    t, b, _ = _pool_template(mem, obs)
    return None if t is None else (t, b)


def _stats(obs, ys, cov_sets, owner, model, cache):
    """Statistics used by `calibrate` to refit sigma0, p_det and lam_fa.

    Returns [squared Mahalanobis residuals, assigned cameras, covering cameras, unassigned
    detections, total detections].
    """
    X, Lam, _, _ = cache
    maha, n_assigned, n_cov = [], 0, 0
    for s in range(len(ys)):
        mem = np.flatnonzero(owner == s)
        if not mem.size:
            continue
        maha.append(_maha(mem, np.asarray(ys[s])[None, :], Lam, X).ravel())
        n_assigned += len({obs[i]["cam"] for i in mem})
        n_cov += len(cov_sets[s])
    m = np.concatenate(maha) if maha else np.zeros(0)
    return [m, n_assigned, n_cov, int((owner < 0).sum()), len(obs)]


def _mk(args, sigma0, p_det, lam_fa, w_app=0.0):
    return Model(sigma0, p_det, lam_fa, sigma_lo=args.sigma_lo, sigma_hi=args.sigma_hi,
                 w_app=w_app, range_ref=args.range_ref, aniso=not args.no_aniso,
                 box_inflate=args.box_inflate)


def calibrate(chunks, cov, args):
    """Fit sigma0, p_det and lam_fa to this clip by EM on a sample of frames.

    If the covariance is right, the squared Mahalanobis residuals follow a chi-square with 2
    degrees of freedom, whose median is 2 ln 2. Comparing the data's median with that value
    gives a scale correction for sigma0. p_det is the fraction of covering cameras that got a
    detection, lam_fa the unassigned detections per m^2 per camera per frame. `--sigma_scale`
    widens the result, since hard assignment fits it a little tight.

    see: https://en.wikipedia.org/wiki/Mahalanobis_distance
    and: https://en.wikipedia.org/wiki/Chi-squared_distribution
    """
    keys = [(f, t) for f in chunks for t in chunks[f]]
    keys.sort(key=lambda kt: int(kt[1]))
    if not keys:
        raise SystemExit("[fuse_global] no timestamps to calibrate on")
    pick = np.linspace(0, len(keys) - 1, min(args.calib_frames, len(keys))).round().astype(int)
    sample = [(int(keys[i][1]), chunks[keys[i][0]][keys[i][1]])
              for i in sorted(set(pick.tolist()))]

    model = _mk(args, args.sigma0, args.p_det, args.lam_fa)
    area = 4.0 * args.arena_x * args.arena_y
    ncam = max(1, len(cov.cams))
    print(f"[fuse_global] calibrating on {len(sample)} frames, prior {model}")
    for it in range(args.calib_iters):
        M, A, C, U, N = [], 0, 0, 0, 0
        for _t, rows in sample:
            _, st = fuse_timestamp_global(rows, cov, model, args, None, None, _t)
            if not st:
                continue
            M.append(st[0]); A += st[1]; C += st[2]; U += st[3]; N += st[4]
        m = np.concatenate(M) if M else np.zeros(0)
        if not m.size:
            print("[fuse_global] calibration produced no people - keeping the prior")
            break
        scale = float(np.sqrt(max(np.median(m), 1e-6) / (2.0 * np.log(2.0))))
        new = _mk(args, model.sigma0 * scale, A / max(C, 1),
                  U / max(area * ncam * len(sample), 1e-6))
        print(f"  iter {it}: {new}  (assigned {A}/{C} covering cams, {U}/{N} unexplained, "
              f"maha median {np.median(m):.2f} -> scale x{scale:.3f})")
        converged = abs(scale - 1.0) < 0.02 and abs(new.p_det - model.p_det) < 5e-3
        model = new
        if converged:
            break
    if args.sigma_scale != 1.0:
        model.sigma0 *= args.sigma_scale
        print(f"[fuse_global] --sigma_scale {args.sigma_scale} applied -> sigma0 {model.sigma0:.3f} m")
    model.w_app = args.w_app
    return model, sample


def load_chunks(paths):
    chunks = {}
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        chunks[str(p)] = d
        print(f"[fuse_global] loaded {Path(p).name}: {len(d)} timestamps")
    return chunks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", nargs="+", required=True,
                    help="one or more foot_detections*.json (chunks are merged by timestamp)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--placer", default=str(MT / "placements" / "placed_cameras_redfox_repaired.json"))
    ap.add_argument("--pnp", default=str(MT / "diagnostics" / "court_pnp_cameras_redfox_repaired.json"))
    ap.add_argument("--arena_x", type=float, default=15.5)
    ap.add_argument("--arena_y", type=float, default=8.5)
    # coverage
    ap.add_argument("--cov_res", type=float, default=0.25, help="coverage grid resolution (m)")
    ap.add_argument("--elev_min", type=float, default=10.0, help="must match detect.py's gate")
    ap.add_argument("--body_h", type=float, default=1.75,
                    help="standing height, used for the coverage head test and the foreground "
                         "silhouette box")
    ap.add_argument("--margin_px", type=float, default=0.0, help="slack on the image border")
    # candidates
    ap.add_argument("--grid_res", type=float, default=0.10)
    ap.add_argument("--blur", type=float, default=0.35, help="occupancy deposit width (m)")
    ap.add_argument("--nms_r", type=float, default=0.5, help="occupancy peak separation (m)")
    ap.add_argument("--min_sep", type=float, default=0.45, help="closest two people may be opened")
    ap.add_argument("--max_people", type=int, default=30)
    ap.add_argument("--min_cams", type=int, default=2, help="hard floor: 1 camera cannot localise")
    # model (priors only - `calibrate` overwrites all three unless --no_calib)
    ap.add_argument("--sigma0", type=float, default=0.35,
                    help="prior for the ACROSS-ray floor error at --range_ref (m); refitted")
    ap.add_argument("--sigma_scale", type=float, default=1.5,
                    help="multiply the fitted sigma0 (the fit tends to underestimate it; 1.5 gave "
                         "the best result in the tests)")
    ap.add_argument("--range_ref", type=float, default=12.0,
                    help="range at which sigma0 is quoted; error scales linearly with range")
    ap.add_argument("--box_inflate", type=float, default=2.5,
                    help="covariance inflation for box-bottom fallback feet (no ankle keypoint)")
    ap.add_argument("--sigma_lo", type=float, default=0.06)
    ap.add_argument("--sigma_hi", type=float, default=2.50)
    ap.add_argument("--no_aniso", action="store_true",
                    help="force a circular error model (the thing that does not work - kept so "
                         "the anisotropy can be A/B'd rather than asserted)")
    ap.add_argument("--p_det", type=float, default=0.70)
    ap.add_argument("--lam_fa", type=float, default=0.05, help="false alarms per m2 per camera")
    ap.add_argument("--w_app", type=float, default=0.0,
                    help="appearance weight in nats per unit cosine distance (0 = geometry only)")
    ap.add_argument("--calib_frames", type=int, default=60)
    ap.add_argument("--calib_iters", type=int, default=4)
    ap.add_argument("--no_calib", action="store_true")
    # solver
    ap.add_argument("--max_splits", type=int, default=0,
                    help="two-way split moves per timestamp (0 = off, which gave the best result "
                         "in the tests)")
    ap.add_argument("--split_ll", type=float, default=0.0,
                    help="log-likelihood improvement a split must show to be accepted")
    ap.add_argument("--em_iters", type=int, default=6)
    ap.add_argument("--em_tol", type=float, default=0.01)
    ap.add_argument("--open_ll", type=float, default=0.0, help="log-likelihood to open a person")
    ap.add_argument("--keep_ll", type=float, default=0.0, help="log-likelihood to keep one")
    ap.add_argument("--temporal", action="store_true", default=True,
                    help="seed candidates from the previous timestamp (4D-association style)")
    ap.add_argument("--no_temporal", dest="temporal", action="store_false")
    # foreground evidence (optional, the pipeline does not use it)
    ap.add_argument("--foreground", default="",
                    help="directory of fg_<cam>.npz foreground masks (optional). Adds a "
                         "detector-independent 'is a body standing here' term (POM)")
    ap.add_argument("--body_r", type=float, default=0.30, help="standing-person cylinder radius")
    ap.add_argument("--w_fg", type=float, default=1.0, help="weight on the foreground llr")
    ap.add_argument("--fg_bins", type=int, default=16)
    ap.add_argument("--fg_max_llr", type=float, default=3.0,
                    help="clamp per-camera foreground evidence, so no single view can outvote "
                         "the geometry of every other")
    ap.add_argument("--fg_grid", type=float, default=0.25,
                    help="floor resolution of the foreground occupancy map")
    ap.add_argument("--fg_peak_ll", type=float, default=2.0,
                    help="summed foreground evidence a peak needs to be proposed as a person")
    ap.add_argument("--fg_max_open", type=int, default=3,
                    help="people the foreground may add per timestamp that the detector missed. "
                         "Each is accepted only if the DETECTION objective improves after a full "
                         "re-assignment, so foreground proposes and detections dispose")
    ap.add_argument("--fg_open_ll", type=float, default=0.0,
                    help="objective improvement an opened person must show")
    ap.add_argument("--fg_clear_m", type=float, default=2.0,
                    help="a floor sample this far from everybody counts as empty for calibration")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N timestamps")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    poses = json.loads(Path(args.placer).read_text(encoding="utf-8"))
    poses = poses.get("cameras", poses)
    pnp = json.loads(Path(args.pnp).read_text(encoding="utf-8"))
    chunks = load_chunks(args.inp)

    cov = Coverage(poses, pnp, (args.arena_x, args.arena_y), args.cov_res,
                   args.elev_min, args.body_h, args.margin_px)
    cg = cov.count_grid()
    print(f"[fuse_global] coverage over {len(cov.cams)} cameras: "
          f"median {np.median(cg):.1f} cameras/cell, min {cg.min()}, max {cg.max()}")

    if args.no_calib:
        model = _mk(args, args.sigma0 * args.sigma_scale, args.p_det, args.lam_fa, args.w_app)
        keys = sorted({(int(t), f) for f in chunks for t in chunks[f]})
        pick = np.linspace(0, len(keys) - 1, min(args.calib_frames, len(keys))).round().astype(int)
        sample = [(keys[i][0], chunks[keys[i][1]][str(keys[i][0])]) for i in sorted(set(pick.tolist()))]
    else:
        model, sample = calibrate(chunks, cov, args)
    print(f"[fuse_global] fitted {model}")

    # Optional foreground channel. Its likelihood ratio is calibrated on the people found by a
    # geometry-only pass over the calibration frames.
    fg = None
    if args.foreground:
        fg = Foreground(args.foreground, cov, args.body_r, args.body_h,
                        args.fg_bins, args.w_fg, args.fg_max_llr)
        if not fg.cams:
            print(f"[fuse_global] no fg_*.npz under {args.foreground} - channel DISABLED")
            fg = None
        else:
            boot = [(t, fuse_timestamp_global(rows, cov, model, args)[0]) for t, rows in sample]
            if not fg.calibrate(boot, (args.arena_x, args.arena_y), args.fg_clear_m):
                fg = None
            else:
                fg.build_grid((args.arena_x, args.arena_y), args.fg_grid)
                pk = [len(fg.peaks(t, args.nms_r, args.fg_peak_ll)) for t, _ in sample]
                print(f"[fuse_global] foreground occupancy: {np.mean(pk):.1f} peaks per frame "
                      f"above {args.fg_peak_ll} nats (median {np.median(pk):.0f})")

    order = sorted({(int(t), f) for f in chunks for t in chunks[f]})
    if args.limit:
        order = order[:args.limit]
    frames, timestamps, seeds = {}, [], None
    for n, (t, f) in enumerate(order):
        people, _ = fuse_timestamp_global(chunks[f][str(t)], cov, model, args, seeds, fg, t)
        frames[str(t)] = people
        timestamps.append(t)
        seeds = [p["pos"][:2] for p in people] if args.temporal else None
        if not args.quiet and (n % 25 == 0 or n == len(order) - 1):
            nc = np.mean([p["n_cams"] for p in people]) if people else 0
            print(f"  t{t:06d} [{n + 1}/{len(order)}]: {len(people)} people, {nc:.1f} cams each")

    Path(args.out).write_text(json.dumps({"timestamps": timestamps, "frames": frames}, indent=1),
                              encoding="utf-8")
    counts = [len(frames[str(t)]) for t in timestamps]
    print(f"[fuse_global] wrote {args.out}")
    print(f"[fuse_global] people per frame: median {np.median(counts):.0f}, "
          f"p05 {np.percentile(counts, 5):.0f}, p95 {np.percentile(counts, 95):.0f}, "
          f"max {max(counts)}")


if __name__ == "__main__":
    main()
