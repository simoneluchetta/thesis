"""Gathers what the 3D scripts need about one person at one instant: views, keypoints, cameras.

The 2D pipeline leaves this spread over files: the roster in work/people/roster.json (with a
`views` list per person and frame), the camera placement in placements/ and the intrinsics
in diagnostics/. This module loads the cameras and yields one PersonFrame per person and
timestamp, each view carrying its keypoints and its calibrated camera. The calibration is
the same the 2D scripts used, so the bodies land in the same world as the cameras and floor.

The folder name starts with a digit, so it is not an importable package. Every script here
runs standalone and fixes up its own import path.

Run from src/:
  uv run python people/3D_People/adapter.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # people/3D_People/
PEOPLE = HERE.parent                             # people/
SRC = PEOPLE.parent                              # src/
MT = SRC.parent                                  # repository root
for _p in (str(SRC), str(HERE)):                 # SRC for repo helpers; here for sibling modules
    if _p not in sys.path:
        sys.path.insert(0, _p)

from camera_frames import _placement_to_cam_from_world
from _floor_homography_sanity import (full_opencv_to_KD, project_world_to_pixel,
                                      warn_if_folding)

PLACER = MT / "placements" / "placed_cameras_redfox_repaired.json"
PNP = MT / "diagnostics" / "court_pnp_cameras_redfox_repaired.json"
PEOPLE_DATA = MT / "work" / "people"
# the roster written by run_people.py; it carries the `views` bundle the fit needs
DEFAULT_TRACKED = PEOPLE_DATA / "roster.json"

# The 10 perimeter cameras are trusted. The 3 ceiling cameras have unreliable floor geometry
# and are flagged untrusted; the 3D scripts skip their views.
PERIMETER = {"cam01", "cam02", "cam03", "cam04", "cam05",
             "cam06", "cam07", "cam08", "cam12", "cam13"}
UNTRUSTED_CAMS = {"cam09L", "cam11L", "cam11R"}
L_ANKLE, R_ANKLE = 15, 16   # COCO-17 indices (KeypointRCNN order)


@dataclass
class Camera:
    """A calibrated camera in the placer-world frame (world->cam R,t + FULL_OPENCV K,dist)."""
    name: str
    K: np.ndarray
    dist: np.ndarray
    R: np.ndarray      # world->cam rotation
    t: np.ndarray      # world->cam translation
    C: np.ndarray      # camera centre in world (= -R^T t)
    trusted: bool      # False for the ceiling cameras
    W: int = 0         # frame width  (px) -- for the truncation down-weight
    H: int = 0         # frame height (px)

    def project(self, world_pts) -> np.ndarray:
        """World (N,3) -> distorted pixels (N,2) via the verified full-distortion projection."""
        return project_world_to_pixel(world_pts, self.R, self.t, self.K, self.dist)


@dataclass
class ViewObs:
    """One camera's evidence for one person at one instant."""
    cam: str
    keypoints: np.ndarray   # (17,3) [u,v,score], full-res
    bbox: list              # [x0,y0,x1,y1] or None
    camera: Camera


@dataclass
class PersonFrame:
    """All views of one tracked person at one timestamp - the unit triangulation/fitting acts on."""
    track_id: int
    t: int
    pos: list               # fused floor position [X,Y,0] (placer-world)
    views: list             # list[ViewObs]

    @property
    def n_views(self) -> int:
        return len(self.views)

    @property
    def n_trusted(self) -> int:
        return sum(1 for v in self.views if v.camera.trusted)


def load_cameras(placer_path: Path = PLACER, pnp_path: Path = PNP) -> dict:
    """name -> Camera, for every cam present in both the placement and the pnp intrinsics."""
    placer = json.loads(Path(placer_path).read_text(encoding="utf-8"))
    pnp = json.loads(Path(pnp_path).read_text(encoding="utf-8"))
    cams = {}
    for name in sorted(set(placer) & set(pnp)):
        R, t = _placement_to_cam_from_world(placer[name])
        K, dist = full_opencv_to_KD(pnp[name]["params"])
        # Warn once if this camera's distortion model folds inside its own image.
        # triangulate.py drops observations in the folded region instead of using a wrong ray.
        warn_if_folding(name, K, dist, int(pnp[name].get("width", 0)) or None,
                        int(pnp[name].get("height", 0)) or None)
        C = (-R.T @ t).reshape(3)
        cams[name] = Camera(name=name, K=K, dist=dist, R=R, t=t, C=C,
                            trusted=name not in UNTRUSTED_CAMS,
                            W=int(pnp[name].get("width", 0)), H=int(pnp[name].get("height", 0)))
    return cams


def iter_person_frames(tracked: dict, cameras: dict):
    """Yield a PersonFrame per (track, timestamp) with its calibrated multi-view evidence."""
    frames = tracked.get("frames", {})
    for t in sorted(int(x) for x in tracked.get("timestamps", [])):
        for p in frames.get(str(t), []):
            views = []
            for v in p.get("views", []):
                cam = cameras.get(v.get("cam"))
                kp = v.get("keypoints")
                if cam is None or kp is None:
                    continue
                views.append(ViewObs(cam=v["cam"], keypoints=np.asarray(kp, dtype=float),
                                     bbox=v.get("bbox"), camera=cam))
            if views:
                yield PersonFrame(track_id=p.get("track_id", p.get("id")),
                                  t=t, pos=p["pos"], views=views)


def _foot_uv(kp: np.ndarray) -> np.ndarray:
    """Ankle-midpoint pixel from a (17,3) COCO keypoint array (the foot the floor pos came from)."""
    return 0.5 * (kp[L_ANKLE, :2] + kp[R_ANKLE, :2])


def _selftest(tracked_path: Path) -> int:
    cameras = load_cameras()
    print(f"[adapter] {len(cameras)} calibrated cameras "
          f"({sum(c.trusted for c in cameras.values())} trusted, "
          f"{sum(not c.trusted for c in cameras.values())} untrusted off-floor)")
    if not Path(tracked_path).exists():
        print(f"[adapter] tracked file not found: {tracked_path}")
        return 2
    tracked = json.loads(Path(tracked_path).read_text(encoding="utf-8"))

    pf_list = list(iter_person_frames(tracked, cameras))
    if not pf_list:
        print("[adapter] NO person-frames with views (is this an older track file without `views`?)")
        return 1
    nviews = [pf.n_views for pf in pf_list]
    ntrust = [pf.n_trusted for pf in pf_list]
    tracks = sorted({pf.track_id for pf in pf_list})

    # Sanity check: the fused floor position, projected into each view, should land near that
    # view's ankle keypoint. It is not exact (the position is a multi-view average with the
    # ankle-height correction), so tens of pixels is normal.
    errs = []
    for pf in pf_list:
        W = np.array([[pf.pos[0], pf.pos[1], 0.0]])
        for v in pf.views:
            uv = v.camera.project(W)[0]
            errs.append(float(np.hypot(*(uv - _foot_uv(v.keypoints)))))
    errs = np.asarray(errs)

    print(f"[adapter] person-frames: {len(pf_list)}  across tracks {tracks}")
    print(f"[adapter] views/frame: min {min(nviews)}  median {int(np.median(nviews))}  max {max(nviews)}"
          f"   (trusted/frame: min {min(ntrust)} median {int(np.median(ntrust))})")
    print(f"[adapter] keypoints/view: {pf_list[0].views[0].keypoints.shape} (COCO-17)")
    print(f"[adapter] reproject fused-pos -> cam vs that cam's ankle: "
          f"median {np.median(errs):.1f}px  p90 {np.percentile(errs, 90):.1f}px  (sanity, not exact)")
    # >=2 views is the minimum for triangulation; flag any single-view frame.
    n_single = sum(1 for n in nviews if n < 2)
    print(f"[adapter] frames with <2 views (can't triangulate alone): {n_single}/{len(pf_list)}")
    print("[adapter] OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracked", default=str(DEFAULT_TRACKED),
                    help="the tracked people with their `views` bundle (default: work/people/roster.json)")
    ap.add_argument("--selftest", action="store_true",
                    help="load + summarize the bundle and run the reprojection sanity check")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(_selftest(Path(args.tracked)))
    # default action: just report camera count
    cams = load_cameras()
    print(f"[adapter] {len(cams)} cameras loaded; pass --selftest to inspect a track file")


if __name__ == "__main__":
    main()
