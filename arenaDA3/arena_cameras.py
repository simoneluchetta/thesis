"""Loading the calibrated cameras and their plates, shared by the three DA3 lanes.

`load_cameras` reads a calibration pair (intrinsics from `diagnostics/`, poses from
`placements/`) into one dictionary per camera: K, the eight distortion coefficients, the
image size, the camera centre and the camera-to-world rotation. `prep_plate` undistorts a
plate into a true pinhole image and shrinks it to a workable size. `make_rays` and
`ray_aabb` turn a pinhole camera into one ray per pixel and clip those rays against the box
the arena lives in, which is how the depth export knows where the floor should be.

These functions are the same ones the NeRF study used, so the depth maps line up with the
cameras the NeRF was trained with.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

PERIMETER = ["cam01", "cam02", "cam03", "cam04", "cam05",
             "cam06", "cam07", "cam08", "cam12", "cam13"]


def euler_zxy_to_R_c2w(rot) -> np.ndarray:
    """The three angles the viewer stores for a camera, turned into a rotation matrix.

    Order: yaw about Z, then pitch about X, then roll about Y. It must match
    camera_frames._euler_zxy_to_rotation_matrix in src/.
    """
    # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html
    return ScipyRotation.from_euler("ZXY", [rot[0], rot[1], rot[2]]).as_matrix()


def load_cameras(pnp_path: Path, pose_path: Path, cams: list[str]) -> dict:
    pnp = json.loads(pnp_path.read_text(encoding="utf-8"))
    poses = json.loads(pose_path.read_text(encoding="utf-8"))
    out = {}
    for name in cams:
        if name not in pnp or name not in poses:
            print(f"  [skip] {name}: missing in pnp/pose")
            continue
        p = pnp[name]["params"]  # [fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6]
        K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.0]], dtype=np.float64)
        dist = np.asarray(p[4:12], dtype=np.float64)
        W, H = int(pnp[name]["width"]), int(pnp[name]["height"])
        C = np.asarray(poses[name]["position"], dtype=np.float64)        # camera center (world)
        R_c2w = euler_zxy_to_R_c2w(poses[name]["rotation"])
        out[name] = dict(K=K, dist=dist, W=W, H=H, C=C, R_c2w=R_c2w)
    return out


def prep_plate(plate_path: Path, cam: dict, long_side: int):
    """Undistort a plate into a pinhole image and shrink it.

    make_rays below assumes a pinhole camera. The shrink keeps the ray count down: a 4K
    frame is eight million rays, times ten cameras.
    """
    bgr = cv2.imread(str(plate_path))
    if bgr is None:
        raise FileNotFoundError(plate_path)
    H, W = bgr.shape[:2]
    K = cam["K"].copy()
    # Rescale K if the plate resolution differs from the calibration resolution.
    if (W, H) != (cam["W"], cam["H"]):
        K[0, :] *= W / cam["W"]
        K[1, :] *= H / cam["H"]
    und = cv2.undistort(bgr, K, cam["dist"], None, K)  # true pinhole under K now
    scale = long_side / max(W, H)
    nw, nh = int(round(W * scale)), int(round(H * scale))
    small = cv2.resize(und, (nw, nh), interpolation=cv2.INTER_AREA)
    Ks = K.copy()
    Ks[0, :] *= nw / W
    Ks[1, :] *= nh / H
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb, Ks, nw, nh


def make_rays(K: np.ndarray, R_c2w: np.ndarray, C: np.ndarray, W: int, H: int):
    """One ray per pixel, in world coordinates.

    Every ray from a camera starts at the same place, its centre, and differs only in
    direction. OpenCV convention throughout: Z forward, Y down.
    """
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")  # i=col(u), j=row(v)
    x = (i + 0.5 - K[0, 2]) / K[0, 0]
    y = (j + 0.5 - K[1, 2]) / K[1, 1]
    dirs_cam = np.stack([x, y, np.ones_like(x)], axis=-1)          # +Z forward (OpenCV)
    dirs_world = dirs_cam @ R_c2w.T                               # d_w = R_c2w @ d_c
    dirs_world /= np.linalg.norm(dirs_world, axis=-1, keepdims=True)
    origins = np.broadcast_to(C.astype(np.float32), dirs_world.shape).reshape(-1, 3)
    return origins.copy(), dirs_world.reshape(-1, 3).astype(np.float32)


def ray_aabb(origins: np.ndarray, dirs: np.ndarray, bbmin, bbmax):
    """Where each ray enters and leaves the box the arena sits in.

    Sampling stays inside the box. Rays that miss it are flagged invalid.
    see: https://en.wikipedia.org/wiki/Minimum_bounding_box
    """
    bbmin = np.asarray(bbmin, np.float32)
    bbmax = np.asarray(bbmax, np.float32)
    inv = 1.0 / np.where(np.abs(dirs) < 1e-8, 1e-8, dirs)
    t0 = (bbmin[None] - origins) * inv
    t1 = (bbmax[None] - origins) * inv
    tmin = np.minimum(t0, t1)
    tmax = np.maximum(t0, t1)
    near = np.max(tmin, axis=-1)
    far = np.min(tmax, axis=-1)
    near = np.maximum(near, 0.0)
    valid = far > (near + 1e-3)
    return near.astype(np.float32), far.astype(np.float32), valid
