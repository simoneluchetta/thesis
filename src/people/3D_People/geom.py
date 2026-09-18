"""Geometry helpers for the body fit, written in torch so gradients can flow through them.

Contains the camera projection with the full OpenCV lens model, conversions between
axis-angle, rotation matrix and the 6D rotation used by VPoser, a robust loss, and the
closed-form rigid alignment (Kabsch) used to start the fit.

The self-test compares the projection against the OpenCV one on every real camera (they must
agree within a thousandth of a pixel) and checks the rotation helpers against OpenCV and for
round-tripping.

Run from src/:  uv run python people/3D_People/geom.py --selftest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent
for _p in (str(SRC), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# projection
def project_points_torch(X_world, R, t, K, dist):
    """World points -> distorted pixels, differentiable (torch twin of cv2.projectPoints).

    Args:
      X_world : (N,3) world points (torch)
      R       : (3,3) world->cam rotation
      t       : (3,)  world->cam translation
      K       : (3,3) intrinsics
      dist    : (8,)  [k1,k2,p1,p2,k3,k4,k5,k6] (the FULL_OPENCV rational model)
    Returns:
      (N,2) distorted pixel coords.
    """
    Xc = X_world @ R.transpose(-1, -2) + t            # (N,3) camera frame
    z = Xc[:, 2].clamp_min(1e-6)
    x = Xc[:, 0] / z
    y = Xc[:, 1] / z
    k1, k2, p1, p2, k3, k4, k5, k6 = (dist[i] for i in range(8))
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * xd + cx
    v = fy * yd + cy
    return torch.stack([u, v], dim=-1)


# rotations
def axis_angle_to_matrix(aa):
    """(...,3) axis-angle -> (...,3,3) rotation matrix (Rodrigues, differentiable)."""
    angle = torch.norm(aa, dim=-1, keepdim=True)
    eps = 1e-8
    axis = aa / angle.clamp_min(eps)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    s = torch.sin(angle)[..., 0]
    c = torch.cos(angle)[..., 0]
    C = 1 - c
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1),
    ], dim=-2)
    outer = axis.unsqueeze(-1) * axis.unsqueeze(-2)
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(K.shape)
    R = c[..., None, None] * eye + s[..., None, None] * K + C[..., None, None] * outer
    # angle ~ 0 -> identity (the formula already -> I as s,C -> 0, axis well-defined via clamp)
    return R


def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    pos = x > 0
    ret[pos] = torch.sqrt(x[pos])
    return ret


def matrix_to_axis_angle(R):
    """(...,3,3) rotation matrix -> (...,3) axis-angle (via quaternion, numerically stable)."""
    batch = R.shape[:-2]
    m = R.reshape(batch + (9,))
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(m, dim=-1)
    q_abs = _sqrt_positive_part(torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1))
    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)
    flr = torch.tensor(0.1, dtype=R.dtype, device=R.device)
    cand = quat_by_rijk / (2.0 * torch.maximum(q_abs[..., None], flr))
    idx = q_abs.argmax(dim=-1)                                   # (...,)
    q = torch.gather(cand, -2, idx[..., None, None].expand(batch + (1, 4))).squeeze(-2)
    # standardize: w >= 0
    q = torch.where(q[..., :1] < 0, -q, q)
    # quaternion -> axis-angle
    vnorm = torch.norm(q[..., 1:], dim=-1, keepdim=True)
    half = torch.atan2(vnorm, q[..., :1])
    angle = 2 * half
    small = angle.abs() < 1e-6
    sin_half_over_angle = torch.where(
        small, 0.5 - angle * angle / 48.0, torch.sin(half) / angle.clamp_min(1e-12))
    return q[..., 1:] / sin_half_over_angle.clamp_min(1e-12)


def rot6d_to_matrix(d6):
    """VPoser 6D rotation rep -> (...,3,3) rotation matrix.

    Matches human_body_prior's ContinousRotReprDecoder exactly: the 6 values are read as a
    (3,2) block (first 3 = first column b1, next 3 = b2), Gram-Schmidt orthonormalized.
    """
    block = d6.reshape(d6.shape[:-1] + (3, 2))
    b1 = torch.nn.functional.normalize(block[..., 0], dim=-1)
    dot = (b1 * block[..., 1]).sum(-1, keepdim=True)
    b2 = torch.nn.functional.normalize(block[..., 1] - dot * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)                     # columns -> (...,3,3)


# robust loss
# the robust loss used by SMPLify, see https://smplify.is.tue.mpg.de/
def gmof(residual, sigma):
    """Geman-McClure robust error: sigma^2 * r^2 / (sigma^2 + r^2). Returns the same shape as
    `residual` (already squared inside); call .sum() or .mean() on the result."""
    r2 = residual ** 2
    return (sigma ** 2) * r2 / (sigma ** 2 + r2)


# rigid init
# see: https://en.wikipedia.org/wiki/Kabsch_algorithm
# and: https://stackoverflow.com/questions/60877274/optimal-rotation-in-3d-with-kabsch-algorithm
def kabsch(A, B, w=None):
    """Best rigid transform (R,t) (no scale) mapping A->B, minimizing sum w*||R A + t - B||^2.
    A,B : (N,3) numpy. Returns (R 3x3, t 3,). Standard Kabsch/Umeyama (rotation only)."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    if w is None:
        w = np.ones(len(A))
    w = np.asarray(w, float)
    W = w.sum()
    ca = (w[:, None] * A).sum(0) / W
    cb = (w[:, None] * B).sum(0) / W
    Ac, Bc = A - ca, B - cb
    H = (w[:, None] * Ac).T @ Bc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = cb - R @ ca
    return R, t


# selftest
def _selftest() -> int:
    import json
    import cv2
    from _floor_homography_sanity import full_opencv_to_KD, project_world_to_pixel

    torch.manual_seed(0)
    np.random.seed(0)

    # projection vs cv2 on every real camera
    MT = SRC.parent
    placer = json.loads((MT / "placements" / "placed_cameras_redfox_repaired.json").read_text("utf-8"))
    pnp = json.loads((MT / "diagnostics" / "court_pnp_cameras_redfox_repaired.json").read_text("utf-8"))
    from camera_frames import _placement_to_cam_from_world

    worst = 0.0
    n_cams = 0
    n_pts_tot = 0
    for name in sorted(set(placer) & set(pnp)):
        R, t = _placement_to_cam_from_world(placer[name])
        K, dist = full_opencv_to_KD(pnp[name]["params"])
        W, H = pnp[name]["width"], pnp[name]["height"]
        # World points spread across the arena volume (people-height band, 0-4 m up).
        pts = np.random.uniform([-8, -5, 0.0], [8, 5, 4.0], (4000, 3))
        uv_cv = project_world_to_pixel(pts, R, t, K, dist)
        Xc = pts @ R.T + t.reshape(3)
        # Only the regime the fit ever evaluates: in front of the cam and inside the image.
        inb = ((Xc[:, 2] > 0.5)
               & (uv_cv[:, 0] >= 0) & (uv_cv[:, 0] < W)
               & (uv_cv[:, 1] >= 0) & (uv_cv[:, 1] < H))
        if inb.sum() == 0:
            continue
        uv_t = project_points_torch(
            torch.tensor(pts[inb], dtype=torch.float64),
            torch.tensor(R, dtype=torch.float64),
            torch.tensor(t.reshape(3), dtype=torch.float64),
            torch.tensor(K, dtype=torch.float64),
            torch.tensor(dist.reshape(-1), dtype=torch.float64),
        ).numpy()
        d = np.linalg.norm(uv_cv[inb] - uv_t, axis=1)
        worst = max(worst, float(d.max()))
        n_cams += 1
        n_pts_tot += int(inb.sum())
    print(f"[geom] projection vs cv2 over {n_cams} cams, {n_pts_tot} in-image pts: worst {worst:.2e} px")
    assert worst < 1e-3, f"torch projection disagrees with cv2 by {worst} px"

    # axis_angle_to_matrix vs cv2.Rodrigues, and the round trip through matrix_to_axis_angle
    aa = torch.tensor(np.random.uniform(-3, 3, (500, 3)), dtype=torch.float64)
    R_t = axis_angle_to_matrix(aa)
    err_fwd = 0.0
    for i in range(len(aa)):
        # Rodrigues turns a rotation vector into a matrix
        # see: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
        Rcv, _ = cv2.Rodrigues(aa[i].numpy())
        err_fwd = max(err_fwd, float(np.abs(Rcv - R_t[i].numpy()).max()))
    aa_back = matrix_to_axis_angle(R_t)
    R_back = axis_angle_to_matrix(aa_back)
    err_round = float((R_back - R_t).abs().max())
    print(f"[geom] axis_angle_to_matrix vs cv2.Rodrigues: max {err_fwd:.2e}")
    print(f"[geom] matrix->aa->matrix round-trip: max {err_round:.2e}")
    assert err_fwd < 1e-9 and err_round < 1e-9

    # rot6d_to_matrix gives a proper rotation
    d6 = torch.tensor(np.random.randn(100, 6), dtype=torch.float64)
    Rm = rot6d_to_matrix(d6)
    ortho = (Rm @ Rm.transpose(-1, -2) - torch.eye(3, dtype=torch.float64)).abs().max()
    dets = torch.det(Rm)
    print(f"[geom] rot6d_to_matrix orthonormality err {float(ortho):.2e}; "
          f"det in [{float(dets.min()):.4f},{float(dets.max()):.4f}]")
    assert float(ortho) < 1e-9 and float(dets.min()) > 0.999

    # kabsch recovers a known transform
    Rgt = axis_angle_to_matrix(torch.tensor([0.3, -0.7, 1.1])).numpy()
    tgt = np.array([1.0, -2.0, 0.5])
    A = np.random.randn(20, 3)
    B = A @ Rgt.T + tgt
    Rk, tk = kabsch(A, B)
    print(f"[geom] kabsch R err {np.abs(Rk - Rgt).max():.2e}  t err {np.abs(tk - tgt).max():.2e}")
    assert np.abs(Rk - Rgt).max() < 1e-6 and np.abs(tk - tgt).max() < 1e-6   # SVD precision

    print("[geom] OK")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(_selftest())
    print("[geom] differentiable geometry core. Run --selftest to validate against cv2.")
