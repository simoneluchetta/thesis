"""Fits an SMPL-X body to each tracked person, in metres, in the arena's world coordinates.

Each track gets one body shape (betas, shared by all its frames) and one pose per frame
(global_orient, body_pose, transl). Adam minimises, in two stages: the 3D distance between the
model's COCO joints and the triangulated joints (metres); the 2D reprojection error against
the raw keypoints in every trusted camera (pixels); an optional silhouette term from masks.py;
a penalty on feet below the floor, plus a soft pull to the floor when the ankle is already
low; mild priors on pose and shape; and a bounded acceleration penalty that removes jitter.
The data terms use a robust loss in [0, 1), so metres and pixels can be weighed together.
Each frame starts from a rigid (Kabsch) alignment of the rest pose onto the triangulated
joints. The COCO to SMPL-X joint mapping is resolved by name and checked against known indices.

Reads the tracked people (adapter.DEFAULT_TRACKED or --tracked), the calibration and the
SMPL-X model in models/smplx/. Writes a JSON with per-frame body parameters and per-track
betas. run_bodies.py runs this. Standalone, from src/:
  uv run python people/3D_People/fit_smplx.py --check                 # just load + introspect SMPL-X
  uv run python people/3D_People/fit_smplx.py --selftest              # fit a couple of frames, sanity
  uv run python people/3D_People/fit_smplx.py --out ../work/bodies/bodies.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent
for _p in (str(SRC), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adapter import DEFAULT_TRACKED, iter_person_frames, load_cameras
from triangulate import triangulate_person_frame
from geom import kabsch, matrix_to_axis_angle, project_points_torch

MODELS = HERE / "models"
DEFAULT_MASKS = SRC.parent / "work" / "bodies" / "masks_dense.npz"   # masks.py output (silhouette term input)

# COCO-17 (KeypointRCNN order) -> SMPL-X joint name. Resolved to indices at runtime by name.
COCO_TO_SMPLX_NAME = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
# expected indices for smplx 0.1.28; an assert at runtime catches a version change.
COCO_TO_SMPLX_IDX_EXPECTED = [55, 57, 56, 59, 58, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
# foot landmarks used for the floor term (sole-level joints), resolved by name too.
FOOT_NAMES = ["left_heel", "right_heel", "left_big_toe", "right_big_toe",
              "left_small_toe", "right_small_toe"]
PELVIS_NAME = "pelvis"


# model loader
def _download_msg(model_dir: Path, err: Exception) -> str:
    return (
        "\n[fit_smplx] SMPL-X model files not found / not loadable.\n"
        f"  looked under: {model_dir}\n"
        f"  underlying error: {type(err).__name__}: {err}\n\n"
        "  SMPL-X is free for research but license-gated and NOT redistributable.\n"
        "  1) register + download at https://smpl-x.is.tue.mpg.de\n"
        f"  2) place the models so this path exists: {model_dir / 'smplx' / 'SMPLX_NEUTRAL.npz'}\n"
        "  3) re-run:  uv run python people/3D_People/fit_smplx.py --check\n"
    )


def build_smplx(gender: str = "neutral", model_dir: Path = MODELS, num_betas: int = 10,
                batch_size: int = 1):
    """Construct an SMPL-X body model from the local (git-ignored) model files."""
    try:
        import smplx
    except ImportError as e:                      # pragma: no cover
        raise SystemExit(f"[fit_smplx] the `smplx` package is not installed: {e}")
    try:
        # the SMPL-X body model
        # see: https://github.com/vchoutas/smplx
        # and: https://smpl-x.is.tue.mpg.de/
        return smplx.create(
            str(model_dir), model_type="smplx", gender=gender,
            use_pca=False, flat_hand_mean=True, num_betas=num_betas, batch_size=batch_size,
        )
    except Exception as e:                         # FileNotFoundError / KeyError on missing files
        raise SystemExit(_download_msg(model_dir, e))


def smplx_joint_index(name: str) -> int:
    from smplx.joint_names import JOINT_NAMES
    return JOINT_NAMES.index(name)


def coco_to_smplx_idx() -> list:
    idx = [smplx_joint_index(n) for n in COCO_TO_SMPLX_NAME]
    assert idx == COCO_TO_SMPLX_IDX_EXPECTED, (
        f"COCO->SMPL-X joint indices changed in this smplx version: {idx} "
        f"!= {COCO_TO_SMPLX_IDX_EXPECTED}. Re-verify the mapping before fitting.")
    return idx


def _check(gender: str) -> int:
    model = build_smplx(gender=gender)
    out = model()
    joints = out.joints.detach().cpu().numpy()[0]
    verts = out.vertices.detach().cpu().numpy()[0]
    cmap = coco_to_smplx_idx()
    feet = [smplx_joint_index(n) for n in FOOT_NAMES]
    print(f"[fit_smplx] SMPL-X '{gender}' loaded OK; vertices {verts.shape}  joints {joints.shape}")
    print(f"[fit_smplx] COCO->SMPL-X joint idx (verified): {cmap}")
    print(f"[fit_smplx] foot joint idx {FOOT_NAMES} -> {feet}")
    print(f"[fit_smplx] pelvis idx {smplx_joint_index(PELVIS_NAME)}")
    return 0


# fit config
class Cfg:
    """Fit hyper-parameters (CLI-overridable). Robust terms are normalised to [0,1) per joint."""
    sigma3d = 0.08          # m   GMoF scale for the 3D-joint term
    sigma2d = 60.0          # px  GMoF scale for the 2D reprojection term
    w3d = 1.0
    w2d = 1.0
    w_silhouette = 0.15     # silhouette/mask term (low, < w3d; stage-2 only; needs masks.py output)
    sil_clamp_px = 48.0     # px  GMoF scale for the silhouette term (matches the mask DT clamp)
    n_sil_verts = 1500      # SMPL-X surface vertices subsampled for the silhouette term
    w_floor_pen = 50.0      # one-sided foot-below-floor penalty (always on)
    w_contact = 2.0         # gated soft contact pull toward the floor
    contact_ankle_z = 0.20  # m   triangulated ankle below this => foot considered planted
    foot_target_z = 0.03    # m   sole-joint height at contact (heel/toe sit ~few cm up)
    w_pose = 0.5            # pose prior (L2 toward rest, or ||z||^2 for vposer)
    w_beta = 1.0            # shape prior (L2 on beta). At 1.0 the body stays close to the neutral
                            #   template; lower it with --w_beta to let the shape vary more.
    free_betas = 10         # optimise only the first N shape dims; the rest stay at zero
    w_smooth_pose = 0.3     # Bounded acceleration smoothness on pose (see _accel2)
    w_smooth_transl = 0.3   # Bounded acceleration smoothness on translation
    fps = 25.0              # source frame rate: turns timestamp gaps into seconds
    accel_ref_transl = 8.0  # m/s^2   the acceleration a sprinting athlete's pelvis reaches;
                            #         smoothness costs ~w at this level and saturates above it
    accel_ref_pose = 30.0   # rad/s^2 same idea for the 21 body-joint angles
    smooth_max_gap_s = 0.5  # never smooth across a gap longer than this (s)
    # graduated non-convexity: start with a wide robust loss and narrow it, so the fit is not
    # trapped by a bad start. see https://arxiv.org/abs/1909.08605
    gnc_sigma3d_start = 0.8 # m   sigma3d anneals from here...
    gnc_sigma2d_start = 400.0  # px ...and sigma2d from here, down to the Cfg values
    gnc_frac = 0.6          # fraction of a stage's iterations spent annealing
    # COCO hips sit on the body surface, SMPL-X hips are joint centres, so they are 0.207 m
    # apart in the data and 0.122 m in the model. The model's hips are widened by this before
    # comparing; the output joints are untouched.
    # see: https://cocodataset.org/#keypoints-2020
    hip_widen_m = 0.085     # m
    kp_thr = 2.0            # min KeypointRCNN score to use a keypoint in the 2D term
    truncation = True       # down-weight 2D residuals of detections cut by the frame edge
    trunc_margin_px = 8.0   # bbox within this many px of the frame border counts as truncated
    trunc_floor = 0.3       # min 2D weight for a fully-truncated view
    stage1_iters = 250      # Adam iters: 3D + priors + floor (no 2D)
    stage2_iters = 200      # Adam iters: + multi-view 2D reprojection
    num_betas = 10
    pose_prior = "l2"       # 'l2' | 'vposer'
    gender = "neutral"


def _robust(r2, s2):
    """Normalised Geman-McClure in [0,1): r^2 / (s^2 + r^2)."""
    return r2 / (s2 + r2)


def _accel2(x, tsec, max_gap_s, a_ref):
    """Bounded acceleration penalty for a per-frame quantity, on non-uniform timestamps.

    The second derivative is estimated with the 3-point formula on the real time gaps, divided
    by `a_ref` (the acceleration a real athlete reaches) and passed through the robust loss.
    Real motion costs about 1 at most, while jitter is suppressed. Frame triples with a gap
    longer than `max_gap_s` seconds are skipped.

    x    : (F, D) per-frame quantity
    tsec : (F,) timestamps in seconds
    """
    F = x.shape[0]
    if F < 3:
        return x.sum() * 0.0
    h1 = tsec[1:-1] - tsec[:-2]                      # (F-2,)
    h2 = tsec[2:] - tsec[1:-1]
    ok = (h1 > 0) & (h2 > 0) & (h1 <= max_gap_s) & (h2 <= max_gap_s)
    if not bool(ok.any()):
        return x.sum() * 0.0
    h1, h2 = h1[ok], h2[ok]
    xa, xb, xc = x[:-2][ok], x[1:-1][ok], x[2:][ok]
    # standard 3-point second derivative on a non-uniform grid
    num = xa * h2[:, None] - xb * (h1 + h2)[:, None] + xc * h1[:, None]
    d2 = 2.0 * num / (h1 * h2 * (h1 + h2))[:, None]
    r2 = (d2 / a_ref).pow(2).sum(-1)
    return _robust(r2, 1.0).mean()


def _widen_hips(Jc, widen_m):
    """Push the model's COCO-hip joints apart along the hip axis by `widen_m` metres.

    COCO marks the hip on the body surface, SMPL-X at the joint centre. Widening the model's
    hips before comparing avoids distorting the shape or pose to absorb that gap.
    """
    if widen_m <= 0:
        return Jc
    v = Jc[:, 11] - Jc[:, 12]                                    # right hip -> left hip
    u = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    half = 0.5 * widen_m * u
    Jc = Jc.clone()
    Jc[:, 11] = Jc[:, 11] + half
    Jc[:, 12] = Jc[:, 12] - half
    return Jc


def _trunc_weight(bbox, W, H, margin, floor):
    """Weight of a view in the 2D term: 1.0 when the bbox is at least `margin` px from every
    frame border, falling to `floor` when it touches one (the person is probably cut off)."""
    if bbox is None or W <= 0 or H <= 0:
        return 1.0
    x0, y0, x1, y1 = bbox
    d = min(x0, y0, W - x1, H - y1)            # nearest bbox edge -> frame border (px)
    if d >= margin:
        return 1.0
    return float(floor + (1.0 - floor) * max(0.0, d) / margin)


# silhouette masks
def load_masks(npz_path):
    """Load masks.py output -> {(track_id, t, cam): {dt:(H,W) float32 px, roi:[x0,y0,w,h]}}.
    Returns None if the bundle is absent (the fit then runs without the silhouette term)."""
    npz_path = Path(npz_path)
    json_path = npz_path.with_name(npz_path.stem + ".json")
    if not (npz_path.exists() and json_path.exists()):
        return None
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    arr = np.load(npz_path)
    lut = {}
    for key, m in meta["keys"].items():
        tid, t, cam = key.split("|")
        lut[(int(tid), int(t), cam)] = {"dt": arr[key].astype(np.float32), "roi": m["roi"]}
    return lut


# per-track data prep
def load_track_data(tracked_path, cameras, cfg: Cfg, sel_px=25.0, min_views=2):
    """Group person-frames by track, triangulate each (3D target + per-joint weight), and keep the
    trusted 2D views. Returns {track_id: {ts, targets(F,17,3), w(F,17), views[F]}}."""
    tracked = json.loads(Path(tracked_path).read_text(encoding="utf-8"))
    by_track = {}
    for pf in iter_person_frames(tracked, cameras):
        joints = triangulate_person_frame(pf, cfg.kp_thr, sel_px, min_views)
        tgt = np.full((17, 3), np.nan)
        w = np.zeros(17)
        for j in range(17):
            r = joints[j]
            if r:
                tgt[j] = r["X"]
                # reliability: present, more views better, lower rms better
                w[j] = (1.0 + 0.5 * (r["n_views"] - 2)) * np.exp(-r["rms_px"] / 15.0)
        # 2D views from trusted cameras only
        vlist = []
        for v in pf.views:
            if v.camera.trusted:
                vlist.append((v.camera, np.asarray(v.keypoints, float), v.bbox))
        d = by_track.setdefault(pf.track_id, {"ts": [], "targets": [], "w": [], "views": []})
        d["ts"].append(pf.t)
        d["targets"].append(tgt)
        d["w"].append(w)
        d["views"].append(vlist)
    # sort each track by time, to numpy
    out = {}
    for tid, d in by_track.items():
        order = np.argsort(d["ts"])
        out[tid] = {
            "ts": [int(d["ts"][i]) for i in order],
            "targets": np.stack([d["targets"][i] for i in order]),  # (F,17,3)
            "w": np.stack([d["w"][i] for i in order]),              # (F,17)
            "views": [d["views"][i] for i in order],
        }
    return out


# the fit
def fit_track(track, cameras, cfg: Cfg, device, vposer=None, masks=None, verbose=True):
    """Fit shared-shape + per-frame-pose SMPL-X to one track. Returns per-frame result dicts."""
    ts = track["ts"]
    tid = track.get("tid")
    F = len(ts)
    targets = torch.tensor(track["targets"], dtype=torch.float32, device=device)   # (F,17,3)
    wj = torch.tensor(track["w"], dtype=torch.float32, device=device)              # (F,17)
    valid3d = torch.isfinite(targets).all(-1) & (wj > 0)                            # (F,17)
    # NaN targets (un-triangulated joints) are masked by valid3d, but NaN*0=NaN would still poison
    # the sum -> replace with a finite dummy; valid3d zeroes their contribution.
    targets_safe = torch.nan_to_num(targets, nan=0.0)

    model = build_smplx(cfg.gender, MODELS, cfg.num_betas, batch_size=F).to(device)
    cmap = torch.tensor(coco_to_smplx_idx(), dtype=torch.long, device=device)
    feet = torch.tensor([smplx_joint_index(n) for n in FOOT_NAMES], dtype=torch.long, device=device)
    pelvis_i = smplx_joint_index(PELVIS_NAME)

    # rest-pose joints (zero pose/shape) for the Kabsch init
    with torch.no_grad():
        rest = model(betas=torch.zeros(F, cfg.num_betas, device=device),
                     global_orient=torch.zeros(F, 3, device=device),
                     body_pose=torch.zeros(F, 63, device=device),
                     transl=torch.zeros(F, 3, device=device))
        Jrest = rest.joints[0].detach().cpu().numpy()       # (127,3)
        nV = rest.vertices.shape[1]                          # 10475
    Jrest_coco = Jrest[COCO_TO_SMPLX_IDX_EXPECTED]          # (17,3)
    pelvis_rest = Jrest[pelvis_i]                            # (3,)
    A = Jrest_coco - pelvis_rest                            # rest joints centred on pelvis

    # per-frame Kabsch init (SMPL-X applies global_orient about the pelvis)
    go0 = np.zeros((F, 3)); tr0 = np.zeros((F, 3))
    tgt_np = track["targets"]; w_np = track["w"]
    upright = np.array([np.pi / 2, 0.0, 0.0])           # SMPL-X Y-up -> world Z-up
    # a finite global fallback centre (mean of each frame's visible joints), for frames w/o init
    per_frame_ctr = []
    for f in range(F):
        mm = np.isfinite(tgt_np[f]).all(1)
        per_frame_ctr.append(tgt_np[f][mm].mean(0) if mm.any() else None)
    finite_ctrs = [c for c in per_frame_ctr if c is not None]
    glob_ctr = np.mean(np.stack(finite_ctrs), 0) if finite_ctrs else np.zeros(3)
    last, n_kabsch = None, 0
    for f in range(F):
        m = np.isfinite(tgt_np[f]).all(1) & (w_np[f] > 0)
        if m.sum() >= 3:
            R0, u0 = kabsch(A[m], tgt_np[f][m], w_np[f][m])
            go0[f] = matrix_to_axis_angle(torch.tensor(R0)).numpy()
            tr0[f] = u0 - pelvis_rest
            last = (go0[f].copy(), tr0[f].copy()); n_kabsch += 1
        elif last is not None:                          # carry the last good pose forward
            go0[f], tr0[f] = last
        else:                                           # stand upright at a known-finite centre
            go0[f] = upright
            ctr = per_frame_ctr[f] if per_frame_ctr[f] is not None else glob_ctr
            tr0[f] = ctr - pelvis_rest
    # make sure nothing non-finite reaches the optimiser
    bad = ~(np.isfinite(go0).all(1) & np.isfinite(tr0).all(1))
    if bad.any():
        go0[bad] = upright
        tr0[bad] = glob_ctr - pelvis_rest
    if verbose:
        print(f"  init: {n_kabsch}/{F} frames Kabsch-aligned, {int(bad.sum())} sanitised")

    betas = torch.zeros(1, cfg.num_betas, device=device, requires_grad=True)
    n_free = max(1, min(int(cfg.free_betas), cfg.num_betas))
    beta_mask = torch.zeros(1, cfg.num_betas, device=device)
    beta_mask[0, :n_free] = 1.0        # pinned dims contribute neither shape nor prior gradient
    global_orient = torch.tensor(go0, dtype=torch.float32, device=device, requires_grad=True)
    transl = torch.tensor(tr0, dtype=torch.float32, device=device, requires_grad=True)
    if cfg.pose_prior == "vposer":
        assert vposer is not None
        pose_lat = torch.zeros(F, 32, device=device, requires_grad=True)   # VPoser latent
        body_pose_param = pose_lat
    else:
        body_pose_param = torch.zeros(F, 63, device=device, requires_grad=True)

    # precompute camera + keypoint (+ silhouette DT) tensors once, not per iteration
    cam_cache = {}
    views_t = []      # per frame: list of (R,t,K,dist, kp_uv(17,2), mask(17,), sil|None)
    trunc_t = []      # per frame: parallel list of truncation down-weights (2D term only)
    n_masked = 0
    for f in range(F):
        fl = []; tl = []
        for cam, kp, bbox in track["views"][f]:
            tl.append(_trunc_weight(bbox, cam.W, cam.H, cfg.trunc_margin_px, cfg.trunc_floor)
                      if cfg.truncation else 1.0)
            if cam.name not in cam_cache:
                cam_cache[cam.name] = (
                    torch.tensor(cam.R, dtype=torch.float32, device=device),
                    torch.tensor(cam.t.reshape(3), dtype=torch.float32, device=device),
                    torch.tensor(cam.K, dtype=torch.float32, device=device),
                    torch.tensor(cam.dist.reshape(-1), dtype=torch.float32, device=device),
                )
            uv = torch.tensor(kp[:, :2], dtype=torch.float32, device=device)
            mask = torch.tensor(kp[:, 2] >= cfg.kp_thr, device=device)
            sil = None
            m = masks.get((tid, ts[f], cam.name)) if masks else None
            if m is not None:
                dt = torch.tensor(m["dt"], dtype=torch.float32, device=device).view(1, 1, *m["dt"].shape)
                sil = (dt, m["roi"])      # roi = [x0,y0,w,h] full-frame px
                n_masked += 1
            fl.append(cam_cache[cam.name] + (uv, mask, sil))
        views_t.append(fl)
        trunc_t.append(tl)
    # fixed surface-vertex subsample for the silhouette term (same indices every frame)
    sil_vidx = torch.arange(0, nV, max(1, nV // cfg.n_sil_verts), device=device)
    ssil = cfg.sil_clamp_px ** 2

    tsec = torch.tensor([t / max(cfg.fps, 1e-6) for t in ts],
                        dtype=torch.float32, device=device)
    # contact gate from triangulated ankle height (planted feet only)
    ank_z = np.nanmin(np.where(valid3d.cpu().numpy()[:, [15, 16]],
                               tgt_np[:, [15, 16], 2], np.nan), axis=1)
    planted = torch.tensor(np.nan_to_num(ank_z, nan=9.9) < cfg.contact_ankle_z,
                           dtype=torch.float32, device=device)               # (F,)

    def body_pose():
        return vposer.decode(body_pose_param) if cfg.pose_prior == "vposer" else body_pose_param

    def forward():
        return model(betas=(betas * beta_mask).expand(F, -1), global_orient=global_orient,
                     body_pose=body_pose(), transl=transl)

    def losses(use_2d: bool, use_sil: bool, s3: float, s2: float):
        out = forward()
        J = out.joints                                   # (F,127,3)
        Jc = _widen_hips(J[:, cmap], cfg.hip_widen_m)    # (F,17,3), COCO-hip convention
        terms = {}
        # 3D term (targets_safe is finite everywhere; valid3d masks the un-triangulated joints)
        r2 = ((Jc - targets_safe) ** 2).sum(-1)          # (F,17)
        e3 = (_robust(r2, s3) * wj * valid3d).sum() / (wj * valid3d).sum().clamp_min(1e-6)
        terms["3d"] = cfg.w3d * e3
        # 2D multi-view reprojection (truncated views down-weighted)
        if use_2d:
            num = Jc.new_zeros(()); den = 0.0
            for f in range(F):
                for vi, (R, t, K, dist, uv_t, mask, sil) in enumerate(views_t[f]):
                    if not bool(mask.any()):
                        continue
                    tw = trunc_t[f][vi]
                    uv = project_points_torch(Jc[f], R, t, K, dist)          # (17,2)
                    rp2 = ((uv[mask] - uv_t[mask]) ** 2).sum(-1)
                    num = num + tw * _robust(rp2, s2).sum()
                    den += tw * int(mask.sum())
            terms["2d"] = cfg.w2d * (num / max(den, 1.0))
        # silhouette: pull the surface vertices inside the person mask. The distance transform
        # is 0 inside the mask and the distance in px outside; the robust loss keeps it in [0,1).
        if use_sil:
            Vs = out.vertices[:, sil_vidx]               # (F,Nv,3)
            num = Vs.new_zeros(()); den = 0
            for f in range(F):
                for R, t, K, dist, uv_t, mask, sil in views_t[f]:
                    if sil is None:
                        continue
                    dt, (x0, y0, w, h) = sil
                    uvv = project_points_torch(Vs[f], R, t, K, dist)         # (Nv,2)
                    gx = 2.0 * (uvv[:, 0] - x0) / w - 1.0
                    gy = 2.0 * (uvv[:, 1] - y0) / h - 1.0
                    grid = torch.stack([gx, gy], -1).view(1, 1, -1, 2)
                    sval = torch.nn.functional.grid_sample(
                        dt, grid, align_corners=False, padding_mode="border").view(-1)
                    num = num + _robust(sval ** 2, ssil).sum()
                    den += sval.numel()
            terms["sil"] = cfg.w_silhouette * (num / max(den, 1))
        # floor: non-penetration (always) + gated contact pull
        fz = J[:, feet, 2]                               # (F,nfeet)
        terms["floor_pen"] = cfg.w_floor_pen * torch.relu(-fz).pow(2).mean()
        min_fz = fz.min(dim=1).values                    # (F,) lowest foot joint
        terms["contact"] = cfg.w_contact * (planted * (min_fz - cfg.foot_target_z) ** 2).sum() \
            / planted.sum().clamp_min(1.0)
        # priors
        if cfg.pose_prior == "vposer":
            terms["pose"] = cfg.w_pose * (body_pose_param ** 2).mean()
        else:
            terms["pose"] = cfg.w_pose * (body_pose_param ** 2).mean()
        terms["beta"] = cfg.w_beta * ((betas * beta_mask) ** 2).sum() / n_free
        # acceleration smoothness on translation and pose (see _accel2)
        if F >= 3:
            terms["smooth_t"] = cfg.w_smooth_transl * _accel2(
                transl, tsec, cfg.smooth_max_gap_s, cfg.accel_ref_transl)
            terms["smooth_p"] = cfg.w_smooth_pose * _accel2(
                body_pose_param, tsec, cfg.smooth_max_gap_s, cfg.accel_ref_pose)
        total = sum(terms.values())
        return total, terms

    params = [betas, global_orient, transl, body_pose_param]

    def run_stage(name, n_iter, use_2d, use_sil, lr=0.05):
        """One Adam stage with gradient clipping and a cosine learning rate; stops on a non-finite loss.

        The robust scales start wide (gnc_sigma*_start) and shrink to the Cfg values over the
        first `gnc_frac` of the iterations, so a body that starts far from its target still
        gets a gradient pulling it home (graduated non-convexity)."""
        # see: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iter)
        last_terms, t0 = {}, time.time()
        n_anneal = max(1, int(cfg.gnc_frac * n_iter))
        lo3, hi3 = cfg.sigma3d, max(cfg.gnc_sigma3d_start, cfg.sigma3d)
        lo2, hi2 = cfg.sigma2d, max(cfg.gnc_sigma2d_start, cfg.sigma2d)
        for i in range(n_iter):
            a = min(1.0, i / n_anneal)                    # 0 -> 1 across the annealing window
            s3 = (hi3 * (lo3 / hi3) ** a) ** 2            # geometric anneal, squared units
            s2 = (hi2 * (lo2 / hi2) ** a) ** 2
            opt.zero_grad()
            total, terms = losses(use_2d, use_sil, s3, s2)
            if not torch.isfinite(total):
                print(f"    [{name}] non-finite loss at iter {i}; stopping stage"); break
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            opt.step(); sched.step()
            last_terms = {k: float(v.detach()) for k, v in terms.items()}
        if verbose:
            ts_str = "  ".join(f"{k}:{v:.4f}" for k, v in last_terms.items())
            print(f"    [{name}] {n_iter}it {time.time()-t0:.1f}s  total {sum(last_terms.values()):.4f}  {ts_str}")

    sil_on = (cfg.w_silhouette > 0) and n_masked > 0
    if verbose:
        print(f"  track {tid}: F={F} frames, {n_masked} masked views, init done; fitting"
              f"{' (+silhouette)' if sil_on else ''}...")
    run_stage("stage1 3D", cfg.stage1_iters, use_2d=False, use_sil=False)
    run_stage("stage2 +2D" + ("+sil" if sil_on else ""), cfg.stage2_iters, use_2d=True, use_sil=sil_on)

    # collect results
    with torch.no_grad():
        out = forward()
        J = out.joints.detach().cpu().numpy()            # (F,127,3)
        bp_aa = body_pose().detach().cpu().numpy()        # (F,63)
        go = global_orient.detach().cpu().numpy()
        tr = transl.detach().cpu().numpy()
        bt = (betas * beta_mask).detach().cpu().numpy()[0]
        # silhouette measurement (independent of whether the term was active): mean outside-mask
        # distance (px) of the projected surface verts. Lower = mesh contour hugs the silhouette.
        sil_metric = None
        if n_masked > 0:
            Vs = out.vertices[:, sil_vidx]
            vals = []
            for f in range(F):
                for R, t, K, dist, uv_t, mask, sil in views_t[f]:
                    if sil is None:
                        continue
                    dt, (x0, y0, w, h) = sil
                    uvv = project_points_torch(Vs[f], R, t, K, dist)
                    gx = 2.0 * (uvv[:, 0] - x0) / w - 1.0
                    gy = 2.0 * (uvv[:, 1] - y0) / h - 1.0
                    grid = torch.stack([gx, gy], -1).view(1, 1, -1, 2)
                    vals.append(torch.nn.functional.grid_sample(
                        dt, grid, align_corners=False, padding_mode="border").view(-1))
            if vals:
                allv = torch.cat(vals)
                sil_metric = {"mean_outside_px": float(allv.mean()),
                              "frac_outside": float((allv > 1).float().mean()),
                              "n_views": n_masked, "n_samples": int(allv.numel())}
    results = []
    for f in range(F):
        results.append({
            "t": ts[f],
            "betas": [float(x) for x in bt],
            "global_orient": [float(x) for x in go[f]],
            "body_pose": [float(x) for x in bp_aa[f]],
            "transl": [float(x) for x in tr[f]],
            "joints_coco": J[f, COCO_TO_SMPLX_IDX_EXPECTED].tolist(),
        })
    return results, J, sil_metric


# validation
def _validate(track, J, cameras, cfg: Cfg):
    """Reprojection (px) of fitted COCO joints vs raw keypoints, 3D residual vs triangulation,
    bone-length stability, and floor stats. Returns a metrics dict."""
    ts = track["ts"]; F = len(ts)
    Jc = J[:, COCO_TO_SMPLX_IDX_EXPECTED]                 # (F,17,3)
    feet = [smplx_joint_index(n) for n in FOOT_NAMES]
    # 2D reprojection vs raw keypoints (trusted cams)
    px = []
    for f in range(F):
        for cam, kp, _bbox in track["views"][f]:
            from _floor_homography_sanity import project_world_to_pixel
            uv = project_world_to_pixel(Jc[f], cam.R, cam.t, cam.K, cam.dist)
            for j in range(17):
                if kp[j, 2] >= cfg.kp_thr:
                    px.append(float(np.hypot(*(uv[j] - kp[j, :2]))))
    # 3D residual vs triangulation
    d3 = []
    tg = track["targets"]
    for f in range(F):
        for j in range(17):
            if np.isfinite(tg[f, j]).all():
                d3.append(float(np.linalg.norm(Jc[f, j] - tg[f, j])))
    # bone-length stability (shared shape => should be near-constant across frames)
    BONES = {"thigh_L": (11, 13), "shin_L": (13, 15), "upperarm_L": (5, 7),
             "shoulder_w": (5, 6), "hip_w": (11, 12)}
    bone_std = {}
    for b, (a, c) in BONES.items():
        L = np.linalg.norm(Jc[:, a] - Jc[:, c], axis=1)
        bone_std[b] = (float(np.median(L)), float(np.std(L)))
    foot_z = J[:, feet, 2]
    return {
        "px": px, "d3": d3, "bone": bone_std,
        "foot_z_min": float(foot_z.min()), "foot_z_med": float(np.median(foot_z.min(1))),
        "penetration_frac": float((foot_z < -0.02).mean()),
    }


def _report_val(tid, m):
    px = np.asarray(m["px"]); d3 = np.asarray(m["d3"])
    print(f"  [track {tid}] 2D reproj: median {np.median(px):.1f}px  p90 {np.percentile(px,90):.1f}px "
          f"(n={px.size})")
    print(f"  [track {tid}] 3D vs triangulation: median {np.median(d3)*100:.1f}cm  "
          f"p90 {np.percentile(d3,90)*100:.1f}cm")
    print(f"  [track {tid}] foot z: lowest {m['foot_z_min']*100:.1f}cm  "
          f"median-min {m['foot_z_med']*100:.1f}cm  below-floor frac {m['penetration_frac']:.2f}")
    bs = "  ".join(f"{b}:{med:.3f}+-{sd:.3f}" for b, (med, sd) in m["bone"].items())
    print(f"  [track {tid}] bone len (median+-std m, shared-shape => low std): {bs}")
    s = m.get("sil")
    if s:
        print(f"  [track {tid}] silhouette: mean vert-outside-mask {s['mean_outside_px']:.2f}px  "
              f"frac-outside {s['frac_outside']:.2f}  ({s['n_views']} masked views) -- lower=tighter")


# driver
def run(tracked_path, cfg: Cfg, device, limit_frames=None, tracks_filter=None, masks_path=None):
    cameras = load_cameras()
    data = load_track_data(tracked_path, cameras, cfg)
    vposer = None
    if cfg.pose_prior == "vposer":
        from vposer import load_vposer
        vposer = load_vposer(device=device)
    masks = load_masks(masks_path) if masks_path else None
    if masks_path:
        print(f"[fit_smplx] silhouette masks: {'loaded ' + str(len(masks)) + ' views' if masks else 'NONE found at ' + str(masks_path) + ' (run masks.py)'}")
    out_doc = {"frames": {}, "tracks": {},
               "params": {k: getattr(cfg, k) for k in vars(Cfg) if not k.startswith("_")},
               "pose_prior": cfg.pose_prior, "silhouette": bool(masks)}
    metrics = {}
    for tid in sorted(data):
        if tracks_filter is not None and tid not in tracks_filter:
            continue
        track = data[tid]; track["tid"] = tid
        if limit_frames:
            for k in ("ts", "views"):
                track[k] = track[k][:limit_frames]
            track["targets"] = track["targets"][:limit_frames]
            track["w"] = track["w"][:limit_frames]
        print(f"[fit_smplx] track {tid}: {len(track['ts'])} frames")
        results, J, sil_metric = fit_track(track, cameras, cfg, device, vposer=vposer, masks=masks)
        m = _validate(track, J, cameras, cfg)
        m["sil"] = sil_metric
        _report_val(tid, m)
        metrics[tid] = m
        out_doc["tracks"][str(tid)] = {"betas": results[0]["betas"], "n_frames": len(results)}
        for r in results:
            out_doc["frames"].setdefault(str(r["t"]), []).append({"track_id": tid, **r})
    out_doc["timestamps"] = sorted(int(t) for t in out_doc["frames"])
    return out_doc, metrics


def _selftest(device) -> int:
    cfg = Cfg()
    cfg.stage1_iters, cfg.stage2_iters = 150, 120
    doc, metrics = run(DEFAULT_TRACKED, cfg, device, limit_frames=4,
                       masks_path=str(DEFAULT_MASKS) if DEFAULT_MASKS.exists() else None)
    assert doc["frames"], "no bodies fitted"
    for tid, m in metrics.items():
        assert np.median(m["px"]) < 60, f"track {tid} reprojection too high"
    print("[fit_smplx] selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracked", default=str(DEFAULT_TRACKED))
    ap.add_argument("--out", default=None, help="write fitted bodies JSON here")
    ap.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    ap.add_argument("--pose_prior", default="l2", choices=["l2", "vposer"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--limit_frames", type=int, default=None, help="cap frames/track (debug)")
    ap.add_argument("--tracks", default=None, help="comma-separated track ids to fit (default: all)")
    ap.add_argument("--stage1_iters", type=int, default=Cfg.stage1_iters)
    ap.add_argument("--stage2_iters", type=int, default=Cfg.stage2_iters)
    ap.add_argument("--masks", default=str(DEFAULT_MASKS),
                    help="masks.py output (.npz) for the silhouette term; absent file => no sil term")
    ap.add_argument("--w_silhouette", type=float, default=Cfg.w_silhouette,
                    help="silhouette term weight (0 disables; stage-2 only)")
    ap.add_argument("--sil_clamp_px", type=float, default=Cfg.sil_clamp_px,
                    help="GMoF scale (px) for the silhouette term; smaller = responds to small residuals")
    ap.add_argument("--fps", type=float, default=Cfg.fps,
                    help="source frame rate; converts timestamp gaps to seconds for the temporal "
                         "terms. MUST match the footage or the smoothing is mis-scaled.")
    ap.add_argument("--w_smooth_transl", type=float, default=Cfg.w_smooth_transl)
    ap.add_argument("--w_smooth_pose", type=float, default=Cfg.w_smooth_pose)
    ap.add_argument("--smooth_max_gap_s", type=float, default=Cfg.smooth_max_gap_s,
                    help="never smooth across a gap longer than this (s)")
    ap.add_argument("--accel_ref_transl", type=float, default=Cfg.accel_ref_transl)
    ap.add_argument("--accel_ref_pose", type=float, default=Cfg.accel_ref_pose)
    ap.add_argument("--gnc_sigma3d_start", type=float, default=Cfg.gnc_sigma3d_start,
                    help="graduated non-convexity: initial 3D robust scale (m). Set it to "
                         "sigma3d to disable annealing.")
    ap.add_argument("--gnc_sigma2d_start", type=float, default=Cfg.gnc_sigma2d_start)
    ap.add_argument("--gnc_frac", type=float, default=Cfg.gnc_frac)
    ap.add_argument("--w_beta", type=float, default=Cfg.w_beta,
                    help=("shape-prior weight: how strongly the body shape is kept near the average "
                          "shape. Lower values let the fit follow the athlete's real proportions; "
                          "run_bodies.py uses 0.005."))
    ap.add_argument("--free_betas", type=int, default=Cfg.free_betas,
                    help=("optimise only the leading N shape dimensions, pinning the rest at zero. "
                          "The fallback when a relaxed prior starts absorbing calibration bias "
                          "instead of individualising shape."))
    ap.add_argument("--hip_widen_m", type=float, default=Cfg.hip_widen_m,
                    help="COCO-vs-SMPL-X hip convention gap (m); 0 disables the correction")
    ap.add_argument("--no_silhouette", action="store_true", help="disable the silhouette term (baseline)")
    ap.add_argument("--no_truncation", action="store_true", help="disable the truncated-view 2D down-weight")
    ap.add_argument("--check", action="store_true", help="just load SMPL-X + print joint layout")
    ap.add_argument("--selftest", action="store_true", help="fit a few frames and assert sanity")
    args = ap.parse_args()

    if args.check:
        raise SystemExit(_check(args.gender))

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"[fit_smplx] device: {device}")
    if args.selftest:
        raise SystemExit(_selftest(device))

    cfg = Cfg()
    cfg.gender = args.gender
    cfg.pose_prior = args.pose_prior
    cfg.fps = args.fps
    cfg.w_smooth_transl = args.w_smooth_transl
    cfg.w_smooth_pose = args.w_smooth_pose
    cfg.smooth_max_gap_s = args.smooth_max_gap_s
    cfg.accel_ref_transl = args.accel_ref_transl
    cfg.accel_ref_pose = args.accel_ref_pose
    cfg.gnc_sigma3d_start = args.gnc_sigma3d_start
    cfg.gnc_sigma2d_start = args.gnc_sigma2d_start
    cfg.gnc_frac = args.gnc_frac
    cfg.hip_widen_m = args.hip_widen_m
    cfg.w_beta = args.w_beta
    cfg.free_betas = args.free_betas
    cfg.stage1_iters = args.stage1_iters
    cfg.stage2_iters = args.stage2_iters
    cfg.w_silhouette = 0.0 if args.no_silhouette else args.w_silhouette
    cfg.sil_clamp_px = args.sil_clamp_px
    cfg.truncation = not args.no_truncation
    tracks_filter = None if not args.tracks else {int(x) for x in args.tracks.split(",")}
    masks_path = None if args.no_silhouette else args.masks

    doc, _ = run(args.tracked, cfg, device, limit_frames=args.limit_frames,
                 tracks_filter=tracks_filter, masks_path=masks_path)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(doc), encoding="utf-8")
        print(f"[fit_smplx] wrote {args.out}  ({len(doc['frames'])} timestamps, "
              f"{len(doc['tracks'])} tracks)")


if __name__ == "__main__":
    main()
