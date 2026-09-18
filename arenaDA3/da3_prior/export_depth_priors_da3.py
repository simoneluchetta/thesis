"""Runs Depth Anything 3 on the arena cameras and saves, for each camera, a depth map and a
confidence map into the --out folder. The static world fusion (da3_fusion) reads them.

The model is told where every camera is, so its depths agree between cameras. Its output is
still scale-and-shift free, so each map is fitted to the known floor before it is used.

see: https://github.com/ByteDance-Seed/Depth-Anything-3
and: https://arxiv.org/abs/1907.01341 (scale-and-shift-invariant depth)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent          # arenaDA3/da3_prior
ARENA = HERE.parent                             # arenaDA3
REPO = ARENA.parent                             # repository root
if str(ARENA) not in sys.path:
    sys.path.insert(0, str(ARENA))
# Same plates and same intrinsics the NeRF builds its rays from, so the maps line up with it.
from arena_cameras import PERIMETER, load_cameras, prep_plate, make_rays, ray_aabb  # noqa: E402


# The next two functions are the same as in the earlier NeRF depth-prior exporter, so the
# quality numbers can be compared.
def floor_analytic_distance(Ks, cam, w, h, bbmin, bbmax):
    """How far away the floor is at every pixel, from the calibration alone."""
    origins, dirs = make_rays(Ks, cam["R_c2w"], cam["C"], w, h)
    near, far, valid = ray_aabb(origins, dirs, bbmin, bbmax)
    dz = dirs[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = -origins[:, 2] / dz
    fp = origins + t_floor[:, None] * dirs
    floor = ((dz < -1e-3) & (t_floor > 0) & valid
             & (np.abs(fp[:, 0]) < 15) & (np.abs(fp[:, 1]) < 9))
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    x = (i + 0.5 - Ks[0, 2]) / Ks[0, 0]
    y = (j + 0.5 - Ks[1, 2]) / Ks[1, 1]
    raynorm = np.sqrt(x * x + y * y + 1.0).reshape(-1).astype(np.float32)
    return t_floor.astype(np.float32), floor, raynorm


def fit_affine(d_pred_dist, d_floor, floor_mask):
    """The scale and offset that line the prediction up with the real floor.

    RANSAC first, so a region the network got badly wrong cannot set the answer. The
    correlation that comes back is the trust metric for this camera.
    """
    m = floor_mask & np.isfinite(d_pred_dist) & (d_pred_dist > 1e-4) & np.isfinite(d_floor)
    dp = d_pred_dist[m]; dg = d_floor[m]
    if dp.size < 200:
        return 1.0, 0.0, int(dp.size), 1.0, 0.0
    rng = np.random.default_rng(0)
    best_in, best_n = None, -1
    n = dp.size
    for _ in range(300):
        i, k = rng.integers(0, n, size=2)
        if abs(dp[i] - dp[k]) < 1e-6:
            continue
        aa = (dg[i] - dg[k]) / (dp[i] - dp[k])
        if aa <= 0:
            continue
        bb = dg[i] - aa * dp[i]
        inl = np.abs(aa * dp + bb - dg) < (0.10 * np.maximum(dg, 1.0))
        if inl.sum() > best_n:
            best_n, best_in = int(inl.sum()), inl
    if best_in is not None and best_n >= 200:
        A = np.polyfit(dp[best_in], dg[best_in], 1)
        a, b = float(A[0]), float(A[1])
    else:
        a, b = float(np.median(dg / np.clip(dp, 1e-6, None))), 0.0
    resid = float(np.median(np.abs(a * dp + b - dg) / np.maximum(dg, 1.0)))
    pear = float(np.corrcoef(dp, dg)[0, 1]) if dp.std() > 1e-9 else 0.0
    return a, b, int(dp.size), resid, pear


def nan_resize(arr, w, h):
    """Resize a depth map without letting invalid pixels smear into their neighbours.

    A plain interpolation would average real depths with holes and give wrong numbers.
    Resizing the mask separately keeps the holes as holes.
    """
    valid = np.isfinite(arr) & (arr > 1e-4)
    fill = np.where(valid, arr, np.median(arr[valid]) if valid.any() else 0.0).astype(np.float32)
    res = cv2.resize(fill, (w, h), interpolation=cv2.INTER_LINEAR)
    mres = cv2.resize(valid.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
    res[~mres] = np.nan
    return res


def spearman(x, y):
    """Spearman correlation, done by hand to avoid a scipy dependency in this lane.

    Rank correlation rather than Pearson, because what matters is whether the network puts
    things in the right order, not whether its numbers are the right size.
    """
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    if x.size < 4 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def w2c_4x4(cam):
    """The camera pose as a single 4x4 matrix, the way open3d and DA3 both want it."""
    T = np.eye(4, dtype=np.float64)
    R = cam["R_c2w"].T
    T[:3, :3] = R
    T[:3, 3] = -R @ cam["C"]
    return T


def load_tie_points(path):
    """The triangulated points above floor level, with the cameras that saw them.

    A camera counts as seeing a point when the triangulation kept its observation.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for tp in data["tie_points"]:
        W = np.asarray(tp["world"], np.float64)
        if W[2] <= 0.5:
            continue
        vis = {o["cam"] for o in tp.get("observations", []) if o.get("used")}
        if vis:
            out.append((tp.get("id", "?"), W, vis))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pnp", default=str(REPO / "diagnostics/court_pnp_cameras_redfox_repaired.json"),
                    help="camera intrinsics; must be the pair of --poses")
    ap.add_argument("--poses", default=str(REPO / "placements/placed_cameras_redfox_repaired.json"))
    ap.add_argument("--plates", required=True, help="folder with one image per camera, <cam>_<plate_n>.png")
    ap.add_argument("--plate_n", type=int, default=30)
    ap.add_argument("--ties", default=str(REPO / "annotations/court_tiepoints_triangulated.json"),
                    help="known points above the floor, only used for the printed quality check")
    ap.add_argument("--model", default="depth-anything/DA3-LARGE-1.1",
                    help="0.35B; GIANT/NESTED need >10GB + the gsplat head (dead here)")
    ap.add_argument("--long_side", type=int, default=1024,
                    help="cached-prior resolution (the frozen/MASt3R exports' contract)")
    ap.add_argument("--process_res", type=int, default=1008,
                    help="DA3 internal processing res; auto-falls back 1008->756->504 on OOM")
    ap.add_argument("--ext_convention", choices=["w2c", "c2w"], default="w2c",
                    help="what DA3 expects in `extrinsics`; w2c(OpenCV) per its own exports. "
                         "If the floor gate fails absurdly, re-run once with c2w to rule out convention")
    ap.add_argument("--cams", default="", help="comma subset for the smoke (default: all perimeter)")
    ap.add_argument("--bbox", default="-26,-22,-0.6,29,14,12")
    ap.add_argument("--out", required=True)
    ap.add_argument("--recon_dir", default="", help="also copy the preview images here (default: <out>)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    recon = Path(args.recon_dir) if args.recon_dir else out
    recon.mkdir(parents=True, exist_ok=True)
    bb = [float(v) for v in args.bbox.split(",")]
    bbmin, bbmax = bb[:3], bb[3:]
    names = [c for c in args.cams.split(",") if c] or list(PERIMETER)
    t0 = time.time()
    print(f"== DA3 depth-prior export == model={args.model} device={device} cams={names}")

    cams = load_cameras(Path(args.pnp), Path(args.poses), names)
    plates, Ks_all, wh = {}, {}, {}
    for n in names:
        rgb, Ks, w, h = prep_plate(Path(args.plates) / f"{n}_{args.plate_n}.png", cams[n], args.long_side)
        plates[n], Ks_all[n], wh[n] = rgb, Ks, (w, h)
    images = [(np.clip(plates[n], 0, 1) * 255).astype(np.uint8) for n in names]
    exts = np.stack([w2c_4x4(cams[n]) for n in names])
    if args.ext_convention == "c2w":
        exts = np.stack([np.linalg.inv(T) for T in exts])
    ixts = np.stack([Ks_all[n] for n in names]).astype(np.float64)

    from depth_anything_3.api import DepthAnything3
    print(f"loading {args.model} ...")
    # see: https://github.com/ByteDance-Seed/Depth-Anything-3
    # weights: https://huggingface.co/depth-anything/DA3-LARGE-1.1
    model = DepthAnything3.from_pretrained(args.model).to(device)
    if hasattr(model, "eval"):
        model.eval()

    pred = None
    for res in dict.fromkeys([args.process_res, 756, 504]):     # ordered, deduped fallback chain
        try:
            t1 = time.time()
            pred = model.inference(images, extrinsics=exts, intrinsics=ixts, process_res=res)
            print(f"inference OK at process_res={res} ({time.time()-t1:.1f}s, "
                  f"peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB)"
                  if device == "cuda" else f"inference OK at process_res={res}")
            break
        except torch.cuda.OutOfMemoryError:
            print(f"OOM at process_res={res} -> stepping down")
            torch.cuda.empty_cache()
    if pred is None:
        raise SystemExit("DA3 inference OOM even at 504 - reduce cams or use DA3-BASE")

    depth = np.asarray(pred.depth, np.float32)                   # (N, Hp, Wp) camera-z
    conf = np.asarray(pred.conf, np.float32) if getattr(pred, "conf", None) is not None else None

    ties = load_tie_points(args.ties) if Path(args.ties).exists() else []
    print(f"{len(ties)} off-floor tie points (z>0.5) for the rank gate")

    meta = {"backbone": args.model, "source": "DA3 pose-conditioned multi-view (all cams jointly)",
            "ext_convention": args.ext_convention, "process_res_used": int(depth.shape[2]),
            "poses": str(Path(args.poses).resolve()), "pnp": str(Path(args.pnp).resolve()),
            "plates": str(Path(args.plates).resolve()), "plate_n": args.plate_n,
            "long_side": args.long_side,
            "note": "camera-z; NeRF consumes scale-and-shift-invariantly per cam; conf cached for the fusion stage",
            "cams": {}}
    rep = ["# DA3 depth priors: pose-conditioned multi-view depth as the SSI mono prior",
           "",
           f"model `{args.model}` · ONE joint inference over {len(names)} plates WITH exact "
           f"poses/intrinsics · camera-z up-to-scale (trainer normalises per cam).",
           "",
           "Trust metrics are the SAME as the frozen/MASt3R exports (directly comparable):",
           "floor = affine fit to the known z=0 plane; tie rho = per-cam floor-calibrated pooled",
           "Spearman on the triangulated off-floor tie points (MASt3R fair number: +0.51).", "",
           "| cam | floor px | affine a | b | resid | pearson | ties | rho(cam) | depth p1/p50/p99 |",
           "|---|---|---|---|---|---|---|---|---|"]

    pooled_pred, pooled_true = [], []
    previews = []
    for idx, n in enumerate(names):
        w, h = wh[n]
        d_cz = nan_resize(depth[idx], w, h)                      # camera-z at the contract res
        c_map = nan_resize(conf[idx], w, h) if conf is not None else None
        t_floor, floor_mask, raynorm = floor_analytic_distance(Ks_all[n], cams[n], w, h, bbmin, bbmax)
        d_dist = (d_cz.reshape(-1) * raynorm)                    # camera-z -> ray distance
        a, b, n_fl, resid, pear = fit_affine(d_dist, t_floor, floor_mask)

        # The second test. Project each known off-floor point into this camera, take the median
        # predicted depth over a small patch around it, and collect that against the truth.
        cam = cams[n]; Ks = Ks_all[n]
        cam_pred, cam_true = [], []
        for _tid, W, vis in ties:
            if n not in vis:
                continue
            pc = cam["R_c2w"].T @ (W - cam["C"])
            if pc[2] < 0.1:
                continue
            u = Ks[0, 0] * pc[0] / pc[2] + Ks[0, 2]
            v = Ks[1, 1] * pc[1] / pc[2] + Ks[1, 2]
            ui, vi = int(round(u)), int(round(v))
            if not (2 <= ui < w - 2 and 2 <= vi < h - 2):
                continue
            patch = d_cz[vi - 2:vi + 3, ui - 2:ui + 3]
            good = patch[np.isfinite(patch) & (patch > 1e-4)]
            if good.size < 5:
                continue
            rn = raynorm[vi * w + ui]
            cam_pred.append(a * float(np.median(good)) * rn + b)  # floor-calibrated ray distance
            cam_true.append(float(np.linalg.norm(W - cam["C"])))
        pooled_pred += cam_pred; pooled_true += cam_true
        rho_cam = spearman(cam_pred, cam_true)

        np.save(out / f"{n}_depth.npy", d_cz)
        if c_map is not None:
            np.save(out / f"{n}_conf.npy", c_map)
        dv = d_cz[np.isfinite(d_cz) & (d_cz > 1e-4)]
        p1, p50, p99 = (np.percentile(dv, [1, 50, 99]) if dv.size else (0, 0, 0))
        meta["cams"][n] = {"w": int(w), "h": int(h), "floor_px": n_fl, "affine_a": a, "affine_b": b,
                           "floor_resid": resid, "floor_pearson": pear,
                           "n_tie": len(cam_pred), "tie_rho": None if np.isnan(rho_cam) else rho_cam,
                           "p1": float(p1), "p50": float(p50), "p99": float(p99)}
        rep.append(f"| {n} | {n_fl} | {a:+.4f} | {b:+.3f} | {resid:.3f} | {pear:+.3f} | "
                   f"{len(cam_pred)} | {'-' if np.isnan(rho_cam) else f'{rho_cam:+.2f}'} | "
                   f"{p1:.2f}/{p50:.2f}/{p99:.2f} |")

        # photograph, depth and confidence side by side
        lo, hi = (np.percentile(dv, 2), np.percentile(dv, 98)) if dv.size else (0.0, 1.0)
        dn = np.clip((np.nan_to_num(d_cz, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1)
        dcol = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        gt = cv2.cvtColor((np.clip(plates[n], 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        panels = [gt, dcol]
        if c_map is not None:
            cn = np.clip(np.nan_to_num(c_map, nan=0.0), 0, np.percentile(c_map[np.isfinite(c_map)], 99) + 1e-6)
            cn = (255 * cn / max(cn.max(), 1e-6)).astype(np.uint8)
            panels.append(cv2.cvtColor(cn, cv2.COLOR_GRAY2BGR))
        sep = np.full((h, 4, 3), 255, np.uint8)
        strip = panels[0]
        for p in panels[1:]:
            strip = np.concatenate([strip, sep, p], axis=1)
        cv2.putText(strip, f"DA3 depth prior {n}  r={pear:+.2f} rho={'-' if np.isnan(rho_cam) else f'{rho_cam:+.2f}'}",
                    (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{n}_preview.png"), strip)
        cv2.imwrite(str(recon / f"depth_prior_da3_{n}.png"), strip)
        previews.append(strip)
        print(f"  {n}: floor {n_fl:>6} px  pearson {pear:+.3f}  ties {len(cam_pred):>2} "
              f"rho {'-' if np.isnan(rho_cam) else f'{rho_cam:+.2f}'}  p50 {p50:.2f}  -> {n}_depth.npy")

    mean_pear = float(np.mean([meta["cams"][n]["floor_pearson"] for n in names]))
    rho_pool = spearman(pooled_pred, pooled_true)
    meta["mean_floor_pearson"] = mean_pear
    meta["pooled_offfloor_tie_rho"] = None if np.isnan(rho_pool) else float(rho_pool)
    meta["gate_floor"] = bool(mean_pear >= 0.95)
    meta["gate_rho"] = bool((not np.isnan(rho_pool)) and rho_pool > 0.51)
    (out / "depth_priors_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rep += ["",
            f"**mean floor pearson {mean_pear:+.3f}** (gate >= +0.95; MASt3R +0.956, frozen ~+0.96)",
            f"**pooled off-floor tie rho {rho_pool:+.3f}** over {len(pooled_pred)} obs "
            f"(gate > +0.51 = MASt3R's fair number)",
            "",
            f"GATES: floor {'PASS' if meta['gate_floor'] else 'FAIL'} · "
            f"rho {'PASS' if meta['gate_rho'] else 'FAIL'}. "
            + ("The depth maps are fine to fuse." if meta["gate_floor"] and meta["gate_rho"] else
               "Check the preview images before trusting the fused world.")]
    (out / "depth_priors_report.md").write_text("\n".join(rep), encoding="utf-8")

    if previews:
        wmax = max(s.shape[1] for s in previews)
        previews = [cv2.copyMakeBorder(s, 0, 0, 0, wmax - s.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    for s in previews]
        cv2.imwrite(str(recon / "depth_priors_da3_ALL.png"), np.concatenate(previews, axis=0))

    print(f"\nDONE in {time.time()-t0:.1f}s -> {out}")
    print(f"GATE floor pearson : {mean_pear:+.3f}  ({'PASS' if meta['gate_floor'] else 'FAIL'}, need >= +0.95)")
    print(f"GATE tie rho       : {rho_pool:+.3f}  ({'PASS' if meta['gate_rho'] else 'FAIL'}, need >  +0.51)"
          f"   [n={len(pooled_pred)}]")


if __name__ == "__main__":
    main()
