"""The same world, rebuilt once per instant, so it can be played back over time.

da3_fusion produces one still cloud of the hall. This runs exactly the same recipe over and
over, once for every 5th frame of the game footage. The output is a sequence of full-world
clouds the viewer can scrub through with the athletes in them.

The ten cameras are synchronised, so the ten frames of one instant are a multi-view set of
one scene. A running player is just another object seen from ten angles, and the depth
network handles them like the walls.

The calibration must be the pair the bodies were fitted with, or the clouds and the bodies
land in different places.

The floor fit is redone every instant, now with people standing on the floor. RANSAC handles
that, since the players are a minority of the floor pixels, but a poor fit is replaced by the
camera's running median so one bad frame cannot make the world jump. Every fit is logged to
<out>/diag.jsonl.

Safe to interrupt and rerun: instants that already have a PLY are skipped unless --force.

Needs the DA3 environment (arenaDA3/run_world.py runs it for you):
  uv run --project arenaDA3/da3_prior python arenaDA3/da3_4d/fuse_4d.py --frames work/frames --n_frames 750 --web_out work/world/da3_4d --out work/world/4d
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

HERE = Path(__file__).resolve().parent          # arenaDA3/da3_4d
ARENA = HERE.parent                             # arenaDA3
REPO = ARENA.parent                             # repository root
for p in (str(ARENA), str(ARENA / "da3_prior")):
    if p not in sys.path:
        sys.path.insert(0, p)
from arena_cameras import PERIMETER, load_cameras, prep_plate            # noqa: E402
from export_depth_priors_da3 import (                                 # noqa: E402
    floor_analytic_distance, fit_affine, nan_resize, w2c_4x4)

# A floor fit worse than this is not trusted, and the camera's running median is used instead.
MIN_FLOOR_PEARSON = 0.85
MIN_FLOOR_PX = 3000


# see: https://en.wikipedia.org/wiki/PLY_(file_format)
def write_ply_xyzrgb(path: Path, pts: np.ndarray, cols: np.ndarray) -> None:
    """Write the cloud as float32 xyz plus uint8 rgb, 15 bytes a point.

    open3d would write doubles, twice the size, and the browser loads 150 of these files.
    """
    n = len(pts)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n").encode("ascii")
    rec = np.empty(n, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    rec["xyz"] = pts.astype(np.float32)
    rec["rgb"] = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(header)
        rec.tofile(f)


def fuse_instant(t, names, cams, plates, Ks_all, wh, depth, conf, geo, affine_hist, args):
    """One TSDF fusion from this instant's DA3 depths. Returns (cloud pts, cols, diag)."""
    # see: https://www.open3d.org/docs/release/tutorial/pipelines/rgbd_integration.html
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel, sdf_trunc=args.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    cam_diag = {}
    for idx, n in enumerate(names):
        cam = cams[n]
        w, h = wh[n]
        Ks = Ks_all[n]
        d_cz = nan_resize(depth[idx], w, h)
        c_map = nan_resize(conf[idx], w, h) if conf is not None else None
        t_floor, floor_mask, raynorm = geo[n]

        d_dist = d_cz.reshape(-1) * raynorm
        a, b, n_fl, resid, pear = fit_affine(d_dist, t_floor, floor_mask)
        used_fallback = False
        if (pear < MIN_FLOOR_PEARSON or n_fl < MIN_FLOOR_PX) and affine_hist.get(n):
            hist = np.asarray(affine_hist[n])
            a, b = float(np.median(hist[:, 0])), float(np.median(hist[:, 1]))
            used_fallback = True
        elif pear >= MIN_FLOOR_PEARSON and n_fl >= MIN_FLOOR_PX:
            affine_hist.setdefault(n, []).append((a, b))

        rn2 = raynorm.reshape(h, w)
        dz = (a * (d_cz * rn2) + b) / rn2
        bad = ~np.isfinite(dz) | (dz <= 0.3) | (dz > args.max_depth)
        if c_map is not None:
            thr = np.nanpercentile(c_map, args.conf_pct)
            bad |= ~(np.nan_to_num(c_map, nan=-1.0) >= thr)
        dz = np.where(bad, 0.0, dz).astype(np.float32)

        color = o3d.geometry.Image((np.clip(plates[n], 0, 1) * 255).astype(np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, o3d.geometry.Image(dz), depth_scale=1.0, depth_trunc=args.max_depth,
            convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, Ks[0, 0], Ks[1, 1], Ks[0, 2], Ks[1, 2])
        vol.integrate(rgbd, intr, w2c_4x4(cam))
        cam_diag[n] = dict(a=round(a, 4), b=round(b, 3), pearson=round(pear, 3),
                           floor_px=int(n_fl), fallback=used_fallback,
                           px=int((~bad).sum()))

    pc = vol.extract_point_cloud()
    pts = np.asarray(pc.points)
    cols = np.asarray(pc.colors)
    bbmin = np.asarray(args.bb[:3])
    bbmax = np.asarray(args.bb[3:])
    inside = np.all((pts >= bbmin) & (pts <= bbmax), axis=1)
    diag = dict(t=int(t), cams=cam_diag, n_raw=int(len(pts)),
                floater_frac=round(1.0 - float(inside.mean()), 4) if len(pts) else None)
    pts, cols = pts[inside], cols[inside]
    if len(pts) > args.max_points:
        keep = np.random.default_rng(int(t)).choice(len(pts), args.max_points, replace=False)
        pts, cols = pts[keep], cols[keep]
    diag["n_final"] = int(len(pts))
    return pts, cols, diag


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="depth-anything/DA3-LARGE-1.1")
    ap.add_argument("--long_side", type=int, default=1024)
    ap.add_argument("--process_res", type=int, default=1008,
                    help="DA3 internal res; auto-falls back 1008->756->504 per instant on OOM")
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--sdf_trunc", type=float, default=0.15)
    ap.add_argument("--conf_pct", type=float, default=30.0)
    ap.add_argument("--max_depth", type=float, default=60.0)
    ap.add_argument("--max_points", type=int, default=450_000)
    ap.add_argument("--bbox", default="-32,-26,-1,32,20,14",
                    help="arena crop; smoke prints raw extents to tighten it")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N instants (smoke)")
    ap.add_argument("--stride", type=int, default=1, help="take every k-th instant of the timeline")
    ap.add_argument("--force", action="store_true", help="redo instants whose PLY already exists")
    ap.add_argument("--preview_every", type=int, default=0,
                    help="write a top-down scatter for every k-th processed instant (0 = off)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frames", required=True, help="frame folder: <frames>/<cam>/tNNNNNN.jpg")
    ap.add_argument("--n_frames", type=int, required=True, help="number of frames in the folder")
    ap.add_argument("--every", type=int, default=5, help="fuse every Nth frame (5 = 5 per second at 25 fps)")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--pnp", default=str(REPO / "diagnostics/court_pnp_cameras_redfox_repaired.json"))
    ap.add_argument("--placer", default=str(REPO / "placements/placed_cameras_redfox_repaired.json"))
    ap.add_argument("--web_out", required=True, help="where the per-instant PLYs and index.json go")
    ap.add_argument("--out", required=True, help="where the diagnostics go")
    args = ap.parse_args()
    args.bb = [float(v) for v in args.bbox.split(",")]
    RAW, PNP, PLACER = Path(args.frames), Path(args.pnp), Path(args.placer)
    WEB_OUT, OUT = Path(args.web_out), Path(args.out)

    WEB_OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    ts_all = list(range(0, args.n_frames, args.every))
    ts = ts_all[:: args.stride]
    if args.limit:
        ts = ts[: args.limit]
    names = list(PERIMETER)
    cams = load_cameras(PNP, PLACER, names)
    print(f"== DA3 4D fusion == {len(ts)}/{len(ts_all)} instants, cams={names}")
    print(f"   calib: {PLACER.name} + {PNP.name}, frames: {RAW}")

    from depth_anything_3.api import DepthAnything3
    print(f"loading {args.model} on {device} ...")
    # see: https://github.com/ByteDance-Seed/Depth-Anything-3
    model = DepthAnything3.from_pretrained(args.model).to(device)
    if hasattr(model, "eval"):
        model.eval()

    # The cameras are bolted to the walls, so everything geometric is the same at every
    # instant. Work it out once from the first frame rather than 150 times.
    Ks_all, wh, geo = {}, {}, {}
    for n in names:
        _, Ks, w, h = prep_plate(RAW / n / f"t{ts[0]:06d}.jpg", cams[n], args.long_side)
        Ks_all[n], wh[n] = Ks, (w, h)
        geo[n] = floor_analytic_distance(Ks, cams[n], w, h, args.bb[:3], args.bb[3:])

    index_path = WEB_OUT / "index.json"
    frames_idx = {}
    if index_path.exists() and not args.force:
        try:
            frames_idx = {f["t"]: f for f in json.loads(index_path.read_text(encoding="utf-8"))["frames"]}
        except Exception:
            frames_idx = {}

    affine_hist: dict[str, list] = {}
    diag_f = open(OUT / "diag.jsonl", "a", encoding="utf-8")
    t_run = time.time()
    done = 0
    for t in ts:
        ply_path = WEB_OUT / f"t{t:06d}.ply"
        if ply_path.exists() and not args.force and t in frames_idx:
            done += 1
            continue
        t0 = time.time()

        plates = {}
        for n in names:
            fp = RAW / n / f"t{t:06d}.jpg"
            if not fp.exists():
                raise FileNotFoundError(f"missing {fp}: the frame cache is incomplete at t={t}")
            plates[n], _, _, _ = prep_plate(fp, cams[n], args.long_side)
        images = [(np.clip(plates[n], 0, 1) * 255).astype(np.uint8) for n in names]
        exts = np.stack([w2c_4x4(cams[n]) for n in names])
        ixts = np.stack([Ks_all[n] for n in names]).astype(np.float64)

        pred = None
        for res in dict.fromkeys([args.process_res, 756, 504]):
            try:
                with torch.no_grad():
                    pred = model.inference(images, extrinsics=exts, intrinsics=ixts, process_res=res)
                break
            except torch.cuda.OutOfMemoryError:
                # out of VRAM: drop the resolution and retry instead of failing the whole run
                print(f"  t{t:06d}: out of memory at process_res={res}, stepping down")
                torch.cuda.empty_cache()
        if pred is None:
            raise SystemExit(f"t{t:06d}: DA3 OOM even at 504")
        depth = np.asarray(pred.depth, np.float32)
        conf = np.asarray(pred.conf, np.float32) if getattr(pred, "conf", None) is not None else None

        pts, cols, diag = fuse_instant(t, names, cams, plates, Ks_all, wh,
                                       depth, conf, geo, affine_hist, args)
        write_ply_xyzrgb(ply_path, pts, cols)
        diag["secs"] = round(time.time() - t0, 1)
        diag_f.write(json.dumps(diag) + "\n")
        diag_f.flush()

        frames_idx[t] = dict(t=int(t), file=f"da3_4d/{ply_path.name}", points=int(len(pts)))
        manifest = dict(
            session="game",
            source="DA3 pose-conditioned multi-view depth -> per-instant TSDF fusion",
            calibration=[PLACER.name, PNP.name], model=args.model, fps=args.fps,
            recipe=dict(voxel=args.voxel, sdf_trunc=args.sdf_trunc, conf_pct=args.conf_pct,
                        max_depth=args.max_depth, long_side=args.long_side, bbox=args.bb),
            point_size=0.05,
            frames=[frames_idx[q] for q in sorted(frames_idx)])
        index_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

        if args.preview_every and (done % args.preview_every == 0):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            sub = slice(None, None, max(1, len(pts) // 200_000))
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.scatter(pts[sub, 0], pts[sub, 1], s=0.3, c=np.clip(cols[sub], 0, 1), linewidths=0)
            ax.set_aspect("equal")
            ax.set_title(f"da3_4d t{t:06d} ({len(pts):,} pts)")
            fig.savefig(OUT / f"preview_t{t:06d}.png", dpi=100, bbox_inches="tight")
            plt.close(fig)

        done += 1
        lo = pts.min(0) if len(pts) else np.zeros(3)
        hi = pts.max(0) if len(pts) else np.zeros(3)
        print(f"  [{done}/{len(ts)}] t{t:06d}: {len(pts):>7,} pts "
              f"(raw {diag['n_raw']:,}, floaters {diag['floater_frac']:.1%}) "
              f"ext x[{lo[0]:+.1f},{hi[0]:+.1f}] y[{lo[1]:+.1f},{hi[1]:+.1f}] "
              f"z[{lo[2]:+.1f},{hi[2]:+.1f}]  {diag['secs']}s")

    diag_f.close()
    print(f"\nDONE {done}/{len(ts)} instants in {(time.time()-t_run)/60:.1f} min -> {WEB_OUT}")
    print(f"index: {index_path} · diagnostics: {OUT/'diag.jsonl'}")


if __name__ == "__main__":
    main()
