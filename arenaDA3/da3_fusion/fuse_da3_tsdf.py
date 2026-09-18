"""Fuses ten depth maps into one surface model of the building.

Each camera's depth map is one estimate of the shape of the arena. TSDF fusion combines
them: for every point in space it accumulates how far in front of or behind a surface each
camera puts it, and the surface is where those values cross zero. Disagreement averages out
instead of doubling up.
see: https://en.wikipedia.org/wiki/Signed_distance_function

On the empty arena the floor lands on z = 0 to within 6 mm and the cameras agree to about
6 cm. The bench and the net come out to within half a metre. Thin things do not work: a hoop
rim is a few centimetres of metal against a wall eight metres behind it, and the depth snaps
to the wall. DA3 gives those pixels low confidence, so the confidence gate removes most of
them.

Written into --out: the fused cloud, a mesh, two scatter plots and a report. The cameras,
their calibration and the plates come from the depth export (depth_priors_meta.json).

Run from the repo root (arenaDA3/run_world.py does this for you):
  uv run --project arenaDA3/da3_prior python arenaDA3/da3_fusion/fuse_da3_tsdf.py --priors <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

HERE = Path(__file__).resolve().parent
ARENA = HERE.parent
REPO = ARENA.parent
if str(ARENA) not in sys.path:
    sys.path.insert(0, str(ARENA))
from arena_cameras import load_cameras, prep_plate  # noqa: E402

BB_MIN = np.array([-26.0, -22.0, -1.0])
BB_MAX = np.array([29.0, 14.0, 12.0])


def w2c_4x4(cam):
    """Camera pose the way open3d wants it: one 4x4 matrix, world into the camera."""
    T = np.eye(4)
    R = cam["R_c2w"].T
    T[:3, :3] = R
    T[:3, 3] = -R @ cam["C"]
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--sdf_trunc", type=float, default=0.15)
    ap.add_argument("--conf_pct", type=float, default=30.0,
                    help="throw away the least confident pixels of each map. Mostly this "
                         "removes thin structure, which the network gets wrong anyway.")
    ap.add_argument("--max_depth", type=float, default=60.0)
    ap.add_argument("--priors", required=True, help="folder written by export_depth_priors_da3.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    PRIORS, OUT = Path(args.priors), Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    meta = json.loads((PRIORS / "depth_priors_meta.json").read_text(encoding="utf-8"))
    names = list(meta["cams"].keys())
    # the same calibration pair and plates the depth maps were made from, never a different one
    cams = load_cameras(Path(meta["pnp"]), Path(meta["poses"]), names)

    # TSDF fusion
    # see: https://www.open3d.org/docs/release/tutorial/pipelines/rgbd_integration.html
    # and: https://www.open3d.org/docs/release/python_api/open3d.pipelines.integration.ScalableTSDFVolume.html
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel, sdf_trunc=args.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for n in names:
        cam = cams[n]
        rgb, Ks, w, h = prep_plate(Path(meta["plates"]) / f"{n}_{meta['plate_n']}.png", cam, meta["long_side"])
        d = np.load(PRIORS / f"{n}_depth.npy")
        conf_p = PRIORS / f"{n}_conf.npy"
        conf = np.load(conf_p) if conf_p.exists() else None

        i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
        x = (i + 0.5 - Ks[0, 2]) / Ks[0, 0]
        y = (j + 0.5 - Ks[1, 2]) / Ks[1, 1]
        raynorm = np.sqrt(x * x + y * y + 1.0)
        a, b = meta["cams"][n]["affine_a"], meta["cams"][n]["affine_b"]
        dz = (a * (d * raynorm) + b) / raynorm                       # into metres, via the floor fit.
        # The scale and offset were fitted on distance along the ray, so the conversion has to
        # go out to ray distance and back rather than being applied to the axis depth directly.

        bad = ~np.isfinite(dz) | (dz <= 0.3) | (dz > args.max_depth)
        if conf is not None:
            thr = np.nanpercentile(conf, args.conf_pct)
            bad |= ~(np.nan_to_num(conf, nan=-1.0) >= thr)
        dz = np.where(bad, 0.0, dz).astype(np.float32)               # open3d treats 0 as no data

        color = o3d.geometry.Image((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        depth = o3d.geometry.Image(dz)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1.0, depth_trunc=args.max_depth,
            convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, Ks[0, 0], Ks[1, 1], Ks[0, 2], Ks[1, 2])
        vol.integrate(rgbd, intr, w2c_4x4(cam))
        print(f"  integrated {n}: {int((~bad).sum()):,} px "
              f"(conf>=p{args.conf_pct:.0f}, depth<= {args.max_depth} m)")

    pc = vol.extract_point_cloud()
    pts = np.asarray(pc.points)
    inside = np.all((pts >= BB_MIN) & (pts <= BB_MAX), axis=1)
    floater_frac = 1.0 - float(inside.mean())
    pc_in = pc.select_by_index(np.where(inside)[0])
    o3d.io.write_point_cloud(str(OUT / "da3_world.ply"), pc_in)

    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    v = np.asarray(mesh.vertices)
    keep_v = np.all((v >= BB_MIN) & (v <= BB_MAX), axis=1)
    mesh = mesh.select_by_index(np.where(keep_v)[0].tolist(), cleanup=True) \
        if hasattr(mesh, "select_by_index") else mesh
    o3d.io.write_triangle_mesh(str(OUT / "da3_world_mesh.ply"), mesh)

    # Three checks: how much of the cloud is outside the building, whether the floor is flat
    # and at zero, and how close it passes to the tie points.
    P = np.asarray(pc_in.points)
    floor_band = P[np.abs(P[:, 2]) < 0.5]
    floor_medz = float(np.median(floor_band[:, 2])) if len(floor_band) else float("nan")
    floor_stdz = float(np.std(floor_band[:, 2])) if len(floor_band) else float("nan")
    kd = o3d.geometry.KDTreeFlann(pc_in)
    ties_path = REPO / "annotations/court_tiepoints_triangulated.json"
    ties = json.loads(ties_path.read_text(encoding="utf-8")) if ties_path.exists() else {"tie_points": []}
    hits = {}
    for tp in ties["tie_points"]:
        W = np.asarray(tp["world"])
        if W[2] <= 0.5:
            continue
        grp = tp.get("group", "?")
        _, _, dist2 = kd.search_knn_vector_3d(W, 1)
        hits.setdefault(grp, []).append(float(np.sqrt(dist2[0])))
    tie_rows = {g: dict(n=len(v), med=float(np.median(v))) for g, v in sorted(hits.items())}

    # Two flat views of the cloud, as scatter plots: open3d's offscreen renderer is not
    # available everywhere.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sub = P[:: max(1, len(P) // 400_000)]
        col = np.asarray(pc_in.colors)[:: max(1, len(P) // 400_000)]
        for tag, (ax0, ax1) in {"topdown": (0, 1), "side": (0, 2)}.items():
            fig, ax = plt.subplots(figsize=(14, 9))
            ax.scatter(sub[:, ax0], sub[:, ax1], s=0.3, c=np.clip(col, 0, 1), linewidths=0)
            ax.set_aspect("equal"); ax.set_title(f"da3_world {tag} ({len(P):,} pts)")
            fig.savefig(OUT / f"da3_world_{tag}.png", dpi=110, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:                                            # not worth failing the run over
        print(f"  [warn] scatter previews skipped: {e}")

    rep = ["# DA3 world fusion", "",
           f"voxel {args.voxel} m, sdf_trunc {args.sdf_trunc}, confidence above p{args.conf_pct:.0f}, "
           f"depth under {args.max_depth} m, {len(names)} cameras, depth calibrated on the floor", "",
           f"- points (inside arena box): **{len(P):,}** ; mesh: {len(np.asarray(mesh.vertices)):,} verts / "
           f"{len(np.asarray(mesh.triangles)):,} tris",
           f"- fraction that ended up outside the building, before cropping: **{floater_frac:.2%}** "
           f"(for comparison, the June NeRF cloud managed 44%)",
           f"- floor band |z|<0.5: median z {floor_medz:+.3f} m, std {floor_stdz:.3f} m",
           "- off-floor tie NN distance by group: " +
           ", ".join(f"{g} {r['med']:.2f} m (n={r['n']})" for g, r in tie_rows.items()),
           "",
           "What this is and is not: surfaces. Thin structure such as the hoop rim and the board "
           "is beyond what depth fusion can resolve at this range."]
    (OUT / "fusion_report.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep[2:]))
    print(f"DONE in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
