"""Turns the trained NeRF into the point cloud shown in the viewer's NeRF tab.

How it works:
- The network is asked how solid the space is at every point of a 6 cm grid. Points that are
  solid enough are kept, coloured as seen from the nearest camera.
- Only the layer just above the floor is kept from the grid. Higher up, a NeRF of this arena
  is mostly floating noise, because the cameras see the stands only from far away.
- The floor itself comes from model/floor_points.ply, the floor of an earlier NeRF of the
  same arena, which is sharper.
- The result is thinned out so a browser can load it, and written to
  src/viewer/public/nerf/nerf_cloud.ply.

The NeRF was trained on the empty arena before the game was recorded, so this cloud does not
change with the videos. The pipeline only creates it when it is missing.

Run from the repository root:
  uv run --project src python NeRF/export_nerf_cloud.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from nerf_model import load_model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_OUT = ROOT / "src" / "viewer" / "public" / "nerf" / "nerf_cloud.ply"

GRID_M = 0.06          # grid step
MIN_ALPHA = 0.25       # how solid a grid point must be to count (the value the shipped cloud was made with)
FLOOR_TOP_M = 0.45     # below this height the floor points are used
KEEP_BELOW_M = 0.6     # nothing above this height is kept
THIN_VOXEL_M = 0.04    # thinning grid for the final cloud
MAX_POINTS = 500_000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, a = load_model(HERE / "model" / "nerf.pt", device)
    poses = json.loads((HERE / "model" / "camera_poses.json").read_text(encoding="utf-8"))
    cam_centres = torch.tensor([p["position"] for p in poses.values()], dtype=torch.float32, device=device)

    # the box the NeRF was trained in
    bb = [float(v) for v in a["bbox"].split(",")]
    xs = np.arange(bb[0], bb[3], GRID_M, dtype=np.float32)
    ys = np.arange(bb[1], bb[4], GRID_M, dtype=np.float32)
    zs = np.arange(max(bb[2], -0.15), min(bb[5], 10.5), GRID_M, dtype=np.float32)
    zs = zs[(zs >= FLOOR_TOP_M - 1e-4) & (zs < KEEP_BELOW_M)]
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    slab = np.stack([X.ravel(), Y.ravel()], -1)

    grid_pts, grid_cols = [], []
    for z in zs:
        P = np.concatenate([slab, np.full((len(slab), 1), z, np.float32)], 1)
        Pt = torch.from_numpy(P).to(device)
        # look at each point from its nearest camera
        D = cam_centres[torch.cdist(Pt, cam_centres).argmin(1)] - Pt
        D = D / D.norm(dim=-1, keepdim=True)
        sig, rgb = [], []
        with torch.no_grad(), torch.autocast("cuda", enabled=device == "cuda"):
            for s in range(0, len(Pt), 400_000):
                sg, rg = net(Pt[s:s + 400_000], D[s:s + 400_000])
                sig.append(sg.float())
                rgb.append(rg.float())
        # density to opacity, equation 3 of the NeRF paper: https://arxiv.org/abs/2003.08934
        alpha = 1.0 - torch.exp(-torch.cat(sig).clamp_min(0) * GRID_M)
        keep = (alpha > MIN_ALPHA).cpu().numpy()
        grid_pts.append(P[keep])
        grid_cols.append(torch.cat(rgb).cpu().numpy()[keep])
    grid_pts = np.concatenate(grid_pts)
    grid_cols = np.concatenate(grid_cols)

    floor = o3d.io.read_point_cloud(str(HERE / "model" / "floor_points.ply"))
    pts = np.concatenate([np.asarray(floor.points), grid_pts])
    cols = np.concatenate([np.asarray(floor.colors), grid_cols])
    print(f"floor points {len(floor.points):,} + NeRF grid points {len(grid_pts):,}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0, 1).astype(np.float64))
    # see: https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html
    pcd = pcd.voxel_down_sample(voxel_size=THIN_VOXEL_M)
    if len(pcd.points) > MAX_POINTS:
        idx = np.random.default_rng(0).choice(len(pcd.points), MAX_POINTS, replace=False)
        pcd = pcd.select_by_index(np.sort(idx))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # written aside and then renamed, so an interrupted export never leaves half a file behind
    tmp = out.with_name(out.stem + ".tmp.ply")
    o3d.io.write_point_cloud(str(tmp), pcd, write_ascii=False)
    tmp.replace(out)
    print(f"wrote {len(pcd.points):,} points -> {out}")


if __name__ == "__main__":
    main()
