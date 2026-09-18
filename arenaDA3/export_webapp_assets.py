"""Puts the reconstructed worlds into the viewer's NeRF tab.

Writes into src/viewer/public/nerf/:

  da3_world.ply    the still Depth Anything 3 world, thinned to 600,000 points
  da3_4d/          one world per instant, plus the index.json the timeline reads
  manifest.json    the list the NeRF tab loads first

The NeRF point cloud (nerf_cloud.ply) is written there by NeRF/export_nerf_cloud.py; this only
lists it in the manifest when it is present.

Each file is written next to its final name first and then renamed, so the viewer never loads
half a file.

Run from the repository root (arenaDA3/run_world.py does this for you):
  uv run --project arenaDA3/da3_prior python arenaDA3/export_webapp_assets.py --world work/world
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "src" / "viewer" / "public" / "nerf"
MAX_WORLD_POINTS = 600_000


def publish_still_world(src: Path, web: Path) -> dict:
    pcd = o3d.io.read_point_cloud(str(src))
    n0 = len(pcd.points)
    if n0 == 0:
        raise SystemExit(f"{src} is empty")
    if n0 > MAX_WORLD_POINTS:
        idx = np.random.default_rng(0).choice(n0, MAX_WORLD_POINTS, replace=False)
        pcd = pcd.select_by_index(np.sort(idx))
    tmp = web / "da3_world.tmp.ply"
    o3d.io.write_point_cloud(str(tmp), pcd, write_ascii=False)
    tmp.replace(web / "da3_world.ply")
    print(f"  da3_world.ply: {n0:,} -> {len(pcd.points):,} points")
    return {"file": "da3_world.ply", "points": len(pcd.points),
            "source": "Depth Anything 3, fused from all cameras", "point_size": 0.05}


def publish_4d(src: Path, web: Path) -> None:
    index = json.loads((src / "index.json").read_text(encoding="utf-8"))
    missing = [f["file"] for f in index["frames"] if not (src.parent / f["file"]).exists()]
    if missing:
        raise SystemExit(f"4D world incomplete, missing: {missing[:5]}")
    tmp = web / "da3_4d.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    dst = web / "da3_4d"
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    print(f"  da3_4d/: {len(index['frames'])} instants")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", required=True, help="the work/world folder of a run")
    ap.add_argument("--web", default=str(WEB), help="the viewer's nerf folder")
    args = ap.parse_args()
    web, world = Path(args.web), Path(args.world)
    web.mkdir(parents=True, exist_ok=True)

    da3_world = publish_still_world(world / "still" / "da3_world.ply", web)
    publish_4d(world / "da3_4d", web)

    cloud_path = web / "nerf_cloud.ply"
    cloud = None
    if cloud_path.exists():
        n = len(o3d.io.read_point_cloud(str(cloud_path)).points)
        cloud = {"file": "nerf_cloud.ply", "points": n, "source": "trained NeRF", "point_size": 0.03}
    else:
        print("  no nerf_cloud.ply yet (run NeRF/export_nerf_cloud.py); the NeRF layer is left out")

    # leftovers of older exports that the tab no longer shows
    for old in ("images", "stands.json", "hoops.json"):
        p = web / old
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()

    manifest = {
        "title": "The arena in 3D",
        "subtitle": "Depth Anything 3 worlds from the game videos, and the NeRF of the empty arena.",
        "summary": "",
        "metrics": [],
        "images": [],
        "cloud": cloud,
        "da3_world": da3_world,
    }
    tmp = web / "manifest.tmp.json"
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(web / "manifest.json")
    print(f"  manifest.json -> {web}")


if __name__ == "__main__":
    main()
