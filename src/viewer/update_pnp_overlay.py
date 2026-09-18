"""Adds a second set of camera poses to scene.json for the viewer's PnP Result tab.

The poses come from any placed_cameras*.json and are drawn next to the cameras the scene was
built with. Everything else in scene.json is left as it was.

Run from src/viewer/ :
  uv run --project .. python update_pnp_overlay.py --placer ../../placements/placed_cameras_redfox_repaired.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent          # the repository root


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--placer", required=True, help="placed_cameras*.json with the poses to show")
    ap.add_argument("--scene", default=str(HERE / "public" / "scene.json"))
    args = ap.parse_args()

    placer = Path(args.placer).resolve()
    poses = json.loads(placer.read_text(encoding="utf-8"))
    scene_path = Path(args.scene)
    scene = json.loads(scene_path.read_text(encoding="utf-8"))

    n = 0
    for cam in scene["cameras"]:
        cam.pop("pnp_position", None)
        cam.pop("pnp_rotation_c2w", None)
        entry = poses.get(cam["name"])
        if not isinstance(entry, dict) or len(entry.get("position", [])) != 3:
            continue
        # the viewer stores a camera's rotation as yaw, pitch, roll about Z, X, Y
        rot = entry.get("rotation") or entry.get("rotationYXZ") or [0.0, 0.0, 0.0]
        cam["pnp_position"] = [float(x) for x in entry["position"]]
        # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html
        cam["pnp_rotation_c2w"] = Rotation.from_euler("ZXY", np.asarray(rot, float)).as_matrix().tolist()
        n += 1
    mtime = datetime.fromtimestamp(placer.stat().st_mtime).astimezone().isoformat(timespec="seconds")
    # recorded relative to the repository when possible, so scene.json carries no machine-specific path
    shown = placer.relative_to(ROOT).as_posix() if placer.is_relative_to(ROOT) else str(placer)
    scene["meta"]["pnp_overlay_source"] = {"path": shown, "mtime": mtime}

    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    print(f"{n} cameras got a PnP pose; wrote {scene_path}")


if __name__ == "__main__":
    main()
