"""The arena in 3D from the game videos, for the viewer's NeRF tab.

Steps, all writing into work/world/:

  plates    one clean picture per camera: the median of 30 frames spread over the window,
            which removes the moving players and leaves the empty hall
  priors    Depth Anything 3 guesses the depth of those ten pictures together, knowing exactly
            where each camera is
  still     the ten depth maps are fused into one point cloud of the building
  4d        the same, repeated on the real frames every 5th frame, so the world can be played
            back with the athletes in it (the long part: about 40 minutes for 30 seconds)
  publish   copy the results into the viewer

It reads the frames that run_people.py extracted into work/frames/. A finished step leaves a
marker in work/world/.done/, so an interrupted run continues where it stopped. pipeline.py (at
the repository root) runs this for you.

Run from the repository root, in the Depth Anything 3 environment:
  uv run --project arenaDA3/da3_prior python arenaDA3/run_world.py --work work
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ARENA = Path(__file__).resolve().parent
ROOT = ARENA.parent
PLACER = ROOT / "placements" / "placed_cameras_redfox_repaired.json"
PNP = ROOT / "diagnostics" / "court_pnp_cameras_redfox_repaired.json"
CAMS = ["cam01", "cam02", "cam03", "cam04", "cam05", "cam06", "cam07", "cam08", "cam12", "cam13"]
PLATE_FRAMES = 30


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list) -> None:
    cmd = [str(c) for c in cmd]
    log("  $ " + " ".join(cmd[1:]))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {' '.join(cmd)}")


def make_plates(frames: Path, n_frames: int, out: Path) -> None:
    """The median of PLATE_FRAMES frames per camera, spread evenly over the whole window.

    A player who moves is in a different place in most of those frames, so at any pixel the
    median is the floor or the wall behind them. Spreading the frames over the whole window
    matters: anyone standing still for a couple of seconds would otherwise stay in the picture.
    """
    out.mkdir(parents=True, exist_ok=True)
    picks = sorted(set(np.linspace(0, n_frames - 1, PLATE_FRAMES).round().astype(int).tolist()))
    for cam in CAMS:
        stack = []
        for t in picks:
            img = cv2.imread(str(frames / cam / f"t{t:06d}.jpg"))
            if img is None:
                raise SystemExit(f"missing frame {frames / cam / f't{t:06d}.jpg'}")
            stack.append(img)
        # the median over time removes whatever moves
        # see: https://numpy.org/doc/stable/reference/generated/numpy.median.html
        plate = np.median(np.stack(stack), axis=0).astype(np.uint8)
        cv2.imwrite(str(out / f"{cam}_{PLATE_FRAMES}.png"), plate)
        log(f"  {cam}: median of {len(picks)} frames")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=str(ROOT / "work"))
    args = ap.parse_args()
    work = Path(args.work).resolve()
    window_file = work / "window.json"
    if not window_file.exists() or not (work / "people" / ".done" / "frames").exists():
        raise SystemExit("The video frames are not there yet: run the whole pipeline first "
                         "(uv run pipeline.py).")
    window = json.loads(window_file.read_text(encoding="utf-8"))
    frames, n_frames, fps = work / "frames", int(window["n_frames"]), float(window["fps"])
    W = work / "world"
    done = W / ".done"
    py = sys.executable
    t0 = time.time()

    steps = [
        ("plates", lambda: make_plates(frames, n_frames, W / "plates")),
        ("priors", lambda: run([py, ARENA / "da3_prior" / "export_depth_priors_da3.py",
                                "--pnp", PNP, "--poses", PLACER, "--plates", W / "plates",
                                "--plate_n", PLATE_FRAMES, "--out", W / "depth_priors"])),
        ("still", lambda: run([py, ARENA / "da3_fusion" / "fuse_da3_tsdf.py",
                               "--priors", W / "depth_priors", "--out", W / "still"])),
        ("4d", lambda: run([py, ARENA / "da3_4d" / "fuse_4d.py", "--frames", frames,
                            "--n_frames", n_frames, "--fps", fps, "--pnp", PNP, "--placer", PLACER,
                            "--web_out", W / "da3_4d", "--out", W / "4d"])),
        ("publish", lambda: run([py, ARENA / "export_webapp_assets.py", "--world", W])),
    ]
    for name, fn in steps:
        log(f"\n== {name} ==")
        # publishing is quick and must show the latest results, so it always runs
        if name != "publish" and (done / name).exists():
            log("  already done")
            continue
        fn()
        done.mkdir(parents=True, exist_ok=True)
        (done / name).write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log(f"\n[run_world] done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
