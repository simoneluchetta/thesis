"""The athletes in 2D: from the synchronised videos to the viewer's People map.

Step by step procedure (each one reading what the previous one wrote into work/people/):

  frames    decode the chosen time window of every video into JPEG frames (work/frames/)
  detect    find people and their feet in every frame, and describe how each person looks
  fuse      combine the ten cameras into positions on the court, frame by frame
  assemble  link those positions through time into tracks
  roles     tell players from referees and bystanders
  jersey    read the shirt numbers (EasyOCR)
  repair    split or join tracks where the shirt numbers say the tracking went wrong
  roster    lock the result to exactly five athletes per team
  viewer    write the People map data into the viewer

A finished step leaves a marker in work/people/.done/, so an interrupted run continues where it stopped.
The file pipeline.py (at the repository root) runs this for you.
It is preferable not to execute this file directly (It's all up to you though...).

Run from the repository root:
  uv run --project src python src/run_people.py --work work
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The camera rig. Videos are named after the recorder number; the calibration after the camera.
CAMS = {1: "cam01", 2: "cam02", 3: "cam03", 4: "cam04", 5: "cam05",
        6: "cam06", 7: "cam07", 8: "cam08", 12: "cam12", 13: "cam13"}
PLACER = ROOT / "placements" / "placed_cameras_redfox_repaired.json"
PNP = ROOT / "diagnostics" / "court_pnp_cameras_redfox_repaired.json"
VIEWER_PUBLIC = SRC / "viewer" / "public"
CHUNK = 150          # frames per detection process; a fresh process per chunk keeps memory in check


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str]) -> None:
    """Run one pipeline module in the same environment and stop everything if it fails."""
    log("  $ " + " ".join(str(c) for c in cmd[1:]))
    r = subprocess.run([str(c) for c in cmd], cwd=str(SRC))
    if r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {' '.join(str(c) for c in cmd)}")


class Work:
    """Where every intermediate file of a run lives."""

    def __init__(self, work: Path):
        self.root = work
        self.frames = work / "frames"
        self.people = work / "people"
        self.done = self.people / ".done"
        self.window = work / "window.json"

    def is_done(self, step: str) -> bool:
        return (self.done / step).exists()

    def mark(self, step: str) -> None:
        self.done.mkdir(parents=True, exist_ok=True)
        (self.done / step).write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# frames


def probe_videos(videos: Path) -> dict:
    """Open every video once and check that they belong together."""
    info = {}
    missing = [f"out{n}.mp4" for n in CAMS if not (videos / f"out{n}.mp4").exists()]
    if missing:
        raise SystemExit(f"Missing videos in {videos}: {', '.join(missing)}. "
                         f"Expected out1..out8, out12 and out13 (.mp4).")
    for n, cam in CAMS.items():
        # see: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
        cap = cv2.VideoCapture(str(videos / f"out{n}.mp4"))
        if not cap.isOpened():
            raise SystemExit(f"Cannot open {videos / f'out{n}.mp4'}")
        # containers store the rate as a fraction, so 25 fps can come back as 25.000195
        info[cam] = {"fps": round(float(cap.get(cv2.CAP_PROP_FPS)), 3),
                     "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                     "size": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]}
        cap.release()
    fps = {round(v["fps"], 2) for v in info.values()}
    if len(fps) != 1:
        raise SystemExit(f"The videos have different frame rates: {info}. They must be synchronised recordings.")
    return info


def decode_camera(video: Path, out_dir: Path, first: int, n: int) -> int:
    """Decode n consecutive frames starting at `first`, writing t000000.jpg, t000001.jpg, ..."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if all((out_dir / f"t{i:06d}.jpg").exists() for i in range(n)):
        return n
    cap = cv2.VideoCapture(str(video))
    try:
        if first > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        got = 0
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            dst = out_dir / f"t{i:06d}.jpg"
            if not dst.exists():
                tmp = out_dir / f"t{i:06d}.tmp.jpg"
                cv2.imwrite(str(tmp), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                tmp.replace(dst)
            got += 1
        return got
    finally:
        cap.release()


def step_frames(w: Work, videos: Path, start_s: float, seconds: float) -> dict:
    info = probe_videos(videos)
    fps = next(iter(info.values()))["fps"]
    if abs(fps - 25.0) > 0.5:
        log(f"  WARNING: the videos run at {fps:.2f} fps. The pipeline was tuned on 25 fps footage.")
    shortest = min(v["frames"] for v in info.values())
    first = int(round(start_s * fps))
    # the very last frame a container reports often does not decode, so stay one short of it
    available = shortest - 1 - first
    if available <= 0:
        raise SystemExit(f"--start {start_s}s is past the end of the shortest video ({shortest / fps:.1f} s).")
    n = int(round(seconds * fps)) if seconds > 0 else available
    if n > available:
        log(f"  the videos are shorter than asked: using {available / fps:.1f} s instead of {seconds:.1f} s")
        n = available

    log(f"  window: {start_s:.1f} s + {n / fps:.1f} s = frames {first}..{first + n - 1} at {fps:.2f} fps, "
        f"{len(CAMS)} cameras -> {w.frames}")
    # see: https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor
    with ThreadPoolExecutor(max_workers=5) as pool:
        got = dict(zip(CAMS.values(), pool.map(
            lambda item: decode_camera(videos / f"out{item[0]}.mp4", w.frames / item[1], first, n),
            CAMS.items())))
    n_ok = min(got.values())
    if n_ok < n:
        log(f"  some videos stopped early ({got}); using the first {n_ok} frames of every camera")
        n = n_ok
    if n < CHUNK:
        raise SystemExit(f"Only {n} frames could be decoded; at least {CHUNK} are needed.")
    window = {"fps": fps, "first_frame": first, "n_frames": n, "cameras": list(CAMS.values()),
              "videos": {cam: info[cam] for cam in CAMS.values()}}
    w.window.write_text(json.dumps(window, indent=1), encoding="utf-8")
    return window


# ---------------------------------------------------------------------------------------------
# detection


def detect_chunk(w: Work, t0: int, t1: int, out: Path) -> None:
    """Run people/detect.py on frames t0..t1-1, which are already on disk."""
    import people.detect as det

    det.PLACER = PLACER
    det.PNP = PNP
    det.RAW_DIR = w.frames
    det.OUT_DIR = w.people
    det.CROPS_DIR = w.people / "crops"
    # detect.py would decode the videos itself; the frames already exist, so just point at them
    det.extract_frames = lambda videos_dir, ts, raw_dir, stereo_cams: {
        t: {n: w.frames / cam / f"t{t:06d}.jpg" for n, cam in CAMS.items()} for t in ts}
    sys.argv = ["detect.py", "--timestamps", ",".join(str(t) for t in range(t0, t1)),
                "--cams", "perimeter", "--stereo_cams", "", "--no_crops",
                "--backend", "osnet", "--out", str(out)]
    det.main()


def chunk_files(w: Work, n_frames: int) -> list[tuple[int, int, Path]]:
    return [(a, min(a + CHUNK, n_frames), w.people / f"detections_c{k:02d}.json")
            for k, a in enumerate(range(0, n_frames, CHUNK))]


# ---------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=str(ROOT / "work"), help="folder for intermediate files")
    ap.add_argument("--videos", default=str(ROOT / "input_videos"))
    ap.add_argument("--start", type=float, default=0.0, help="where the window starts, in seconds")
    ap.add_argument("--seconds", type=float, default=30.0, help="how long the window is (0 = to the end)")
    ap.add_argument("--detect_chunk", nargs=2, type=int, metavar=("T0", "T1"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    w = Work(Path(args.work).resolve())
    w.people.mkdir(parents=True, exist_ok=True)

    if args.detect_chunk:
        t0, t1 = args.detect_chunk
        detect_chunk(w, t0, t1, w.people / f"detections_c{t0 // CHUNK:02d}.json")
        return

    py = sys.executable
    t_start = time.time()

    log("\n== frames ==")
    if w.is_done("frames"):
        window = json.loads(w.window.read_text(encoding="utf-8"))
        log(f"  already done ({window['n_frames']} frames per camera)")
    else:
        window = step_frames(w, Path(args.videos).resolve(), args.start, args.seconds)
        w.mark("frames")
    fps, n_frames = window["fps"], window["n_frames"]

    log("\n== detect ==")
    for t0, t1, out in chunk_files(w, n_frames):
        if (w.done / out.stem).exists():
            log(f"  frames {t0}..{t1 - 1}: already done")
            continue
        log(f"  frames {t0}..{t1 - 1}")
        run([py, __file__, "--work", w.root, "--detect_chunk", t0, t1])
        w.mark(out.stem)

    P = w.people
    fused, assembled, repaired = P / "fused.json", P / "assembled.json", P / "repaired.json"
    jersey_a, jersey_r, roster = P / "jersey_assembled.json", P / "jersey_repaired.json", P / "roster.json"
    cal = ["--placer", PLACER, "--pnp", PNP]
    gpu = ["--gpu"] if _cuda() else []
    steps = [
        ("fuse", [py, "-m", "people.fuse_global", "--in", *[c[2] for c in chunk_files(w, n_frames)],
                  "--out", fused, *cal]),
        ("assemble", [py, "-m", "people.assemble", "--in", fused, "--out", assembled,
                      "--fps", fps, "--cal_pos_q", "0.95"]),
        ("roles", [py, "-m", "people.roles", "--in", assembled, "--appearance", fused, "--fps", fps]),
        ("jersey", [py, "-m", "people.jersey_mv", "--tracked", assembled, "--raw4k", w.frames,
                    "--out", jersey_a, "--read_budget", "120", *cal, *gpu]),
        ("repair", [py, "-m", "people.jersey_repair", "--tracked", assembled, "--jersey", jersey_a,
                    "--fused", fused, "--out", repaired, "--fps", fps]),
        ("roles_repaired", [py, "-m", "people.roles", "--in", repaired, "--appearance", fused, "--fps", fps]),
        ("jersey_repaired", [py, "-m", "people.jersey_mv", "--tracked", repaired, "--raw4k", w.frames,
                             "--out", jersey_r, "--read_budget", "120", *cal, *gpu]),
        # --gt= : no hand-made answers. Those belong to one old run and would be wrong here.
        ("roster", [py, "people/roster_lock.py", "--tracked", repaired, "--jersey", jersey_r, "--gt=",
                    "--out", roster, "--report", P / "roster_report.md", "--fig", P / "roster_topdown.png",
                    "--fps", fps]),
    ]
    for name, cmd in steps:
        log(f"\n== {name} ==")
        if w.is_done(name):
            log("  already done")
            continue
        run(cmd)
        w.mark(name)

    log("\n== viewer (People map) ==")
    thumbs = VIEWER_PUBLIC / "people_thumbs" / "game"
    if thumbs.exists():
        shutil.rmtree(thumbs)
    run([py, "viewer/update_people_block.py", "--people", roster, "--raw4k", w.frames,
         "--thumb_subdir", "game", "--session", "game", "--no_backup",
         "--label", f"game, {n_frames / fps:.0f} s, roster-locked 10 athletes",
         "--calibration", PLACER.name, "--pos_field", "pos_smooth", "--fps", fps])
    log(f"\n[run_people] done in {(time.time() - t_start) / 60:.0f} min")


def _cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    main()
