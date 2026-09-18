# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Runs the whole project: the synchronised videos go in, the People map and the NeRF tab come out.

    uv run pipeline.py                        the first 30 seconds of the videos
    uv run pipeline.py --start 120 --seconds 30
    uv run pipeline.py --only people          just one part: people, bodies, nerf or world
    uv run pipeline.py --fresh                throw away earlier work and start again

Buckle the seatbelts... This is what's going to happen:

  people   frames, detection, tracking, shirt numbers, (Capping to 10-athletes... I 10 giocatori nel roster)    ----> People map (2D)
  bodies   SMPL-X bodies of the ten athletes                                                                    ----> People map (3D)
  nerf     the point cloud of the trained NeRF (only if it is missing)                                          ----> NeRF tab
  world    Depth Anything 3 worlds: one still, one per instant                                                  ----> NeRF tab

Everything in between is written to the work/ folder.
If the run stops (or you stop it), run the same command again: it continues where it left off.
The time window of the earlier run is remembered, so --start and --seconds do not need to be repeated.
If the videos or the time window change, a full run starts again from scratch (results from different videos never mix).
--only never deletes earlier work.

Two Python environments are used, one in src/ and one in arenaDA3/da3_prior/
(Depth Anything 3 needs older library versions than the rest).
UV will create both of them on the first run.

More about uv, and about the `# /// script` header at the top of this file:
https://docs.astral.sh/uv/guides/scripts/ and https://peps.python.org/pep-0723/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIDEOS = ROOT / "input_videos"
WORK = ROOT / "work"
STAMP = WORK / "run.json"     # remembers which videos and time window work/ belongs to
MAIN_ENV = ROOT / "src"
DA3_ENV = ROOT / "arenaDA3" / "da3_prior"
PUBLIC = MAIN_ENV / "viewer" / "public"
RECORDERS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]     # the ten cameras around the court
MODELS = [MAIN_ENV / "people" / "3D_People" / "models" / "smplx" / "SMPLX_NEUTRAL.npz", MAIN_ENV / "people" / "models" / "osnet_ain_x1_0.pth"]
STAGES = ["people", "bodies", "nerf", "world"]
GB_PER_SECOND = 0.6     # disk space per second of video (a 30 s run used about 17 GB)


def say(msg: str = "") -> None:
    """Print a line right away."""
    print(msg, flush=True)


def uv() -> str:
    """Find the uv program."""
    exe = os.environ.get("UV") or shutil.which("uv")
    if not exe:
        raise SystemExit("uv was not found on PATH. Install it from https://docs.astral.sh/uv/")
    return exe


def child_env() -> dict:
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    env.pop("VIRTUAL_ENV", None)     # this script's own environment; the stages have their own
    return env


def run_stage(name: str, env_dir: Path, script: Path, args: list[str]) -> None:
    """Run one stage. Its output is shown on screen and also saved in work/logs/."""
    logs = WORK / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    cmd = [uv(), "run", "--project", str(env_dir), "python", str(script), *args]
    say(f"\n{'=' * 78}\n{name.upper()}   (log: {logs / (name + '.log')})\n{'=' * 78}")
    t0 = time.time()
    with open(logs / f"{name}.log", "a", encoding="utf-8") as log:
        log.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(cmd)}\n")
        # the output is read line by line as it arrives
        # see: https://docs.python.org/3/library/subprocess.html#subprocess.Popen
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=child_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        code = proc.wait()
    if code != 0:
        raise SystemExit(f"\nThe {name} stage failed (exit code {code}). The full output is in {logs / (name + '.log')}. Fix the problem and run the same command again: finished steps are not repeated.")
    say(f"\n[{name}] finished in {(time.time() - t0) / 60:.0f} min")


def check_env(env_dir: Path, code: str) -> str:
    """Run a short test script in an environment (uv installs it first if needed). Returns the last line it printed."""
    r = subprocess.run([uv(), "run", "--project", str(env_dir), "python", "-c", code], cwd=str(ROOT), env=child_env(), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"The Python environment in {env_dir.relative_to(ROOT)} does not work:\n{r.stderr[-3000:]}")
    lines = r.stdout.strip().splitlines()
    return lines[-1] if lines else ""


# ---------------------------------------------------------------------------------------------
# discriminate videos (respective to each run)


def video_id(p: Path) -> str:
    """A short id for a video: its size plus a hash of its first and last 4 MB.

    A copy of the same video gets the same id. A different video gets another one.
    """
    h = hashlib.sha1()
    size = p.stat().st_size
    with open(p, "rb") as f:
        h.update(f.read(4 << 20))
        f.seek(max(0, size - (4 << 20)))
        h.update(f.read(4 << 20))
    return f"{size}-{h.hexdigest()[:16]}"


def fingerprint(start: float, seconds: float) -> dict:
    """What a run is made of: the ten videos and the time window."""
    return {"videos": {f"out{n}.mp4": video_id(VIDEOS / f"out{n}.mp4") for n in RECORDERS}, "start": start, "seconds": seconds}


def read_stamp() -> dict | None:
    """The fingerprint saved in work/, or None if there is none."""
    try:
        return json.loads(STAMP.read_text(encoding="utf-8"))
    except Exception:
        return None


def folder_gb(p: Path) -> float:
    """Size of a folder in GB."""
    total = 0
    for dirpath, _, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total / 1e9


# ---------------------------------------------------------------------------------------------
# the NeRF tab


def ply_points(p: Path) -> int:
    """Number of points in a .ply file, read from its header."""
    with open(p, "rb") as f:
        for line in f:
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
            if line.startswith(b"end_header"):
                break
    return 0


def write_nerf_manifest(keep_world: bool) -> None:
    """Update the small file that tells the NeRF tab which point clouds exist.

    The world stage writes the complete file when it ends. Until then this one lists the NeRF
    cloud, and the Depth Anything 3 world only when keep_world is True and its file is there.
    """
    web = PUBLIC / "nerf"
    web.mkdir(parents=True, exist_ok=True)
    path = web / "manifest.json"
    try:
        man = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        man = {"title": "The arena in 3D", "subtitle": "", "summary": "", "metrics": [], "images": []}
    cloud = web / "nerf_cloud.ply"
    man["cloud"] = {"file": "nerf_cloud.ply", "points": ply_points(cloud), "source": "trained NeRF", "point_size": 0.03} if cloud.exists() else None
    if not (keep_world and (web / "da3_world.ply").exists()):
        man["da3_world"] = None
    tmp = web / "manifest.tmp.json"
    tmp.write_text(json.dumps(man, indent=2), encoding="utf-8")
    tmp.replace(path)


def clear_viewer_results() -> None:
    """Take the results of an earlier run out of the viewer before a new run starts.

    Otherwise the viewer could mix the old game with the new one while the pipeline runs.
    The NeRF cloud does not come from the videos, so it stays.
    """
    scene = PUBLIC / "scene.json"
    if scene.exists():
        doc = json.loads(scene.read_text(encoding="utf-8"))
        if doc.pop("people", None) is not None:
            scene.write_text(json.dumps(doc), encoding="utf-8")
    for p in (PUBLIC / "people_thumbs" / "game", PUBLIC / "bodies", PUBLIC / "bodies.json", PUBLIC / "nerf" / "da3_4d", PUBLIC / "nerf" / "da3_world.ply"):
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    write_nerf_manifest(keep_world=False)


# ---------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=float, default=None, help="start of the time window, in seconds (default 0, or the earlier run's)")
    ap.add_argument("--seconds", type=float, default=None, help="length of the time window in seconds (default 30, or the earlier run's; 0 = to the end of the videos)")
    ap.add_argument("--only", choices=STAGES, help="run just this part")
    ap.add_argument("--fresh", action="store_true", help="delete work/ and start from scratch")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    t0 = time.time()

    if args.only == "nerf":
        # the NeRF cloud only needs the trained model: no videos, no work folder
        run_stage("nerf", MAIN_ENV, ROOT / "NeRF" / "export_nerf_cloud.py", [])
        write_nerf_manifest(keep_world=True)
        say("\nDone. Reload the viewer's NeRF tab.")
        return

    # --- check what is needed before touching anything
    missing = [f"out{n}.mp4" for n in RECORDERS if not (VIDEOS / f"out{n}.mp4").exists()]
    if missing:
        raise SystemExit(f"Put the synchronised videos in {VIDEOS}. Missing: {', '.join(missing)}\n(one file per camera, named after the recorder: out1.mp4 ... out8.mp4, out12.mp4, out13.mp4)")
    lost = [str(m.relative_to(ROOT)) for m in MODELS if not m.exists()]
    if lost:
        raise SystemExit("Model files are missing:\n  " + "\n  ".join(lost))

    stamp = None if args.fresh else read_stamp()
    if args.start is None:
        args.start = float(stamp["start"]) if stamp else 0.0
    if args.seconds is None:
        args.seconds = float(stamp["seconds"]) if stamp else 30.0

    say("Checking the Python environments, the GPU and the videos (the first run installs the environments, which takes a while) ...")
    # a small test script, run in the main environment: do the libraries load, is there a
    # GPU, how long are the videos
    probe_code = f"""import json, torch, cv2, easyocr, smplx, torchreid, open3d
v = {{}}
for n in {RECORDERS!r}:
    c = cv2.VideoCapture(r'{VIDEOS}' + '/out%d.mp4' % n)
    v[n] = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)), float(c.get(cv2.CAP_PROP_FPS))]
    c.release()
print(json.dumps({{'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else '', 'videos': v}}))
"""
    info = json.loads(check_env(MAIN_ENV, probe_code))
    if not info["gpu"]:
        raise SystemExit("No NVIDIA GPU is usable from Python (torch.cuda.is_available() is False).\nThe pipeline needs one: on the CPU it would take days.")
    say(f"GPU: {info['gpu']}")
    # a broken video reports 0 or a negative frame count
    # see: https://stackoverflow.com/questions/50962251/opencv-cap-prop-frame-count-return-negative-larg-number
    unreadable = [f"out{n}.mp4" for n, (frames, fps) in info["videos"].items() if frames <= 0 or fps <= 0]
    if unreadable:
        raise SystemExit(f"These videos cannot be read: {', '.join(unreadable)}")
    fps = min(fps for _, fps in info["videos"].values())
    length_s = (min(frames for frames, _ in info["videos"].values()) - 1) / fps
    say(f"Videos: {length_s:.1f} s long (the shortest), {fps:.2f} fps")
    if args.start >= length_s - 6:
        raise SystemExit(f"--start {args.start:g} s is too late: the videos are only {length_s:.1f} s long.")
    if args.only in (None, "world"):
        # checked now, not hours later when the world stage starts
        check_env(DA3_ENV, "import torch, open3d; from depth_anything_3.api import DepthAnything3; print('ok')")
        say("Depth Anything 3 environment: ok")

    # --- the work folder: go on with it, or start again
    fp = fingerprint(args.start, args.seconds)
    old = read_stamp()
    wipe = WORK.exists() and (args.fresh or old != fp)
    if wipe and args.only and not args.fresh:
        old_window = f"(--start {old['start']:g} --seconds {old['seconds']:g})" if old else ""
        raise SystemExit(f"work/ holds a run of other videos or another time window {old_window}, and --only never deletes earlier work.\nRun without --only to start the new run, or add --fresh to throw the old work away.")

    seconds = args.seconds if args.seconds > 0 else max(length_s - args.start, 0)
    need_gb = 10 + GB_PER_SECOND * seconds
    have_gb = shutil.disk_usage(ROOT).free / 1e9 + (folder_gb(WORK) if WORK.exists() else 0.0)
    if have_gb < need_gb:
        raise SystemExit(f"Not enough disk space: about {need_gb:.0f} GB are needed and only {have_gb:.0f} GB can be used (the video frames are stored as JPEGs while the pipeline runs).")

    if wipe:
        reason = "--fresh was given." if args.fresh else "the videos or the time window differ from the previous run."
        say(f"Starting fresh: {reason}")
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)
    STAMP.write_text(json.dumps(fp, indent=1), encoding="utf-8")

    # --- run the stages
    todo = [args.only] if args.only else STAGES
    work = ["--work", str(WORK)]
    if "people" in todo:
        if not (WORK / "people" / ".done" / "frames").exists():
            clear_viewer_results()
        run_stage("people", MAIN_ENV, MAIN_ENV / "run_people.py", work + ["--videos", str(VIDEOS), "--start", str(args.start), "--seconds", str(args.seconds)])
    if "bodies" in todo:
        run_stage("bodies", MAIN_ENV, MAIN_ENV / "run_bodies.py", work)
    if "nerf" in todo:
        if (PUBLIC / "nerf" / "nerf_cloud.ply").exists():
            say("\nNERF: the NeRF point cloud is already in the viewer, nothing to do.")
        else:
            run_stage("nerf", MAIN_ENV, ROOT / "NeRF" / "export_nerf_cloud.py", [])
            write_nerf_manifest(keep_world=True)
    if "world" in todo:
        run_stage("world", DA3_ENV, ROOT / "arenaDA3" / "run_world.py", work)

    say(f"\nAll done in {(time.time() - t0) / 3600:.1f} h.")
    say("To see the results, start the viewer:")
    say("    cd src/viewer")
    if not (MAIN_ENV / "viewer" / "node_modules").exists():
        say("    npm install          (first time only)")
    say("    npm run dev")
    say("then open http://127.0.0.1:5173 and choose the People map or NeRF tab.")


if __name__ == "__main__":
    main()
