"""The athletes in 3D: from the roster of run_people.py to SMPL-X bodies in the viewer.

Steps by step procedure (all making a mess into work/bodies/):

  subset       the ten roster athletes, every 5th frame (5 per second at 25 fps)
  triangulate  a 3D skeleton per athlete and instant, from the 2D keypoints of all cameras
  masks        a person silhouette per camera view (Mask R-CNN), to keep the body inside its outline
  fit          fit a SMPL-X body to each athlete: one body shape per athlete, a pose per instant
  export       drop the instants the body cannot explain, fill the gaps, colour the bodies from
               the video frames and write them into the viewer

The fit is the long part (it took me three hours or more for a 30 second window on an RTX 3080).
A finished step leaves a marker in work/bodies/.done/, so an interrupted run continues where it stopped.
The file pipeline.py (at the repository root) runs this for you.

From the repository root:
  uv run --project src python src/run_bodies.py --work work
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
TDP = SRC / "people" / "3D_People"
for _p in (str(SRC), str(TDP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PLACER = ROOT / "placements" / "placed_cameras_redfox_repaired.json"
PNP = ROOT / "diagnostics" / "court_pnp_cameras_redfox_repaired.json"
VIEWER_PUBLIC = SRC / "viewer" / "public"
SMPLX_MODEL = TDP / "models" / "smplx" / "SMPLX_NEUTRAL.npz"

STRIDE = 5            # fit every 5th frame
GATE_M = 0.5          # a fitted frame whose joints are this far from the triangulated ones is dropped
GATE_REL = 4.0        # ... or this many times the athlete's own usual error (floor 0.20 m)
GATE_REL_FLOOR_M = 0.20
W_BETA = 0.005        # how strongly body shapes are pulled towards the average body


def log(msg: str) -> None:
    print(msg, flush=True)


class Paths:
    def __init__(self, work: Path):
        self.work = work
        self.frames = work / "frames"
        self.roster = work / "people" / "roster.json"
        self.out = work / "bodies"
        self.done = self.out / ".done"
        self.subset = self.out / "tracked_subset.json"
        self.skeletons = self.out / "skeletons.json"
        self.masks = self.out / "masks_dense.npz"
        self.bodies = self.out / "bodies.json"
        self.gated = self.out / "bodies_gated.json"
        self.full = self.out / "bodies_full.json"
        self.export = self.out / "viewer_export"

    def is_done(self, step):
        return (self.done / step).exists()

    def mark(self, step):
        self.done.mkdir(parents=True, exist_ok=True)
        (self.done / step).write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")


def repoint(p: Paths) -> None:
    """Point the 3D_People modules at this run's calibration and files.

    load_cameras binds its default paths when it is defined, so changing adapter.PLACER alone
    is not enough: the function itself is replaced, before any stage module imports it.
    """
    import adapter
    adapter.PLACER = PLACER
    adapter.PNP = PNP
    adapter.PEOPLE_DATA = p.roster.parent
    adapter.DEFAULT_TRACKED = p.roster
    original = adapter.load_cameras
    adapter.load_cameras = lambda placer_path=PLACER, pnp_path=PNP: original(placer_path, pnp_path)
    assert "triangulate" not in sys.modules and "fit_smplx" not in sys.modules


def make_subset(p: Paths) -> None:
    """The roster athletes at every STRIDE-th frame, keeping their team and shirt number."""
    src = json.loads(p.roster.read_text(encoding="utf-8"))
    ts = [int(t) for t in src["timestamps"]][::STRIDE]
    frames = {str(t): src["frames"][str(t)] for t in ts if src["frames"].get(str(t))}
    doc = {"timestamps": ts, "frames": frames, "source": p.roster.name}
    if isinstance(src.get("slots"), dict) and src["slots"]:
        doc["slots"] = src["slots"]
    p.subset.write_text(json.dumps(doc), encoding="utf-8")
    n = sum(len(v) for v in frames.values())
    log(f"  {len(ts)} instants, {n} athlete-instants -> {p.subset.name}")


def triangulate(p: Paths) -> None:
    import triangulate as tri
    sys.argv = ["triangulate.py", "--tracked", str(p.subset), "--out", str(p.skeletons)]
    tri.main()


def masks(p: Paths) -> None:
    import masks as m
    m.RAW_DIR = p.frames
    m.OUT_DIR = p.out
    sys.argv = ["masks.py", "--tracked", str(p.subset), "--out_dir", str(p.out)]
    m.main()


def fit(p: Paths, fps: float) -> None:
    import fit_smplx
    # the silhouettes help the fit; without them it still works, on the skeleton alone
    sil = ["--masks", str(p.masks)] if p.masks.exists() else ["--no_silhouette"]
    sys.argv = ["fit_smplx.py", "--tracked", str(p.subset), "--out", str(p.bodies),
                *sil, "--w_beta", str(W_BETA), "--fps", str(fps)]
    fit_smplx.main()


def gate_bodies(p: Paths) -> None:
    """Drop the fitted instants that the athlete's own body cannot explain.

    One body shape is fitted per athlete. If a track picked up a second person somewhere, those
    instants end up with joints far from the triangulated skeleton, and showing them would draw
    a body that belongs to nobody. Two tests: a fixed distance, and a multiple of the athlete's
    own usual error, which catches single-instant pose glitches.
    """
    B = json.loads(p.bodies.read_text(encoding="utf-8"))
    S = json.loads(p.skeletons.read_text(encoding="utf-8"))
    tri = {(q["track_id"], int(t)): {j["j"]: np.asarray(j["xyz"], float)
                                     for j in q["joints"] if j and j.get("xyz")}
           for t in S["timestamps"] for q in S["frames"][str(t)]}

    def err(q, t):
        ref = tri.get((q["track_id"], int(t)))
        if not ref:
            return None
        F = np.asarray(q["joints_coco"], float)
        d = [float(np.linalg.norm(F[j] - v)) for j, v in ref.items() if j < len(F)]
        return float(np.median(d)) if d else None

    per = {}
    for t in B["timestamps"]:
        for q in B["frames"][str(t)]:
            e = err(q, t)
            if e is not None:
                per.setdefault(q["track_id"], []).append(e)
    med = {k: float(np.median(v)) for k, v in per.items()}

    kept, dropped = {}, 0
    for t in B["timestamps"]:
        keep = []
        for q in B["frames"][str(t)]:
            e = err(q, t)
            if e is not None and (e > GATE_M or e > max(GATE_REL * med.get(q["track_id"], 0.0), GATE_REL_FLOOR_M)):
                dropped += 1
                continue
            keep.append(q)
        if keep:
            kept[str(t)] = keep
    B["frames"] = kept
    B["timestamps"] = sorted(int(t) for t in kept)
    present = {q["track_id"] for fr in kept.values() for q in fr}
    B["tracks"] = {k: v for k, v in B.get("tracks", {}).items() if int(k) in present}
    p.gated.write_text(json.dumps(B), encoding="utf-8")
    log(f"  dropped {dropped} fitted instants the body could not explain")


def export(p: Paths) -> None:
    import infill_bodies
    import export_bodies

    gate_bodies(p)
    # fill the instants the fit or the gate left empty, following each athlete's court trajectory
    sys.argv = ["infill_bodies.py", "--bodies", str(p.gated), "--roster", str(p.roster), "--out", str(p.full)]
    infill_bodies.main()

    # write into a staging folder first, then swap into the viewer, so the viewer never sees half an export
    if p.export.exists():
        shutil.rmtree(p.export)
    export_bodies.RAW_DIR = p.frames
    export_bodies.DEFAULT_MASKS = p.masks
    sys.argv = ["export_bodies.py", "--bodies", str(p.full), "--tracked", str(p.subset),
                "--out_dir", str(p.export)]
    export_bodies.main()

    target_dir = VIEWER_PUBLIC / "bodies"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.move(str(p.export / "bodies"), str(target_dir))
    shutil.copy2(p.export / "bodies.json", VIEWER_PUBLIC / "bodies.json")
    log(f"  published -> {VIEWER_PUBLIC / 'bodies.json'} and {target_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=str(ROOT / "work"))
    args = ap.parse_args()
    p = Paths(Path(args.work).resolve())
    if not p.roster.exists():
        raise SystemExit(f"{p.roster} is missing: run the people step first.")
    if not SMPLX_MODEL.exists():
        raise SystemExit(f"The SMPL-X body model is missing: {SMPLX_MODEL}")
    fps = json.loads((p.work / "window.json").read_text(encoding="utf-8"))["fps"]
    p.out.mkdir(parents=True, exist_ok=True)
    repoint(p)
    t0 = time.time()

    steps = [("subset", lambda: make_subset(p)), ("triangulate", lambda: triangulate(p)),
             ("masks", lambda: masks(p)), ("fit", lambda: fit(p, fps)), ("export", lambda: export(p))]
    for name, fn in steps:
        log(f"\n== {name} ==")
        if p.is_done(name):
            log("  already done")
            continue
        fn()
        p.mark(name)
    log(f"\n[run_bodies] done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
