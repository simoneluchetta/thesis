"""Merges the foot detections of all cameras into one position per person, per timestamp.

Each camera has already put every foot it found on the court floor. Where several cameras
put feet in nearly the same place they are looking at the same person, and this merges them.

An ankle sits about 10 cm above the floor but is back-projected as if it were on the floor,
so the point lands too far from the camera, more so at shallow viewing angles. height_correct
pulls each point back towards its camera. The merge is greedy: seed on a detection, absorb
anything within merge_r metres, at most one detection per camera, and keep the person only
if at least min_cams cameras agree. An optional appearance test (--app_split) can keep two
close people apart; it is off by default.

The pipeline runs fuse_global.py instead, which imports the helpers defined here.

Run from src/:
    uv run python people/fuse.py

Reads  work/people/foot_detections.json
Writes work/people/people_fused.json, which looks like this:

    {"timestamps": [t, ...],
     "frames": {t: [person, person, ...]}}

    and each person has these fields:
      id                   number of the person within that frame
      pos                  [X, Y, 0], where they stand on the court, in metres
      n_cams, cams         how many cameras saw them, and which ones
      spread_m             how far apart those cameras put them, in metres
      views                one entry per camera: {cam, bbox, keypoints, foot_uv, det_score}
      appearance           the ReID vector, with appearance_backend and appearance_dim
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent          # src/people/
SRC = ROOT.parent                                # src/
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
PEOPLE = SRC.parent / "work" / "people"

from people.reid import cosine_distance          # distance between appearance vectors

# Height above the floor of each kind of foot estimate, in metres. An ankle joint is about
# 10 cm up; the bottom of a bounding box is close to the ground.
SRC_HEIGHT = {"ankle": 0.10, "ankle1": 0.10, "box": 0.03}
# these cameras give unreliable floor positions, so their detections are skipped
EXCLUDE_CAMS = {"cam09L", "cam11L", "cam11R"}


# see: https://en.wikipedia.org/wiki/Line-plane_intersection (the foot ray meets the floor plane)
def height_correct(world_xy, cam_xy, elev_deg, src):
    """Pull a foot estimate back towards its camera, to undo the ankle-height overshoot.

    The shallower the camera's view, the bigger the correction: at 10 degrees an ankle 10 cm
    up lands more than half a metre too far away.
    """
    h = SRC_HEIGHT.get(src, 0.10)
    elev = np.radians(max(elev_deg, 1.0))
    d = h / np.tan(elev)
    v = np.asarray(world_xy) - np.asarray(cam_xy)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.asarray(world_xy)
    return np.asarray(world_xy) - d * (v / n)


def _appearance(r):
    """This detection's appearance vector, if it has one, and which encoder produced it."""
    a = r.get("appearance")
    if a is None:
        return None, None
    return np.asarray(a, dtype=np.float32), r.get("appearance_backend")


def _pool_template(idxs, obs):
    """One appearance vector per person, averaged over the cameras that saw them.

    Uses the same weights as the position. Only vectors from the same encoder are mixed,
    and the encoder most members share is the one used.
    """
    by_backend = {}
    for k in idxs:
        o = obs[k]
        if o["app"] is not None and o["backend"] is not None:
            by_backend.setdefault(o["backend"], []).append((o["app"], o["w"]))
    if not by_backend:
        return None, None, None
    backend = max(by_backend, key=lambda b: len(by_backend[b]))   # the backend most members agree on
    arr = np.stack([v for v, _ in by_backend[backend]])
    ws = np.asarray([w for _, w in by_backend[backend]], dtype=np.float32)
    tpl = (arr * ws[:, None]).sum(0) / ws.sum()
    n = float(np.linalg.norm(tpl))
    if n > 1e-8:
        tpl = tpl / n
    return tpl, backend, int(arr.shape[1])


def fuse_timestamp(rows, arena_xy, merge_r, min_cams, app_split):
    """Merge the detections of one instant into people.
    Returns list of {id,pos,n_cams,spread_m,cams,views[,appearance,...]}."""
    ax, ay = arena_xy
    obs = []
    for r in rows:
        if r["cam"] in EXCLUDE_CAMS:
            continue
        p = height_correct(r["world"][:2], r["cam_xy"], r["elev_deg"], r["foot_src"])
        if abs(p[0]) > ax or abs(p[1]) > ay:
            continue
        w = np.sin(np.radians(r["elev_deg"]))            # steeper ray -> lower ground error
        w *= 0.4 if r["foot_src"] == "box" else 1.0       # down-weight box-bottom fallback
        vec, backend = _appearance(r)
        obs.append({"xy": p, "cam": r["cam"], "w": max(w, 1e-3), "score": r["det_score"],
                    "app": vec, "backend": backend, "row": r})

    obs.sort(key=lambda o: -(o["w"] * o["score"]))
    used = [False] * len(obs)
    people = []
    for i, seed in enumerate(obs):
        if used[i]:
            continue
        seen_cams = {}
        for j, o in enumerate(obs):
            if used[j]:
                continue
            if np.linalg.norm(o["xy"] - seed["xy"]) > merge_r:
                continue
            # optional appearance test: a detection that looks unlike the seed is somebody else
            if (app_split is not None and seed["app"] is not None and o["app"] is not None
                    and o["backend"] == seed["backend"]
                    and cosine_distance(seed["app"], o["app"]) >= app_split):
                continue
            # a camera sees a person once, so a second detection from it is somebody else
            if o["cam"] in seen_cams:
                k = seen_cams[o["cam"]]
                if o["w"] <= obs[k]["w"]:
                    continue
            seen_cams[o["cam"]] = j
        idxs = list(seen_cams.values())
        if len({obs[k]["cam"] for k in idxs}) < min_cams:
            continue
        for k in idxs:
            used[k] = True
        pts = np.array([obs[k]["xy"] for k in idxs])
        ws = np.array([obs[k]["w"] for k in idxs])
        pos = (pts * ws[:, None]).sum(0) / ws.sum()
        spread = float(np.linalg.norm(pts - pos, axis=1).max())
        views = []
        for k in idxs:
            rr = obs[k]["row"]
            views.append({"cam": rr["cam"], "bbox": rr.get("bbox"),
                          "keypoints": rr.get("keypoints"), "foot_uv": rr.get("foot_uv"),
                          "det_score": rr.get("det_score")})
        person = {
            "pos": [float(pos[0]), float(pos[1]), 0.0],
            "n_cams": len(idxs),
            "spread_m": round(spread, 3),
            "cams": sorted(obs[k]["cam"] for k in idxs),
            "views": views,
        }
        tpl, backend, dim = _pool_template(idxs, obs)
        if tpl is not None:
            person["appearance"] = [round(float(x), 4) for x in tpl]
            person["appearance_backend"] = backend
            person["appearance_dim"] = dim
        people.append(person)
    people.sort(key=lambda p: (-p["n_cams"], p["pos"][0]))
    for n, p in enumerate(people):
        p["id"] = n
    return people


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=str(PEOPLE / "foot_detections.json"))
    ap.add_argument("--out", default=str(PEOPLE / "people_fused.json"))
    ap.add_argument("--arena_x", type=float, default=15.5)
    ap.add_argument("--arena_y", type=float, default=8.5)
    ap.add_argument("--merge_r", type=float, default=0.8, help="cluster radius (m) for one person")
    ap.add_argument("--min_cams", type=int, default=2, help="distinct cameras required to accept a person")
    ap.add_argument("--app_split", type=float, default=None,
                    help="appearance cosine distance above which two within-merge_r detections "
                         "are split into separate people. UNSET = pure geometry (default, "
                         "non-regressive). Reliable only with OSNet (~0.4); cnn is too weak.")
    args = ap.parse_args()

    dets = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    timestamps = sorted(int(t) for t in dets)
    frames = {}
    for t in timestamps:
        people = fuse_timestamp(dets[str(t)], (args.arena_x, args.arena_y),
                                args.merge_r, args.min_cams, args.app_split)
        frames[str(t)] = people
        msg = ", ".join(f"({p['pos'][0]:.1f},{p['pos'][1]:.1f}) {p['n_cams']}cam/{p['spread_m']:.2f}m"
                        for p in people)
        print(f"  t{t:06d}: {len(people)} people -> {msg}")

    out = {"timestamps": timestamps, "frames": frames}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    mode = "geometry-only" if args.app_split is None else f"appearance-split@{args.app_split}"
    print(f"[people_fuse] wrote {args.out}  ({mode})")


if __name__ == "__main__":
    main()
