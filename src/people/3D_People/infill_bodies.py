"""Fills in the frames where a roster slot has a position but no fitted body.

The roster has every slot at every frame, but the body fit only produces a body where there
were enough views, and the quality gate in run_bodies.py removes more. For each missing
frame the position comes from the slot's smoothed roster trajectory (pos_smooth), and the
pose is interpolated between the two nearest fitted frames: each joint rotates along the
shortest arc, with eased timing so the motion starts and stops smoothly. At the start or end
of a track the nearest fitted pose is copied. The body shape (betas) is the slot's own.
Every added record is marked "interp": true so the viewer can show it.

In : bodies_gated.json (fit output after the gate) and work/people/roster.json.
Out: bodies_full.json with one record per slot per timestamp, plus an "infill" summary.

Run (normally via run_bodies.py; standalone, from src/:)
    uv run python people/3D_People/infill_bodies.py \
        --bodies ../work/bodies/bodies_gated.json \
        --roster ../work/people/roster.json \
        --out    ../work/bodies/bodies_full.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SRC = Path(__file__).resolve().parent.parent.parent
WORK = SRC.parent / "work"


def slerp_pose(aa_a, aa_b, u):
    """Spherical interpolation of stacked axis-angle joints (J*3,) at fraction u."""
    A = np.asarray(aa_a, float).reshape(-1, 3)
    B = np.asarray(aa_b, float).reshape(-1, 3)
    # see: https://en.wikipedia.org/wiki/Slerp
    # and: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    Ra, Rb = Rotation.from_rotvec(A), Rotation.from_rotvec(B)
    rel = (Ra.inv() * Rb).as_rotvec()
    return (Ra * Rotation.from_rotvec(rel * u)).as_rotvec().reshape(-1).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bodies", default=str(WORK / "bodies" / "bodies_gated.json"))
    ap.add_argument("--roster", default=str(WORK / "people" / "roster.json"))
    ap.add_argument("--out", default=str(WORK / "bodies" / "bodies_full.json"))
    args = ap.parse_args()

    doc = json.loads(Path(args.bodies).read_text(encoding="utf-8"))
    roster = json.loads(Path(args.roster).read_text(encoding="utf-8"))
    ts = sorted(int(t) for t in doc["frames"])
    slots = sorted(int(s) for s in doc["tracks"])

    # per-slot smoothed court trajectory at every 25 fps frame
    traj = {s: {} for s in slots}
    for t_s, rows in roster["frames"].items():
        for r in rows:
            if r["track_id"] in traj:
                p = r.get("pos_smooth", r["pos"])
                traj[r["track_id"]][int(t_s)] = np.array(p[:2], float)

    recs = {(int(p["track_id"]), int(t)): p
            for t in ts for p in doc["frames"][str(t)]}
    n_have = len(recs)
    added = {s: 0 for s in slots}
    for s in list(slots):
        fitted = sorted(t for (tid, t) in recs if tid == s)
        if not fitted:
            # nothing to copy a body shape or a pose from: leave this athlete out of the 3D view
            print(f"[infill] slot {s} has no fitted frames at all; it gets no body")
            slots.remove(s)
            continue
        betas = doc["tracks"][str(s)]["betas"]
        for t in ts:
            if (s, t) in recs:
                continue
            t_a = max((x for x in fitted if x < t), default=None)
            t_b = min((x for x in fitted if x > t), default=None)
            ra = recs.get((s, t_a)) if t_a is not None else None
            rb = recs.get((s, t_b)) if t_b is not None else None
            base = traj[s].get(t)
            if base is None:                      # roster guarantees this never happens
                raise SystemExit(f"slot {s} t={t}: no roster trajectory")

            def _offset(r, t_r):
                q = traj[s].get(t_r)
                o = np.array(r["transl"][:2], float) - (q if q is not None
                                                        else np.array(r["transl"][:2]))
                return o, float(r["transl"][2])

            if ra and rb:
                u = (t - t_a) / (t_b - t_a)
                ue = u * u * (3 - 2 * u)          # ease pose timing
                oa, za = _offset(ra, t_a)
                ob, zb = _offset(rb, t_b)
                off = (1 - u) * oa + u * ob
                z = (1 - u) * za + u * zb
                go = slerp_pose(ra["global_orient"], rb["global_orient"], ue)
                bp = slerp_pose(ra["body_pose"], rb["body_pose"], ue)
            else:
                r0, t0 = (ra, t_a) if ra else (rb, t_b)
                off, z = _offset(r0, t0)
                go = list(map(float, r0["global_orient"]))
                bp = list(map(float, r0["body_pose"]))
            rec = {"track_id": s, "t": t, "betas": betas,
                   "global_orient": go, "body_pose": bp,
                   "transl": [float(base[0] + off[0]), float(base[1] + off[1]), z],
                   "interp": True}
            recs[(s, t)] = rec
            added[s] += 1

    doc["tracks"] = {k: v for k, v in doc["tracks"].items() if int(k) in slots}
    frames = {str(t): [recs[(s, t)] for s in slots] for t in ts}
    for t in ts:
        assert len(frames[str(t)]) == len(slots)
    doc["frames"] = frames
    doc["infill"] = {"added_per_slot": {str(s): added[s] for s in slots},
                     "n_fitted": n_have, "n_total": len(slots) * len(ts)}
    Path(args.out).write_text(json.dumps(doc), encoding="utf-8")
    tot = sum(added.values())
    print(f"infill: {n_have} fitted + {tot} synthesised = {len(slots) * len(ts)} "
          f"records ({len(slots)} slots x {len(ts)} timestamps)")
    for s in slots:
        print(f"  slot {s}: +{added[s]}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
