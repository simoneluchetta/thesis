"""Works out which of the tracked people are actually playing.

Each track gets a role from its motion: `bystander` when it barely moves (median speed below
--static_speed), `peripheral` when it spends most of its time outside the court lines
(--outside_frac), `unknown` when it is shorter than --min_len samples, and `player` otherwise.
The fusion accepts anyone inside the arena, so coaches, substitutes and spectators next to
the lines are real detections that are not athletes. Referees move like players inside the
lines, so they are found from the kit instead: the tracks' appearance vectors are clustered
(k chosen by silhouette), the two largest clusters are the two teams, and every other track
is flagged `off_team`. The flag is kept separate from the role because it only says "on
neither team", not whether the person is a referee, a coach or a substitute.

Reads  : the tracks (--in) and the fused file with the appearance vectors (--appearance)
Writes : the same file with a `role` per person-instant and a `roles` block per track

Run from src/:
  uv run python -m people.roles --in ../work/people/assembled.json --appearance ../work/people/fused.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

# half length and half width of the court (m); the fusion accepts people well outside this
COURT_X, COURT_Y = 14.0, 7.5


def off_team_flags(data, fused_path, min_samples, z_thr, k_max=5, max_flag_frac=0.40):
    """Flag the tracks that wear neither team's kit.

    Each track with at least `min_samples` appearance vectors gets a mean template. The
    templates are clustered with k-means for k = 2..k_max, and k is picked by silhouette.
    The two largest clusters are the two teams; every other track is off-team. If the flagged
    tracks would hold more than `max_flag_frac` of the person-instants, the clustering is not
    separating kits and nothing is flagged. The robust z of each track's distance to its
    nearest cluster centre is returned for the log only; it does not decide anything.

    Returns {track_id: (off_team, z)}, or {} if the appearance is unavailable.
    """
    try:
        import sys
        den = Path(__file__).resolve().parent.parent
        if str(den) not in sys.path:
            sys.path.insert(0, str(den))
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        import people.assemble as A
    except Exception as e:                                     # pragma: no cover
        print(f"[people_roles] appearance test unavailable ({e}); skipping")
        return {}

    fus = json.loads(Path(fused_path).read_text(encoding="utf-8"))
    app = {}
    for t in fus["timestamps"]:
        for j, q in enumerate(fus["frames"][str(t)]):
            a = q.get("appearance")
            if a is not None:
                app[(int(t), j)] = A._unit(np.asarray(a, dtype=np.float32))
    if not app:
        return {}

    tpl = {}
    for t in sorted(int(x) for x in data["timestamps"]):
        for j, q in enumerate(data["frames"][str(t)]):
            if (t, j) in app:
                tpl.setdefault(q.get("track_id", q.get("id")), []).append(app[(t, j)])
    # a template from only a few samples is too noisy, so short tracks are skipped
    tpl = {k: A._unit(np.mean(np.stack(v), axis=0)) for k, v in tpl.items()
           if len(v) >= min_samples}
    if len(tpl) < 8:
        return {}
    keys = sorted(tpl)
    X = np.stack([tpl[k] for k in keys])

    best = None
    for k in range(2, min(k_max, len(keys) - 1) + 1):
        # see: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        km = KMeans(n_clusters=k, n_init=50, random_state=0).fit(X)
        if len(set(km.labels_)) < k:
            continue
        # see: https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient
        # and: https://stackoverflow.com/questions/51138686/how-to-use-silhouette-score-in-k-means-clustering-from-sklearn-library
        sc = float(silhouette_score(X, km.labels_))
        if best is None or sc > best[0]:
            best = (sc, k, km)
    if best is None:
        return {}
    sc, k, km = best
    sizes = np.bincount(km.labels_, minlength=k)
    teams = set(np.argsort(sizes)[::-1][:2].tolist())
    off = {keys[i]: bool(km.labels_[i] not in teams) for i in range(len(keys))}

    # The two teams must be most of the cast. If the flagged tracks would hold most of the
    # person-instants, the clustering is not separating kits and the flag is withheld: a wrong
    # off-team label would remove an athlete from the reconstruction.
    n_tracked = {kk: 0 for kk in keys}
    for t in sorted(int(x) for x in data["timestamps"]):
        for q in data["frames"][str(t)]:
            tid = q.get("track_id", q.get("id"))
            if tid in n_tracked:
                n_tracked[tid] += 1
    mass = sum(n_tracked.values()) or 1
    flag_mass = sum(n_tracked[kk] for kk in keys if off[kk]) / mass
    d = np.linalg.norm(X[:, None, :] - km.cluster_centers_[None], axis=2).min(1)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    z = (d - med) / max(1.4826 * mad, 1e-6)
    print(f"[people_roles] off-team: k={k} by silhouette ({sc:.3f}), cluster sizes "
          f"{sorted(sizes.tolist(), reverse=True)}, teams are the two largest; "
          f"flagged {sum(off.values())} tracks = {flag_mass * 100:.1f}% of scored mass")
    if flag_mass > max_flag_frac:
        print(f"[people_roles] off-team WITHHELD: that is over {max_flag_frac * 100:.0f}% of the "
              f"mass, so the clustering is not separating kits; nothing flagged")
        return {kk: (False, round(float(zz), 2)) for kk, zz in zip(keys, z)}
    return {kk: (off[kk], round(float(zz), 2)) for kk, zz in zip(keys, z)}


def classify(data: dict, fps: float, static_speed: float, outside_frac: float,
             min_len: int, off_team: dict | None = None) -> tuple[dict, dict]:
    timestamps = sorted(int(t) for t in data.get("timestamps", []))
    frames = data.get("frames", {})

    seq: dict[int, list] = collections.defaultdict(list)
    for t in timestamps:
        for p in frames.get(str(t), []):
            tid = p.get("track_id", p.get("id"))
            seq[tid].append((t, np.asarray(p["pos"][:2], float)))
    for k in seq:
        seq[k].sort()

    stats = {}
    for tid, s in seq.items():
        P = np.array([p for _, p in s])
        sp = []
        for (ta, a), (tb, b) in zip(s, s[1:]):
            dt = (tb - ta) / fps
            if dt > 0:
                sp.append(float(np.linalg.norm(b - a) / dt))
        sp = np.array(sp) if sp else np.array([0.0])
        outside = float(np.mean((np.abs(P[:, 0]) > COURT_X) | (np.abs(P[:, 1]) > COURT_Y)))
        med_v = float(np.median(sp))
        # too short to judge; leave it unlabelled rather than guess
        if len(s) < min_len:
            role = "unknown"
        elif med_v < static_speed:
            role = "bystander"
        elif outside >= outside_frac:
            role = "peripheral"
        else:
            role = "player"
        stats[tid] = {"n": len(s), "median_speed": round(med_v, 2),
                      "p90_speed": round(float(np.percentile(sp, 90)), 2),
                      "outside_frac": round(outside, 3),
                      "median_abs_x": round(float(np.median(np.abs(P[:, 0]))), 1),
                      "median_abs_y": round(float(np.median(np.abs(P[:, 1]))), 1),
                      "role": role}
        if off_team and tid in off_team:
            flag, z = off_team[tid]
            stats[tid]["off_team"] = flag
            stats[tid]["off_team_z"] = z

    out_frames = {}
    for t in timestamps:
        out = []
        for p in frames.get(str(t), []):
            rec = dict(p)
            st = stats[p.get("track_id", p.get("id"))]
            rec["role"] = st["role"]
            if st.get("off_team"):
                rec["off_team"] = True
            out.append(rec)
        out_frames[str(t)] = out
    out = {"timestamps": timestamps, "frames": out_frames}
    for k, v in data.items():
        if k not in ("timestamps", "frames"):
            out[k] = v
    out["roles"] = {str(k): v for k, v in sorted(stats.items())}
    return out, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None, help="default: patch the input in place")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--static_speed", type=float, default=1.0,
                    help="median speed (m/s) below which a track is a bystander")
    ap.add_argument("--outside_frac", type=float, default=0.6,
                    help="fraction of samples outside the court lines to call a track peripheral")
    ap.add_argument("--appearance", default=None,
                    help="people_fused*.json; enables the off-team appearance test (see the "
                         "module docstring). Position and speed cannot see a referee.")
    ap.add_argument("--off_team_min_samples", type=int, default=20,
                    help="minimum person-instants before a track gets a pooled appearance "
                         "template; short tracks make noisy templates that flatten the statistic")
    ap.add_argument("--off_team_z", type=float, default=3.0,
                    help="kept for the log only: the robust-z threshold on the distance to the "
                         "nearest team cluster. off_team_flags no longer uses it to decide")
    ap.add_argument("--min_len", type=int, default=5,
                    help="tracks shorter than this stay 'unknown'")
    args = ap.parse_args()

    inp = Path(args.inp).resolve()
    data = json.loads(inp.read_text(encoding="utf-8"))
    off = (off_team_flags(data, args.appearance, args.off_team_min_samples, args.off_team_z)
           if args.appearance else {})
    out, stats = classify(data, args.fps, args.static_speed, args.outside_frac, args.min_len, off)

    dst = Path(args.out).resolve() if args.out else inp
    # this usually overwrites its own input, so write to a temporary file and swap, so an
    # interruption cannot leave a half-written file behind
    tmp = dst.with_name(dst.stem + ".tmp.json")
    tmp.write_text(json.dumps(out), encoding="utf-8")
    tmp.replace(dst)

    counts = collections.Counter(v["role"] for v in stats.values())
    inst = collections.Counter()
    for v in stats.values():
        inst[v["role"]] += v["n"]
    print(f"[people_roles] {inp.name} -> {dst.name}")
    for role in ("player", "peripheral", "bystander", "unknown"):
        if counts[role]:
            print(f"  {role:<11} {counts[role]:3d} tracks, {inst[role]:5d} person-instants")
    ts = out["timestamps"]
    per = [sum(1 for p in out["frames"][str(t)] if p["role"] == "player") for t in ts]
    allp = [len(out["frames"][str(t)]) for t in ts]
    print(f"  cast per instant: all {min(allp)}-{max(allp)} (median {int(np.median(allp))})"
          f"   players only {min(per)}-{max(per)} (median {int(np.median(per))})")
    for tid, v in sorted(stats.items(), key=lambda kv: -kv[1]["n"])[:12]:
        print(f"    track {tid:3d}  n={v['n']:4d}  v~{v['median_speed']:4.1f} m/s  "
              f"outside {100*v['outside_frac']:3.0f}%  |x|~{v['median_abs_x']:4.1f} "
              f"|y|~{v['median_abs_y']:4.1f}  -> {v['role']}")


if __name__ == "__main__":
    main()
