"""Locks the tracks to a roster of exactly ten players, five per team.

The tracks are first cut wherever they are obviously broken (a teleport, or a long hole).
The pieces are sorted into the two teams by kit colour, with hand-labelled ground truth
overriding the colour where it exists. Pieces that are one athlete traded between two track
ids are merged, and a piece that shadows another piece of the same team within 80 cm is
trimmed. Then a small integer program per team assigns every piece to one of five chains,
as cheaply as possible. The ten slots are named from the ground truth where it exists and
from the jersey reads otherwise. Gaps are filled by interpolating between the two observed
ends, limited to a running speed, and every filled row is marked `interp: true` so nothing
downstream mistakes it for an observation. Observed rows keep all their original fields.

Reads  : work/people/repaired.json and jersey_repaired.json (plus an optional GT file)
Writes : work/people/roster.json, roster_report.md and roster_topdown.png

Run:
    uv run python people/roster_lock.py             # reads and writes work/people/
    uv run python people/roster_lock.py --selftest  # synthetic 10-athlete check
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import lil_matrix

SRC = Path(__file__).resolve().parent.parent
PEO = SRC.parent / "work" / "people"

# Kit colour of each team as (brightness V, saturation S) of the shirt. A real run learns the
# two teams from the footage (learn_kit_centers); these fixed values are only used
# by --selftest.
KIT_CENTERS = {"A": (145.0, 63.0), "B": (80.0, 95.0)}

FPS = 25.0
TELEPORT_M_PER_S = 12.0   # split an atom where implied speed exceeds this
TELEPORT_SLACK_M = 0.6    # fusion position noise allowance on a 1-frame step
HOLE_SPLIT_S = 2.0        # split an atom at a coasting hole longer than this
SHADOW_MED_M = 0.8        # co-present median distance below which a pair is a shadow
SHADOW_MIN_SHARED = 5
GT_DUP_M = 2.0            # same-(team,number) GT rows further apart are both excluded
LINK_SPEED_VETO = 12.0    # m/s implied over a gap
FILL_SPEED_MAX = 9.0      # m/s ceiling for interpolated motion
SWEEP_SPEED = 9.0         # two observed rows implying more than this cannot be one athlete
SWEEP_DT_S = 0.2          # ... over this window (the export stride), not per frame


def build_atoms(frames, ts, athlete_tracks):
    """Cut each track wherever it is obviously broken: a teleport or a long hole.

    Cutting too much is fine, because the assignment afterwards puts the pieces back together.
    Returns a list of atoms (see _atom).
    """
    seq = defaultdict(list)  # tid -> [(t, row)]
    for t in ts:
        for i, r in enumerate(frames[str(t)]):
            if r["track_id"] in athlete_tracks:
                seq[r["track_id"]].append((t, r))
    atoms = []
    for tid, obs in seq.items():
        cur = [obs[0]]
        for prev, nxt in zip(obs, obs[1:]):
            dt = (nxt[0] - prev[0]) / FPS
            d = float(np.linalg.norm(np.array(nxt[1]["pos"][:2]) -
                                     np.array(prev[1]["pos"][:2])))
            if dt > HOLE_SPLIT_S or d > TELEPORT_M_PER_S * dt + TELEPORT_SLACK_M:
                atoms.append(_atom(tid, cur))
                cur = [nxt]
            else:
                cur.append(nxt)
        atoms.append(_atom(tid, cur))
    return atoms


def _atom(tid, obs):
    p = np.array([o[1]["pos"][:2] for o in obs], float)
    return {
        "track": tid, "obs": obs,
        "t0": obs[0][0], "t1": obs[-1][0], "len": len(obs),
        "p0": p[0], "p1": p[-1],
        "ncams": float(np.mean([o[1].get("n_cams", 1) for o in obs])),
    }


def _end_velocity(atom, tail=True, span_s=0.4):
    """Which way a fragment was heading at its start or its end, for joining it to another."""
    obs = atom["obs"]
    t_ref = obs[-1][0] if tail else obs[0][0]
    win = [o for o in obs if abs(o[0] - t_ref) <= span_s * FPS]
    if len(win) < 2:
        return np.zeros(2)
    dt = (win[-1][0] - win[0][0]) / FPS
    if dt <= 0:
        return np.zeros(2)
    v = (np.array(win[-1][1]["pos"][:2]) - np.array(win[0][1]["pos"][:2])) / dt
    n = float(np.linalg.norm(v))
    return v * (8.0 / n) if n > 8.0 else v


# Which team each fragment belongs to, from the kit, the ground truth and the shirt numbers.

def load_gt_anchors(gt_path):
    """Load the hand-labelled ground truth as {(track_id, t): (team, number)}.

    If one label appears twice at the same instant far apart, both rows are dropped.
    """
    anchors = {}
    if not gt_path or not Path(gt_path).exists():
        return anchors
    gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    for inst in gt["instants"]:
        t = inst["t"]
        rows = [p for p in inst["people"]
                if p.get("status") != "ghost" and p.get("team") in ("A", "B")
                and p.get("number")]
        by_label = defaultdict(list)
        for p in rows:
            by_label[(p["team"], p["number"])].append(p)
        for lab, v in by_label.items():
            if len(v) > 1:
                dmax = max(np.linalg.norm(np.array(a["pos"][:2]) -
                                          np.array(b["pos"][:2]))
                           for a in v for b in v)
                if dmax >= GT_DUP_M:      # physically impossible - drop both rows
                    continue
                v = [max(v, key=lambda p: p.get("conf", 0))]
            anchors[(v[0]["track_id"], t)] = lab
    return anchors


def learn_kit_centers(roles, jersey):
    """Find the two team kit colours (V, S) with a length-weighted 2-means over the player tracks.

    The kit colours come from the shirt-number reader. Referees were already set aside by
    roles.py. The brighter kit is called team A.
    """
    kits, weights = [], []
    for tid, meta in roles.items():
        k = jersey.get(tid, {}).get("kit")
        if meta.get("role") != "player" or meta.get("off_team") or not isinstance(k, (list, tuple)):
            continue
        kits.append([float(k[0]), float(k[1])])
        weights.append(max(1.0, float(meta.get("n", 1))))
    if len(kits) < 2:
        raise SystemExit("roster_lock: fewer than two player tracks have a kit colour; "
                         "cannot tell the two teams apart")
    X, wts = np.asarray(kits), np.asarray(weights)
    centres = np.stack([X[X[:, 0].argmax()], X[X[:, 0].argmin()]])   # start from brightest and darkest
    for _ in range(50):
        lab = np.linalg.norm(X[:, None] - centres[None], axis=2).argmin(1)
        if len(set(lab)) < 2:
            raise SystemExit("roster_lock: all player kits look the same; cannot split the two teams")
        new = np.stack([np.average(X[lab == c], axis=0, weights=wts[lab == c]) for c in (0, 1)])
        if np.allclose(new, centres):
            break
        centres = new
    a, b = sorted(centres.tolist(), key=lambda c: -c[0])
    return {"A": tuple(a), "B": tuple(b)}


def team_of_kit(kit):
    if kit is None:
        return None
    v = np.asarray(kit, float)
    return min(KIT_CENTERS, key=lambda c: np.linalg.norm(v - KIT_CENTERS[c]))


def attach_evidence(atoms, jersey, anchors):
    """Attach team, GT labels and jersey number to each atom; drop atoms with no team."""
    out = []
    for a in atoms:
        j = jersey.get(str(a["track"]), {})
        a["gt"] = Counter()
        for (tid, t), lab in anchors.items():
            if tid == a["track"] and a["t0"] <= t <= a["t1"]:
                a["gt"][lab] += 1
        gt_team = Counter(lab[0] for lab in a["gt"].elements())
        kit_team = team_of_kit(j.get("kit"))
        # a human who labelled this fragment twice outranks the colour test
        if gt_team and gt_team.most_common(1)[0][1] >= 2:
            a["team"] = gt_team.most_common(1)[0][0]
        else:
            a["team"] = kit_team
        a["jersey"] = (str(j["number"]), float(j.get("margin", 0))) \
            if j.get("number") is not None and j.get("margin", 0) >= 0.5 else None
        if a["team"] in ("A", "B"):
            out.append(a)
    return out


# One athlete traded back and forth between two track ids, which reads as two people.

MERGE_MAX_COPRESENT_M = 2.0   # any co-present distance beyond this forbids a merge
MERGE_MED_COPRESENT_M = 1.0
MERGE_MIN_SWITCHES = 3        # union must alternate source ids at least this often
MERGE_MIN_OVERLAP = 5         # frames of span overlap to consider a pair at all
HYST_CAMS = 2                 # camera-count margin needed to switch source mid-merge


def interleave_merge(atoms, log):
    """Join two atoms that are one athlete handed back and forth between two track ids.

    A pair is merged when their spans overlap, they are never far apart while co-present,
    the union has no teleport, and the union switches source id many times (a real crossing
    of two athletes switches once). Pairs whose GT labels differ are never merged.
    """
    changed = True
    refused_seen = set()
    while changed:
        changed = False
        atoms.sort(key=lambda a: (a["t0"], a["t1"]))
        for i in range(len(atoms)):
            if changed:
                break
            for j in range(i + 1, len(atoms)):
                a, b = atoms[i], atoms[j]
                if a["team"] != b["team"]:
                    continue
                if min(a["t1"], b["t1"]) - max(a["t0"], b["t0"]) < MERGE_MIN_OVERLAP:
                    continue
                ga, gb = a["gt"].most_common(1), b["gt"].most_common(1)
                if ga and gb and ga[0][0] != gb[0][0]:
                    continue
                why = [] if (a["len"] >= 50 and b["len"] >= 50) else None
                u = _union_atom(a, b, why)
                if u is None:
                    key = (a["track"], a["t0"], b["track"], b["t0"])
                    if why and key not in refused_seen:
                        refused_seen.add(key)
                        log.append(f"merge refused: track {a['track']}@{a['t0']} "
                                   f"({a['len']}f) x track {b['track']}@{b['t0']} "
                                   f"({b['len']}f): {why[0]}")
                    continue
                u["team"] = a["team"]
                u["gt"] = a["gt"] + b["gt"]
                if a["jersey"] and b["jersey"]:
                    u["jersey"] = max((a["jersey"], b["jersey"]), key=lambda x: x[1])
                else:
                    u["jersey"] = a["jersey"] or b["jersey"]
                u["srcs"] = sorted(set(a.get("srcs", [a["track"]]))
                                   | set(b.get("srcs", [b["track"]])))
                log.append(f"interleave-merge: track {a['track']}@{a['t0']} "
                           f"({a['len']}f) + track {b['track']}@{b['t0']} "
                           f"({b['len']}f) -> {u['len']}f")
                atoms[i] = u
                del atoms[j]
                changed = True
                break
    return atoms


def _union_atom(a, b, why=None):
    """Build the merged atom of a pair, or return None if the pair fails a check.

    `why` (a list) collects the reason for a refusal.
    """
    def _no(reason):
        if why is not None:
            why.append(reason)
        return None

    rows_a = dict(a["obs"])
    rows_b = dict(b["obs"])
    shared = sorted(set(rows_a) & set(rows_b))
    ambiguous = set()
    if shared:
        d = {t: float(np.linalg.norm(np.array(rows_a[t]["pos"][:2])
                                     - np.array(rows_b[t]["pos"][:2])))
             for t in shared}
        far = [t for t in shared if d[t] > MERGE_MAX_COPRESENT_M]
        # a few bad frames must not veto a merge that is right about hundreds of others,
        # so they are set aside as ambiguous instead
        if len(far) >= max(3, 0.05 * len(shared)):
            return _no(f"co-present > {MERGE_MAX_COPRESENT_M} m at {len(far)} of "
                       f"{len(shared)} shared frames (first t={far[0]})")
        if len(shared) >= 3 and float(np.median(list(d.values()))) > \
                MERGE_MED_COPRESENT_M:
            return _no(f"co-present median {np.median(list(d.values())):.1f} m")
        ambiguous = {t for t in shared if d[t] > 1.5}
    obs = []
    srcseq = []
    held = None          # which side supplied the previous frame
    for t in sorted(set(rows_a) | set(rows_b)):
        if t in ambiguous:
            continue
        if t in rows_a and t in rows_b:
            # Both ids hold this frame, a few tenths of a metre apart. Switching between them
            # every frame would add speed the person never had, so stay with the side that
            # supplied the last frame unless the other has clearly more cameras.
            na = rows_a[t].get("n_cams", 0)
            nb = rows_b[t].get("n_cams", 0)
            if held == "a" and na >= nb - HYST_CAMS:
                pick = "a"
            elif held == "b" and nb >= na - HYST_CAMS:
                pick = "b"
            else:
                pick = "a" if na >= nb else "b"
            r = rows_a[t] if pick == "a" else rows_b[t]
            held = pick
        else:
            held = "a" if t in rows_a else "b"
            r = rows_a.get(t) or rows_b[t]
            srcseq.append(held)
        obs.append((t, r))
    if sum(1 for x, y in zip(srcseq, srcseq[1:]) if x != y) < MERGE_MIN_SWITCHES:
        return _no("interleaves fewer than MERGE_MIN_SWITCHES times")
    for (t0, r0), (t1, r1) in zip(obs, obs[1:]):
        dt = (t1 - t0) / FPS
        d = float(np.linalg.norm(np.array(r1["pos"][:2]) - np.array(r0["pos"][:2])))
        # switching sides can jump by the duplicate offset (up to a metre), so this check is
        # looser than the one used on a single track
        if dt <= HOLE_SPLIT_S and d > TELEPORT_M_PER_S * dt + 1.2:
            return _no(f"union teleport {d:.1f} m over {dt:.2f} s at t={t0}")
    primary = a if a["len"] >= b["len"] else b
    return _atom(primary["track"], obs)


# Shadows: a second copy of somebody left by the fusion, following them a short distance away.

def shadow_cut(atoms, log):
    """Cut the frames where the weaker of two atoms shadows the other within SHADOW_MED_M.

    Pairs whose GT labels name different athletes, or that carry two different confident
    jersey numbers, are left alone.
    """
    changed = []
    for i in range(len(atoms)):
        for k in range(i + 1, len(atoms)):
            a, b = atoms[i], atoms[k]
            if a["team"] != b["team"] or a["track"] == b["track"]:
                continue
            ga = a["gt"].most_common(1)
            gb = b["gt"].most_common(1)
            if ga and gb and ga[0][0] != gb[0][0]:
                continue
            if (a["jersey"] and b["jersey"] and not ga and not gb
                    and a["jersey"][0] != b["jersey"][0]):
                continue
            pa = {t: np.array(r["pos"][:2]) for t, r in a["obs"]}
            pb = {t: np.array(r["pos"][:2]) for t, r in b["obs"]}
            shared = sorted(set(pa) & set(pb))
            if len(shared) < SHADOW_MIN_SHARED:
                continue
            dists = [float(np.linalg.norm(pa[t] - pb[t])) for t in shared]
            if float(np.median(dists)) >= SHADOW_MED_M:
                continue
            weak = a if a["ncams"] <= b["ncams"] else b
            near = {t for t, d in zip(shared, dists) if d < SHADOW_MED_M}
            before = weak["len"]
            weak["obs"] = [(t, r) for t, r in weak["obs"] if t not in near]
            log.append(f"shadow: atoms track {a['track']}@{a['t0']} & track "
                       f"{b['track']}@{b['t0']} - cut {before - len(weak['obs'])}f "
                       f"from track {weak['track']}")
            changed.append(id(weak))
    # cutting frames out can split a fragment in two, or leave nothing at all
    rebuilt = []
    for a in atoms:
        if id(a) in changed:
            rebuilt.extend(_resplit(a))
        else:
            rebuilt.append(a)
    return [a for a in rebuilt if a["len"] > 0]


def _resplit(atom):
    if not atom["obs"]:
        return []
    pieces, cur = [], [atom["obs"][0]]
    for prev, nxt in zip(atom["obs"], atom["obs"][1:]):
        dt = (nxt[0] - prev[0]) / FPS
        d = float(np.linalg.norm(np.array(nxt[1]["pos"][:2]) -
                                 np.array(prev[1]["pos"][:2])))
        if dt > HOLE_SPLIT_S or d > TELEPORT_M_PER_S * dt + TELEPORT_SLACK_M:
            pieces.append(_atom(atom["track"], cur))
            cur = [nxt]
        else:
            cur.append(nxt)
    pieces.append(_atom(atom["track"], cur))
    for p in pieces:
        p.update({k: atom[k] for k in ("team", "gt", "jersey")
                  if k in atom})
        if "srcs" in atom:
            p["srcs"] = atom["srcs"]
    return pieces


# The assignment itself: cover every fragment with exactly five chains per team.

def solve_team(atoms, K, t_last, log):
    """Assign every atom to one of exactly K athletes, as cheaply as possible (a small MILP).

    Variables: e_ij (link i -> j), s_i (a path starts at i), z_i (ends at i), d_i (atom dropped).
    Each atom: s_i + sum_j e_ji + d_i = 1 and z_i + sum_j e_ij + d_i = 1; sum s_i = K.
    Returns (paths as lists of atom indices, indices of the dropped atoms).
    """
    n = len(atoms)
    edges = []
    for i in range(n):
        for j in range(n):
            if atoms[j]["t0"] <= atoms[i]["t1"]:
                continue
            gap_s = (atoms[j]["t0"] - atoms[i]["t1"]) / FPS
            d = float(np.linalg.norm(atoms[i]["p1"] - atoms[j]["p0"]))
            if d > LINK_SPEED_VETO * gap_s + TELEPORT_SLACK_M:
                continue
            cost = d / (1.0 + gap_s) + 0.2 * gap_s + _label_pen(atoms[i], atoms[j])
            edges.append((i, j, cost))

    nv = len(edges) + 3 * n           # e..., s..., z..., d...
    c = np.zeros(nv)
    for k, (i, j, cost) in enumerate(edges):
        c[k] = cost
    for i in range(n):
        c[len(edges) + i] = 0.3 * atoms[i]["t0"] / FPS               # start late
        c[len(edges) + n + i] = 0.3 * (t_last - atoms[i]["t1"]) / FPS  # end early
        c[len(edges) + 2 * n + i] = 1.0 * atoms[i]["len"]            # drop

    # one constraint row per fragment for what comes in, another for what goes out, and one
    # final row forcing exactly K chains to begin
    A = lil_matrix((2 * n + 1, nv))
    for k, (i, j, _) in enumerate(edges):
        A[j, k] = 1                  # e_ij is incoming for j
        A[n + i, k] = 1              # and outgoing for i
    for i in range(n):
        A[i, len(edges) + i] = 1                 # s_i incoming
        A[i, len(edges) + 2 * n + i] = 1         # d_i
        A[n + i, len(edges) + n + i] = 1         # z_i outgoing
        A[n + i, len(edges) + 2 * n + i] = 1     # d_i
        A[2 * n, len(edges) + i] = 1             # sum s_i
    rhs = np.ones(2 * n + 1)
    rhs[2 * n] = K

    # a small mixed-integer program
    # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html
    res = milp(c=c,
               constraints=LinearConstraint(A.tocsc(), rhs, rhs),
               integrality=np.ones(nv),
               bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"path cover MILP failed: {res.message}")
    x = np.round(res.x).astype(int)

    succ = {i: j for k, (i, j, _) in enumerate(edges) if x[k]}
    starts = [i for i in range(n) if x[len(edges) + i]]
    dropped = [i for i in range(n) if x[len(edges) + 2 * n + i]]
    for i in dropped:
        log.append(f"dropped atom track {atoms[i]['track']}@{atoms[i]['t0']} "
                   f"({atoms[i]['len']}f)")
    paths = []
    for s in starts:
        path, i = [s], s
        while i in succ:
            i = succ[i]
            path.append(i)
        paths.append(path)
    assert len(paths) == K
    covered = {i for p in paths for i in p}
    assert covered.isdisjoint(dropped) and len(covered) + len(dropped) == n
    return paths, dropped


def _label_pen(a, b):
    ga, gb = a["gt"].most_common(1), b["gt"].most_common(1)
    if ga and gb:
        return -3.0 if ga[0][0] == gb[0][0] else 50.0
    if a["jersey"] and b["jersey"]:
        return -1.0 if a["jersey"][0] == b["jersey"][0] else 5.0
    return 0.0


def label_paths(paths, atoms, team, log):
    """Name each path with a roster number: GT votes dominate, jersey breaks ties."""
    score = defaultdict(Counter)           # path index -> number -> score
    for pi, path in enumerate(paths):
        for i in path:
            for (tm, num), c in atoms[i]["gt"].items():
                if tm == team:
                    score[pi][num] += 3.0 * c
            if atoms[i]["jersey"]:
                score[pi][atoms[i]["jersey"][0]] += 0.5
    numbers = Counter()
    for pi in range(len(paths)):
        numbers.update(score[pi])
    roster = [n for n, _ in numbers.most_common(len(paths))]
    while len(roster) < len(paths):
        roster.append(f"u{len(roster)}")
    cost = np.zeros((len(paths), len(roster)))
    for pi in range(len(paths)):
        for ni, num in enumerate(roster):
            cost[pi, ni] = -score[pi].get(num, 0.0)
    # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
    ri, ci = linear_sum_assignment(cost)
    names = {}
    for pi, ni in zip(ri, ci):
        names[pi] = roster[ni]
        log.append(f"slot {team}{roster[ni]}: atoms "
                   + ", ".join(f"t{atoms[i]['track']}@{atoms[i]['t0']}"
                               for i in paths[pi]))
    return names


# Filling in the instants nobody was observed at, between two instants where they were.

def fill_slot(obs_by_t, atoms_of_path, ts):
    """Full per-frame trajectory of one slot: observed positions, Hermite-filled gaps, held ends.

    Returns {t: (pos2, interp_flag)} over every frame of ts.
    """
    knots = sorted(obs_by_t)
    vel = {}
    for a in atoms_of_path:
        vel[a["t0"]] = _end_velocity(a, tail=False)
        vel[a["t1"]] = _end_velocity(a, tail=True)
    out = {}
    for t in ts:
        if t in obs_by_t:
            out[t] = (np.asarray(obs_by_t[t], float), False)
    for ta, tb in zip(knots, knots[1:]):
        if tb - ta <= 1:
            continue
        pa, pb = np.asarray(obs_by_t[ta], float), np.asarray(obs_by_t[tb], float)
        gap_s = (tb - ta) / FPS
        va = vel.get(ta, np.zeros(2)) * gap_s
        vb = vel.get(tb, np.zeros(2)) * gap_s
        seg = {}
        for t in range(ta + 1, tb):
            u = (t - ta) / (tb - ta)
            h00 = 2 * u**3 - 3 * u**2 + 1
            h10 = u**3 - 2 * u**2 + u
            h01 = -2 * u**3 + 3 * u**2
            h11 = u**3 - u**2
            seg[t] = h00 * pa + h10 * va + h01 * pb + h11 * vb
        # clamp: if the Hermite overshoots human speed, ease linearly instead
        pts = [pa] + [seg[t] for t in range(ta + 1, tb)] + [pb]
        step_max = max(float(np.linalg.norm(q - p)) for p, q in zip(pts, pts[1:]))
        if step_max * FPS > FILL_SPEED_MAX and step_max * FPS > \
                np.linalg.norm(pb - pa) / max(gap_s, 1e-6) * 1.5:
            for t in range(ta + 1, tb):
                u = (t - ta) / (tb - ta)
                u = u * u * (3 - 2 * u)          # smoothstep ease
                seg[t] = (1 - u) * pa + u * pb
        for t, p in seg.items():
            out[t] = (p, True)
    for t in ts:                                  # hold before first / after last
        if t not in out:
            ref = knots[0] if t < knots[0] else knots[-1]
            out[t] = (np.asarray(obs_by_t[ref], float), True)
    return out


def run(args):
    d = json.loads(Path(args.tracked).read_text(encoding="utf-8"))
    jersey = json.loads(Path(args.jersey).read_text(encoding="utf-8"))
    anchors = load_gt_anchors(args.gt)
    frames, roles = d["frames"], d["roles"]
    KIT_CENTERS.clear()
    KIT_CENTERS.update(learn_kit_centers(roles, jersey))
    print("team kits (V, S): " + ", ".join(f"{k} ({v[0]:.0f}, {v[1]:.0f})" for k, v in KIT_CENTERS.items()))
    ts = sorted(int(t) for t in frames)
    t_last = ts[-1]
    log = []

    # athlete pool: player-role non-off-team tracks with a team kit, plus any track the
    # hand GT anchors to a team at >=2 instants (catches e.g. a stationary athlete that
    # roles.py called a bystander)
    gt_tracks = Counter(tid for (tid, _t) in anchors)
    pool = set()
    for tid_s, meta in roles.items():
        tid = int(tid_s)
        j = jersey.get(tid_s, {})
        if meta.get("role") == "player" and not meta.get("off_team") \
                and team_of_kit(j.get("kit")) in ("A", "B"):
            pool.add(tid)
        elif gt_tracks.get(tid, 0) >= 2:
            pool.add(tid)
            log.append(f"pooled non-player track {tid} on GT evidence "
                       f"(role {meta.get('role')})")

    atoms = build_atoms(frames, ts, pool)
    atoms = attach_evidence(atoms, jersey, anchors)
    n0 = len(atoms)
    atoms = interleave_merge(atoms, log)
    n1 = len(atoms)
    atoms = shadow_cut(atoms, log)
    print(f"{len(pool)} tracks -> {n0} atoms -> {n1} after interleave merge "
          f"-> {len(atoms)} after shadow cut")

    slots = {}                     # slot id -> dict
    slot_rows = defaultdict(dict)  # slot id -> {t: original row}
    slot_atoms = {}                # slot id -> [atom]
    sid = 0
    dropped_atoms = []
    for team in ("A", "B"):
        team_atoms = [a for a in atoms if a["team"] == team]
        paths, dropped = solve_team(team_atoms, args.k, t_last, log)
        dropped_atoms += [team_atoms[i] for i in dropped]
        names = label_paths(paths, team_atoms, team, log)
        order = sorted(range(len(paths)), key=lambda pi: str(names[pi]))
        for pi in order:
            for i in paths[pi]:
                for t, r in team_atoms[i]["obs"]:
                    slot_rows[sid][t] = r
            slot_atoms[sid] = [team_atoms[i] for i in paths[pi]]
            slots[sid] = {
                "team": team, "number": names[pi],
                "src_tracks": sorted({s for i in paths[pi] for s in
                                      team_atoms[i].get("srcs",
                                                        [team_atoms[i]["track"]])}),
                "path_atoms": [[team_atoms[i]["track"], team_atoms[i]["t0"],
                                team_atoms[i]["t1"]] for i in paths[pi]],
            }
            sid += 1

    def _fill(s):
        return fill_slot({t: r["pos"][:2] for t, r in slot_rows[s].items()},
                         slot_atoms[s], ts)

    def speed_audit(s, log):
        """Log the slot segments that imply more than SWEEP_SPEED m/s over SWEEP_DT_S. Never drops rows.

        Over 0.2 s rather than per frame, because frame-to-frame jitter alone reaches 12 m/s.
        A jump that fast means one of the two rows is misassigned; the log names the source
        tracks so it can be checked.
        """
        obs = sorted(slot_rows[s])
        step = max(1, int(round(SWEEP_DT_S * FPS)))
        flagged = 0
        for k in range(len(obs) - 1):
            ta = obs[k]
            tb = next((x for x in obs[k + 1:] if x - ta >= step), None)
            if tb is None:
                break
            dt = (tb - ta) / FPS
            d = float(np.linalg.norm(np.array(slot_rows[s][tb]["pos"][:2])
                                     - np.array(slot_rows[s][ta]["pos"][:2])))
            v = d / dt
            if v > SWEEP_SPEED:
                flagged += 1
                sa = slot_rows[s][ta]["track_id"]
                sb = slot_rows[s][tb]["track_id"]
                why = ("sprint burst + fusion jitter" if sa == sb
                       else "SOURCE CHANGE - one row may be misassigned")
                log.append(f"speed audit: slot {slots[s]['team']}{slots[s]['number']} "
                           f"t={ta}->{tb} implies {v:.1f} m/s ({d:.2f} m in {dt:.2f} s), "
                           f"tracks {sa}->{sb}: {why}")
        return flagged

    flagged = sum(speed_audit(s, log) for s in sorted(slot_rows))
    if flagged:
        print(f"speed audit: {flagged} segment(s) imply > {SWEEP_SPEED} m/s over "
              f"{SWEEP_DT_S} s - see the report (not auto-repaired: see the docstring)")

    traj = {s: _fill(s) for s in slot_rows}

    # A dropped atom may follow different athletes, so each row is judged on its own against
    # the slot trajectories of its team: grafted into the nearest slot's hole when the match
    # is clear, dropped when that slot is already observed there.
    GRAFT_M, AMBIG_M = 1.2, 1.8
    lost_rows = 0
    grafted_keys = set()
    graft_src = defaultdict(set)    # slot -> source track ids grafted into it

    def _label_slot(a):
        """The slot the atom's own confident evidence names, else None."""
        if a["gt"]:
            (tm, num), c = a["gt"].most_common(1)[0]
            if c >= 2:
                for s2, m2 in slots.items():
                    if m2["team"] == tm and str(m2["number"]) == str(num):
                        return s2
        if a["jersey"]:
            for s2, m2 in slots.items():
                if m2["team"] == a["team"] and str(m2["number"]) == a["jersey"][0]:
                    return s2
        return None

    for pass_no in (1, 2):          # pass 2 re-judges against the refilled paths
        changed = set()
        for a in dropped_atoms:
            stats = Counter()
            lab = _label_slot(a)
            for t, r in a["obs"]:
                if (a["track"], t) in grafted_keys:
                    continue
                p = np.array(r["pos"][:2])
                cand = sorted((float(np.linalg.norm(p - traj[s][t][0])), s)
                              for s, meta in slots.items()
                              if meta["team"] == a["team"])
                d_lab = next((d for d, s in cand if s == lab), None)
                if lab is not None and d_lab is not None and d_lab <= GRAFT_M:
                    s = lab             # the atom's own label wins any tie
                elif lab is not None and d_lab is not None and d_lab <= AMBIG_M:
                    stats["ambiguous"] += 1     # near its label but not clearly on it
                    continue
                elif not cand or cand[0][0] > GRAFT_M:
                    stats["far"] += 1
                    continue
                elif len(cand) > 1 and cand[1][0] < AMBIG_M:
                    stats["ambiguous"] += 1
                    continue
                else:
                    s = cand[0][1]      # label far or absent: unambiguous geometry
                if t in slot_rows[s]:
                    stats["shadow"] += 1
                    continue
                slot_rows[s][t] = r
                grafted_keys.add((a["track"], t))
                graft_src[s].add(a["track"])
                changed.add(s)
                stats[f"graft {slots[s]['team']}{slots[s]['number']}"] += 1
            if pass_no == 2 or stats:
                if pass_no == 1:
                    gt_note = f", GT {dict(a['gt'])}" if a["gt"] else ""
                    log.append(f"drop audit: track {a['track']}@{a['t0']} "
                               f"({a['len']}f): {dict(stats)}{gt_note}")
                elif any(k.startswith("graft") for k in stats):
                    log.append(f"drop audit pass 2: track {a['track']}@{a['t0']}: "
                               f"{dict(stats)}")
            if pass_no == 2:
                lost_rows += stats.get("far", 0) + stats.get("ambiguous", 0)
        for s in changed:
            slots[s]["src_tracks"] = sorted(set(slots[s]["src_tracks"])
                                            | graft_src[s])
            traj[s] = _fill(s)
        if not changed:
            break
    if lost_rows:
        print(f"NOTE: {lost_rows} dropped row(s) match no slot unambiguously "
              f"(shadows near screens / fusion outliers) - see the drop audit")

    # emit
    out_frames = {str(t): [] for t in ts}
    n_interp = Counter()
    for s, rows in slot_rows.items():
        for t in ts:
            p, interp = traj[s][t]
            if interp:
                row = {"id": s, "pos": [float(p[0]), float(p[1]), 0.0],
                       "n_cams": 0, "cams": [], "views": [], "track_id": s,
                       "slot": s, "role": "player", "interp": True}
                n_interp[s] += 1
            else:
                row = dict(rows[t])
                row["src_track_id"] = row["track_id"]
                row["track_id"] = s
                row["slot"] = s
                # every slot is an athlete, so any bystander or off-team label
                # from the source track is dropped
                row["role"] = "player"
                row.pop("off_team", None)
                row["interp"] = False
            out_frames[str(t)].append(row)
        slots[s]["n_obs"] = len(rows)
        slots[s]["n_interp"] = n_interp[s]

    # slot-level smoothing for display and infill anchoring (raw pos untouched)
    smooth_step = 0.0
    for s in slots:
        p = np.array([next(r for r in out_frames[str(t)]
                           if r["track_id"] == s)["pos"][:2] for t in ts])
        kern = np.exp(-0.5 * (np.arange(-4, 5) / 2.0) ** 2)
        kern /= kern.sum()
        pad = np.vstack([p[:1].repeat(4, 0), p, p[-1:].repeat(4, 0)])
        ps = np.stack([np.convolve(pad[:, k], kern, "valid") for k in (0, 1)], 1)
        for i, t in enumerate(ts):
            row = next(r for r in out_frames[str(t)] if r["track_id"] == s)
            row["pos_smooth"] = [float(ps[i, 0]), float(ps[i, 1]), 0.0]
        smooth_step = max(smooth_step,
                          float(np.linalg.norm(np.diff(ps, axis=0), axis=1).max()))

    # self-checks
    for t in ts:
        assert len(out_frames[str(t)]) == 2 * args.k, f"frame {t}: wrong row count"
    worst = 0.0
    for s in slots:
        p_prev = None
        for t in ts:
            p = np.array(next(r for r in out_frames[str(t)]
                              if r["track_id"] == s)["pos"][:2])
            if p_prev is not None:
                worst = max(worst, float(np.linalg.norm(p - p_prev)))
            p_prev = p
    frac_interp = sum(n_interp.values()) / (len(ts) * 2 * args.k)
    print(f"self-check: {2 * args.k} rows at all {len(ts)} frames; "
          f"max raw step {worst:.2f} m/frame, smoothed {smooth_step:.2f}; "
          f"interpolated {frac_interp:.1%}")

    # GT agreement of the locked naming (slot_rows holds the original rows, so
    # r["track_id"] is still the source track id the anchors are keyed on)
    ok = bad = 0
    for s, meta in slots.items():
        for t, r in slot_rows[s].items():
            lab = anchors.get((r["track_id"], t))
            if lab is None:
                continue
            if lab == (meta["team"], str(meta["number"])):
                ok += 1
            else:
                bad += 1
                log.append(f"GT disagree: slot {meta['team']}{meta['number']} "
                           f"holds track {r['track_id']} at t={t}, GT says "
                           f"{lab[0]}{lab[1]}")
    print(f"GT anchors on locked slots: {ok} agree / {bad} disagree")

    out = {
        "timestamps": ts,
        "frames": out_frames,
        "roles": {str(s): {"role": "player", "off_team": False, "n": len(ts)}
                  for s in slots},
        "slots": {str(s): v for s, v in slots.items()},
        "roster_lock": {
            "source": Path(args.tracked).name, "jersey": Path(args.jersey).name,
            "gt": Path(args.gt).name if args.gt else None,
            "k_per_team": args.k, "n_atoms": len(atoms),
            "max_step_m": round(worst, 3), "frac_interp": round(frac_interp, 4),
            "gt_agree": ok, "gt_disagree": bad,
            "argv": sys.argv[1:],
        },
        "source": d.get("source", "repaired.json"),
    }
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {args.out}")

    if args.fig:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 5, figsize=(22, 7), sharex=True,
                                     sharey=True)
            for s, ax in zip(sorted(slots), axes.ravel()):
                rows = [next(r for r in out_frames[str(t)] if r["track_id"] == s)
                        for t in ts]
                obs = np.array([r["pos"][:2] for r in rows if not r["interp"]])
                itp = np.array([r["pos"][:2] for r in rows if r["interp"]])
                ax.plot(obs[:, 0], obs[:, 1], ".", ms=1.5, color="tab:blue",
                        label="observed")
                if len(itp):
                    ax.plot(itp[:, 0], itp[:, 1], ".", ms=2.5,
                            color="tab:orange", label="interpolated")
                ax.set_title(f"slot {s} = {slots[s]['team']}"
                             f"{slots[s]['number']} "
                             f"({slots[s]['n_interp']} interp)")
                ax.set_xlim(-16, 16)
                ax.set_ylim(-9, 9)
                ax.set_aspect("equal")
            axes[0, 0].legend(loc="upper left", fontsize=8)
            fig.suptitle("Roster lock: ten athlete slots, observed vs "
                         "interpolated court positions")
            fig.tight_layout()
            fig.savefig(args.fig, dpi=110)
            print(f"wrote {args.fig}")
        except Exception as e:      # figure is QA-only, never fail the lock on it
            print(f"figure skipped: {e}")

    rep = ["# Roster lock report", "",
           f"in: {args.tracked}", f"out: {args.out}", ""]
    for s, meta in slots.items():
        rep.append(f"- slot {s} = {meta['team']}{meta['number']}: "
                   f"{meta['n_obs']} obs + {meta['n_interp']} interp, "
                   f"tracks {meta['src_tracks']}")
    rep += ["", "## log"] + [f"- {x}" for x in log]
    Path(args.report).write_text("\n".join(rep), encoding="utf-8")
    print(f"wrote {args.report}")


# A synthetic ten-athlete clip, so the solver can be checked against a known answer.

def selftest():
    """Synthetic 10-athlete clip: fragmented tracks plus a shadow and an interleaved duplicate.
    The lock must reassemble every slot with no mixed athletes and full coverage."""
    rng = np.random.default_rng(7)
    T = 300
    ts = list(range(T))
    truth = {}
    for a in range(10):
        way = rng.uniform([-12, -6], [12, 6], size=(4, 2))
        path = np.vstack([np.linspace(way[i], way[i + 1], T // 3)
                          for i in range(3)])
        truth[a] = path + rng.normal(0, 0.05, path.shape)

    frames = {str(t): [] for t in ts}
    jersey = {}
    roles = {}
    tid = 0
    gt = {"instants": []}
    seg_of = {}
    for a in range(10):
        team = "A" if a < 5 else "B"
        kit = np.array(KIT_CENTERS[team]) + rng.normal(0, 4, 2)
        cuts = sorted(int(x) for x in rng.choice(range(20, T - 20), size=3,
                                                 replace=False))
        bounds = [0] + cuts + [T]
        for b0, b1 in zip(bounds, bounds[1:]):
            hole = int(rng.integers(0, 12))     # frames lost at the seam
            for t in range(b0, max(b0 + 1, b1 - hole)):
                frames[str(t)].append({
                    "pos": [float(truth[a][t, 0]), float(truth[a][t, 1]), 0.0],
                    "n_cams": 6, "cams": [], "views": [], "track_id": tid,
                })
            jersey[str(tid)] = {"kit": kit.tolist(),
                                "number": str(a * 7 % 30), "margin": 0.9}
            roles[str(tid)] = {"role": "player", "off_team": False, "n": b1 - b0}
            seg_of[tid] = a
            tid += 1
    # a shadow duplicate of athlete 0 and a junk fragment
    for t in range(50, 80):
        frames[str(t)].append({
            "pos": [float(truth[0][t, 0] + 0.3), float(truth[0][t, 1]), 0.0],
            "n_cams": 2, "cams": [], "views": [], "track_id": tid})
    jersey[str(tid)] = {"kit": (np.array(KIT_CENTERS["A"])
                               + rng.normal(0, 4, 2)).tolist(),
                        "number": "0", "margin": 0.9}
    roles[str(tid)] = {"role": "player", "off_team": False, "n": 30}
    seg_of[tid] = 0
    tid += 1
    # an interleaved duplicate: every other observation of athlete 8 in [120,220]
    # is traded to a fresh track id (the tracker's duplicate-identity failure mode)
    moved = 0
    for t in range(120, 220, 2):
        for r in frames[str(t)]:
            if seg_of.get(r["track_id"]) == 8:
                r["track_id"] = tid
                moved += 1
                break
    jersey[str(tid)] = {"kit": (np.array(KIT_CENTERS["B"])
                               + rng.normal(0, 4, 2)).tolist(),
                        "number": str(8 * 7 % 30), "margin": 0.9}
    roles[str(tid)] = {"role": "player", "off_team": False, "n": moved}
    seg_of[tid] = 8
    tid += 1

    for t in range(0, T, 25):
        inst = {"t": t, "people": []}
        for r in frames[str(t)]:
            a = seg_of.get(r["track_id"])
            if a is None or rng.random() > 0.6:
                continue
            inst["people"].append({
                "track_id": r["track_id"], "team": "A" if a < 5 else "B",
                "number": str(a * 7 % 30), "pos": r["pos"], "status": "ok"})
        gt["instants"].append(inst)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "tracked.json").write_text(json.dumps(
            {"timestamps": ts, "frames": frames, "roles": roles}), encoding="utf-8")
        (td / "jersey.json").write_text(json.dumps(jersey), encoding="utf-8")
        (td / "gt.json").write_text(json.dumps(gt), encoding="utf-8")
        args = argparse.Namespace(
            tracked=str(td / "tracked.json"), jersey=str(td / "jersey.json"),
            gt=str(td / "gt.json"), out=str(td / "out.json"),
            report=str(td / "rep.md"), fig="", k=5)
        run(args)
        out = json.loads((td / "out.json").read_text(encoding="utf-8"))
        rep = (td / "rep.md").read_text(encoding="utf-8")
        handled = [l for l in rep.splitlines()
                   if l.startswith(("- shadow:", "- dropped", "- interleave"))]
        print("merge/shadow/drop log:", *handled, sep="\n  ")
        assert any("interleave-merge" in l for l in handled), \
            "interleave merge did not fire on the planted duplicate"

        # purity: every observed row's src track must belong to the slot's athlete
        errs = 0
        slot_athlete = {}
        for s, meta in out["slots"].items():
            athletes = Counter(seg_of[t] for t in meta["src_tracks"])
            slot_athlete[s] = athletes.most_common(1)[0][0]
            if len(athletes) > 1:
                errs += 1
                print(f"IMPURE slot {s}: {dict(athletes)}")
        # trajectory accuracy incl. filled frames
        worst = 0.0
        for t in ts:
            for r in out["frames"][str(t)]:
                a = slot_athlete[str(r["slot"])]
                worst = max(worst, float(np.linalg.norm(
                    np.array(r["pos"][:2]) - truth[a][t])))
        print(f"selftest: impure slots {errs}, worst position error "
              f"{worst:.2f} m (incl. interpolated)")
        assert errs == 0 and worst < 1.5, "SELFTEST FAILED"
        print("SELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracked", default=str(PEO / "repaired.json"))
    ap.add_argument("--jersey", default=str(PEO / "jersey_repaired.json"))
    ap.add_argument("--gt", default="",
                    help="optional hand-labelled answers (gt_answers.json from the Annotate tab) "
                         "for these same tracks; they override the automatic team and number")
    ap.add_argument("--out", default=str(PEO / "roster.json"))
    ap.add_argument("--report", default=str(PEO / "roster_report.md"))
    ap.add_argument("--fig", default=str(PEO / "roster_topdown.png"),
                    help="top-down QA figure; empty string skips it")
    ap.add_argument("--k", type=int, default=5, help="athletes per team")
    ap.add_argument("--fps", type=float, default=25.0, help="frame rate of the videos")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    global FPS
    FPS = args.fps
    if args.selftest:
        selftest()
    else:
        run(args)


if __name__ == "__main__":
    main()
