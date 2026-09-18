# -*- coding: utf-8 -*-
"""Fixes identity errors in the assembled tracks using the shirt numbers.

The errors left by the assembly are mostly swaps between teammates, which appearance cannot
separate. A shirt number does not change with lighting, pose or camera, so this stage can do
two things. It splits a track where its reads change from one well-supported number to
another, because such a track carries two people. And it joins two pieces that carry the
same confident number, never exist at the same instant, and could be one person's path.

Every edit has a guard. A split needs enough weight behind both numbers (--min_reads_side,
--min_weight, --min_share), so one stray read cannot cut a track. A join needs a walkable
merged trajectory (--max_speed, --max_gap_s) and both pieces to look like the same team
(--team_gate on the appearance MMD^2 from the fused file), because both teams number from
the same range. The edit counts are printed and stored in the `jersey_repair` block.

Reads  : the assembled tracks, the jersey reads and the fused file (for the appearance)
Writes : the same tracks with new ids; `roles` is dropped, so run roles.py again afterwards

    uv run python -m people.jersey_repair \\
        --tracked ../work/people/assembled.json \\
        --jersey  ../work/people/jersey_assembled.json \\
        --fused   ../work/people/fused.json \\
        --out     ../work/people/repaired.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def track_reads(jer):
    tracks = jer.get("tracks", jer)
    out = {}
    for k, v in tracks.items():
        if isinstance(v, dict) and v.get("reads"):
            out[int(k)] = sorted(v["reads"], key=lambda r: r["t"])
    return out


def best_number(reads, lo, hi):
    """The winning number over reads[lo:hi]. Returns (number, its weight, its share of the total)."""
    w = collections.Counter()
    for r in reads[lo:hi]:
        w[r["n"]] += r["c"]
    if not w:
        return None, 0.0, 0.0
    n, ww = w.most_common(1)[0]
    return n, ww, ww / max(sum(w.values()), 1e-9)


def find_splits(reads, args, depth=0):
    """Indices in `reads` where the winning number changes from one to another, found recursively."""
    if depth >= args.max_depth or len(reads) < 2 * args.min_reads_side:
        return []
    best = None
    for m in range(args.min_reads_side, len(reads) - args.min_reads_side + 1):
        na, wa, sa = best_number(reads, 0, m)
        nb, wb, sb = best_number(reads, m, len(reads))
        if na is None or nb is None or na == nb:
            continue
        if wa < args.min_weight or wb < args.min_weight:
            continue
        if sa < args.min_share or sb < args.min_share:
            continue
        score = min(wa, wb) * min(sa, sb)
        if best is None or score > best[0]:
            best = (score, m, na, nb)
    if best is None:
        return []
    _, m, na, nb = best
    return (find_splits(reads[:m], args, depth + 1)
            + [m]
            + [m + x for x in find_splits(reads[m:], args, depth + 1)])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracked", required=True)
    ap.add_argument("--jersey", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--min_reads_side", type=int, default=3,
                    help="reads a side of a split must have before the split is considered")
    ap.add_argument("--min_weight", type=float, default=2.0,
                    help="confidence-weighted support each side must reach for its own number")
    ap.add_argument("--min_share", type=float, default=0.6,
                    help="share of its side's weight the winning number must hold")
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--no_split", action="store_true")
    ap.add_argument("--no_join", action="store_true")
    ap.add_argument("--join_margin", type=float, default=0.5,
                    help="vote margin a piece's number needs before it may anchor a join")
    ap.add_argument("--join_min_weight", type=float, default=3.0)
    ap.add_argument("--max_speed", type=float, default=9.0,
                    help="no step in a joined trajectory may exceed this (m/s) - the OCR-independent"
                         " veto")
    ap.add_argument("--max_gap_s", type=float, default=10.0)
    ap.add_argument("--fused", default=None,
                    help="people_fused*.json; enables the same-team check on joins. Strongly "
                         "recommended: without it, a white #15 and a navy #15 that never co-exist "
                         "can be joined into one athlete.")
    ap.add_argument("--team_gate", type=float, default=0.30,
                    help="max squared-MMD between two pieces for them to be the same team")
    args = ap.parse_args()

    trk = json.loads(Path(args.tracked).read_text(encoding="utf-8"))
    jer = json.loads(Path(args.jersey).read_text(encoding="utf-8"))
    reads_of = track_reads(jer)
    if not reads_of:
        raise SystemExit("[repair] the jersey file has no per-read detail; re-run people/jersey.py")

    ts = sorted(int(t) for t in trk["timestamps"])
    occ = collections.defaultdict(list)
    pos = {}
    for t in ts:
        for j, p in enumerate(trk["frames"][str(t)]):
            k = p.get("track_id", p["id"])
            occ[k].append((t, j))
            pos[(t, j)] = np.asarray(p["pos"][:2], float)

    # first the splits
    pieces, n_split = [], 0
    piece_num = {}
    for k, sam in sorted(occ.items()):
        rd = reads_of.get(k, [])
        cuts_t = []
        if rd and not args.no_split:
            idx = find_splits(rd, args)
            for m in idx:
                # cut halfway between the two reads that disagree
                cuts_t.append(0.5 * (rd[m - 1]["t"] + rd[m]["t"]))
            n_split += len(idx)
        cuts_t = sorted(cuts_t)
        cur, bounds = [], list(cuts_t) + [float("inf")]
        bi = 0
        for s in sam:
            while s[0] > bounds[bi]:
                if cur:
                    pieces.append((k, cur))
                cur = []
                bi += 1
            cur.append(s)
        if cur:
            pieces.append((k, cur))
    # each piece gets its number from the reads inside its own span; the source track travels
    # with the samples so it does not have to be looked up afterwards
    for pi, (src_tid, sam) in enumerate(pieces):
        t0, t1 = sam[0][0], sam[-1][0]
        rr = [r for r in reads_of.get(src_tid, []) if t0 <= r["t"] <= t1]
        piece_num[pi] = best_number(rr, 0, len(rr))
    pieces = [sam for _, sam in pieces]

    # appearance per piece, only used for the same-team test
    emb = {}
    sigma = 1.0
    if args.fused and not args.no_join:
        import sys
        den = Path(__file__).resolve().parent.parent
        if str(den) not in sys.path:
            sys.path.insert(0, str(den))
        import people.assemble as A
        fus = json.loads(Path(args.fused).read_text(encoding="utf-8"))
        app = {}
        for t in fus["timestamps"]:
            for j, q in enumerate(fus["frames"][str(t)]):
                a = q.get("appearance")
                if a is not None:
                    app[(int(t), j)] = A._unit(np.asarray(a, dtype=np.float32))
        if app:
            X = np.stack(list(app.values()))
            A._APP_DIM[0] = X.shape[1]
            sigma = A.median_sigma(X)
            owns = []
            for i, sam in enumerate(pieces):
                E = [app[s] for s in sam if s in app]
                if len(E) >= 4:
                    emb[i] = np.stack([E[q] for q in A._thin(list(range(len(E))), 48)])
                    owns.append(A.self_affinity(emb[i], sigma, shrink=False))
            A._KAA_POOL[0] = float(np.median(owns)) if owns else 0.9
            kaa = {i: A.self_affinity(E, sigma) for i, E in emb.items()}

            def same_team(i, j):
                if i not in emb or j not in emb:
                    return True                 # cannot tell -> do not veto on it
                return A.mmd2(emb[i], emb[j], sigma, kaa[i], kaa[j]) <= args.team_gate
        else:
            def same_team(i, j):
                return True
    else:
        def same_team(i, j):
            return True

    # then the joins: same number, same team, no shared instant, and a walkable trajectory
    n_join = n_team_veto = 0
    parent = list(range(len(pieces)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if not args.no_join:
        groups = {i: set(t for t, _ in pieces[i]) for i in range(len(pieces))}
        members = {i: [i] for i in range(len(pieces))}
        cands = []
        for i in range(len(pieces)):
            ni, wi, si = piece_num[i]
            if ni is None or wi < args.join_min_weight or si < args.join_margin:
                continue
            for j in range(i + 1, len(pieces)):
                nj, wj, sj = piece_num[j]
                if nj != ni or wj < args.join_min_weight or sj < args.join_margin:
                    continue
                if not same_team(i, j):
                    n_team_veto += 1
                    continue
                cands.append((-(min(wi, wj)), i, j))
        cands.sort()
        for _, i, j in cands:
            a, b = find(i), find(j)
            if a == b or (groups[a] & groups[b]):
                continue
            sam = sorted(s for m in members[a] + members[b] for s in pieces[m])
            ok = True
            for u in range(1, len(sam)):
                dt = (sam[u][0] - sam[u - 1][0]) / args.fps
                if dt <= 0:
                    continue
                if dt > args.max_gap_s:
                    ok = False
                    break
                if float(np.hypot(*(pos[sam[u]] - pos[sam[u - 1]]))) / dt > args.max_speed:
                    ok = False
                    break
            if not ok:
                continue
            parent[b] = a
            groups[a] |= groups[b]
            members[a] = members[a] + members[b]
            n_join += 1


    final = collections.defaultdict(list)
    for i in range(len(pieces)):
        final[find(i)].extend(pieces[i])
    order = sorted(final, key=lambda r: min(t for t, _ in final[r]))
    relabel = {r: i for i, r in enumerate(order)}
    assign = {}
    for r, sam in final.items():
        for s in sam:
            assign[s] = relabel[r]

    out_frames = {}
    for t in ts:
        rows = []
        for j, p in enumerate(trk["frames"][str(t)]):
            rec = dict(p)
            rec["id"] = rec["track_id"] = assign[(t, j)]
            rows.append(rec)
        out_frames[str(t)] = rows
    out = {"timestamps": ts, "frames": out_frames}
    for k, v in trk.items():
        if k not in ("timestamps", "frames", "roles"):
            out[k] = v
    out["jersey_repair"] = {"splits": n_split, "joins": n_join,
                            "same_number_different_team_vetoes": n_team_veto,
                            "tracks_in": len(occ), "pieces": len(pieces),
                            "tracks_out": len(final)}
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"[repair] {len(occ)} tracks -> {n_split} number change-points -> {len(pieces)} pieces "
          f"-> {n_join} number-anchored joins -> {len(final)} tracks")
    if n_team_veto:
        print(f"[repair] {n_team_veto} same-number pairs refused as different teams")
    print(f"[repair] wrote {args.out}")
    print("[repair] NOTE: `roles` was dropped (ids changed); re-run people/roles.py on the output.")


if __name__ == "__main__":
    main()
