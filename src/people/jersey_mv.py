"""Reads the shirt number of each track from all cameras at once, one instant at a time.

Several cameras reading the same number at the same synchronised instant are independent
looks at one shirt, so their agreement counts. For each track this picks instants spread
over its lifetime, cuts the torso in every camera that sees it (between the shoulders and
the hips when the keypoints are confident, a fixed band of the box otherwise), reads each
crop with EasyOCR and adds up the reads. Reads that agree across cameras at one instant get
a bonus, scaled by how far apart the cameras are. The instant scores are then summed into
one number per track, with a margin over the runner-up. --read_budget is the total number
of crops read per track.

Reads  : the tracks (with per-view bboxes and keypoints) and the frames in work/frames
Writes : {track_id: {number, margin, score, n_reads, votes, reads, kit, ...}}

Run from src/:
    uv run python -m people.jersey_mv \\
        --tracked ../work/people/assembled.json \\
        --raw4k   ../work/frames \\
        --out     ../work/people/jersey_assembled.json \\
        --read_budget 120 --gpu \\
        --placer ../placements/placed_cameras_redfox_repaired.json \\
        --pnp ../diagnostics/court_pnp_cameras_redfox_repaired.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from people.jersey import kit_stat, torso  # noqa: E402  (identical band + kit statistic)

# COCO-17
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12


def torso_from_keypoints(img, bbox, kps, kp_thr, pad_x=0.18, top_f=0.05, bot_f=0.85):
    """Torso crop from the keypoints, or None if the shoulders or hips are not confident.

    The band is a fraction of the shoulder-to-hip span, so it still holds on a jumping or
    bending athlete.
    """
    if kps is None or len(kps) < 13:
        return None
    k = np.asarray(kps, dtype=np.float32)
    if k.ndim != 2 or k.shape[1] < 3:
        return None
    sho = [i for i in (L_SHO, R_SHO) if k[i, 2] >= kp_thr]
    hip = [i for i in (L_HIP, R_HIP) if k[i, 2] >= kp_thr]
    if not sho or not hip:
        return None
    sy = float(np.mean(k[sho, 1]))
    hy = float(np.mean(k[hip, 1]))
    if hy - sy < 8:
        return None
    xs = k[sho + hip, 0]
    x0, x1 = float(xs.min()), float(xs.max())
    w = max(x1 - x0, 8.0)
    x0 -= pad_x * w
    x1 += pad_x * w
    yA = sy + top_f * (hy - sy)
    yB = sy + bot_f * (hy - sy)
    H, W = img.shape[:2]
    x0, x1 = int(max(0, x0)), int(min(W, x1))
    yA, yB = int(max(0, yA)), int(min(H, yB))
    if yB - yA < 10 or x1 - x0 < 8:
        return None
    return img[yA:yB, x0:x1]


def facing_score(kps, kp_thr):
    """Fraction of confident head keypoints: which way the player is facing. Recorded only."""
    if kps is None or len(kps) < 5:
        return None
    k = np.asarray(kps, dtype=np.float32)
    if k.ndim != 2 or k.shape[1] < 3:
        return None
    head = k[[NOSE, L_EYE, R_EYE, L_EAR, R_EAR], 2]
    return float((head >= kp_thr).mean())


def camera_bearings(placer_path, pnp_path):
    """Floor position (x, y) of each calibrated camera, from the placement and PnP files."""
    import json as _json
    poses = _json.loads(Path(placer_path).read_text(encoding="utf-8"))
    poses = poses.get("cameras", poses)
    pnp = _json.loads(Path(pnp_path).read_text(encoding="utf-8"))
    from camera_frames import _placement_to_cam_from_world
    out = {}
    for cam in sorted(set(poses) & set(pnp)):
        Rwc, twc = _placement_to_cam_from_world(poses[cam])
        C = -Rwc.T @ twc
        out[cam] = (float(C[0]), float(C[1]))
    return out


def viewpoint_diversity(cams, centres, at_xy):
    """How independent the agreeing views are, from 0 (same viewpoint) to 1 (opposite sides).

    Two cameras on the same side see the same face of the jersey and fail the same way, so
    they should not count as two observations. The value is `(1 - cos dtheta) / 2` for the
    widest angle between any two of the cameras, as seen from the athlete at `at_xy`.
    """
    pts = [centres[c] for c in cams if c in centres]
    if len(pts) < 2:
        return 0.0
    th = [np.arctan2(cy - at_xy[1], cx - at_xy[0]) for cx, cy in pts]
    best = 0.0
    for i in range(len(th)):
        for j in range(i + 1, len(th)):
            best = max(best, float((1.0 - np.cos(th[i] - th[j])) / 2.0))
    return best


def instants(tracked, roles_filter, min_h, n_buckets):
    """Pick the instants to read for each track, spread over its lifetime.

    The instants are bucketed in time; inside a bucket the largest box comes first, then the
    number of cameras. Returns {track_id: [(t, views)]} in reading order.
    """
    per = collections.defaultdict(dict)
    for t in tracked["timestamps"]:
        for p in tracked["frames"][str(t)]:
            tid = p.get("track_id", p.get("id"))
            if roles_filter is not None and p.get("role") not in roles_filter:
                continue
            vs = [v for v in p.get("views", [])
                  if v.get("bbox") and (v["bbox"][3] - v["bbox"][1]) >= min_h]
            if vs:
                per[tid][int(t)] = vs
    out = {}
    for tid, byt in per.items():
        rows = [(t, vs) for t, vs in byt.items()]
        if not rows:
            continue
        ts = [r[0] for r in rows]
        t0, t1 = min(ts), max(ts)
        span = max(t1 - t0, 1)
        buckets = collections.defaultdict(list)
        for t, vs in rows:
            b = min(int((t - t0) / span * n_buckets), n_buckets - 1)
            buckets[b].append((t, vs))
        for b in buckets:
            buckets[b].sort(key=lambda r: (-max(v["bbox"][3] - v["bbox"][1] for v in r[1]),
                                           -len(r[1])))
        order, i, keys = [], 0, sorted(buckets)
        while any(len(buckets[b]) > i for b in keys):
            for b in keys:
                if len(buckets[b]) > i:
                    order.append(buckets[b][i])
            i += 1
        out[tid] = order
    return out


def read_crop(reader, crop, upscale_to, min_conf):
    """Run the recogniser on one torso crop. Returns [(number, confidence)] for 1-2 digit reads."""
    if crop is None or crop.size == 0 or crop.shape[0] < 12:
        return []
    s = max(1.0, upscale_to / crop.shape[0])
    if s > 1.0:
        crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)),
                          interpolation=cv2.INTER_CUBIC)
    try:
        res = reader.readtext(crop, allowlist="0123456789", detail=1, paragraph=False)
    except Exception:
        return []
    out = []
    for r in res:
        txt = str(r[1]).strip()
        conf = float(r[2]) if len(r) > 2 else 0.0
        if txt.isdigit() and 1 <= len(txt) <= 2 and conf >= min_conf:
            out.append((str(int(txt)), conf))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracked", required=True)
    ap.add_argument("--raw4k", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--read_budget", type=int, default=120,
                    help="TOTAL recogniser calls per track, so this is comparable to "
                         "jersey.py --per_track at the same cost")
    ap.add_argument("--max_views", type=int, default=5,
                    help="cameras read per instant, largest box first")
    ap.add_argument("--min_h", type=float, default=150.0, help="skip boxes shorter than this (px)")
    ap.add_argument("--time_buckets", type=int, default=12)
    ap.add_argument("--min_conf", type=float, default=0.30)
    ap.add_argument("--kp_thr", type=float, default=3.0,
                    help="keypoint score for the pose-guided band (detect.py used 3.0)")
    ap.add_argument("--agree_bonus", type=float, default=1.0,
                    help="weight on cross-view agreement, scaled by how INDEPENDENT the agreeing "
                         "views were (see viewpoint_diversity). 0 reduces this to a flat vote")
    ap.add_argument("--rel_h", type=float, default=0.55,
                    help="skip a view whose box is smaller than this fraction of the best view at "
                         "the same instant - a worse look at the same number is noise, not evidence")
    ap.add_argument("--placer", default="",
                    help="camera placement JSON; without it the agreement bonus is disabled, "
                         "because independence cannot be established")
    ap.add_argument("--pnp", default="")
    ap.add_argument("--roles", default="player")
    ap.add_argument("--upscale_to", type=int, default=160)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    import easyocr
    # see: https://github.com/JaidedAI/EasyOCR
    reader = easyocr.Reader(["en"], gpu=args.gpu, verbose=False)

    tracked = json.loads(Path(args.tracked).read_text(encoding="utf-8"))
    roles_filter = None if args.roles == "all" else set(args.roles.split(","))
    raw = Path(args.raw4k)

    centres = {}
    if args.placer and args.pnp:
        centres = camera_bearings(args.placer, args.pnp)
        print(f"[jersey_mv] {len(centres)} camera centres for viewpoint-diversity weighting")
    else:
        print("[jersey_mv] no --placer/--pnp: cross-view agreement DISABLED (independence "
              "cannot be established without the camera geometry)")
    pos_at = {}
    for t in tracked["timestamps"]:
        for pp in tracked["frames"][str(t)]:
            pos_at[(pp.get("track_id", pp.get("id")), int(t))] = pp["pos"][:2]
    per = instants(tracked, roles_filter, args.min_h, args.time_buckets)
    print(f"[jersey_mv] {len(per)} tracks; budget {args.read_budget} reads each, "
          f"up to {args.max_views} views per instant")

    out, t_start, n_read = {}, time.time(), 0
    cache: dict[tuple, np.ndarray] = {}
    n_pose = n_fixed = 0
    for tid in sorted(per, key=lambda k: -len(per[k])):
        score: collections.Counter = collections.Counter()
        reads, kits, div_seen = [], [], []
        used = 0
        n_inst = n_multi = 0
        for t, views in per[tid]:
            if used >= args.read_budget:
                break
            vs = sorted(views, key=lambda v: -(v["bbox"][3] - v["bbox"][1]))
            # a view much smaller than the best one at this instant is a worse look at the
            # same number, so it is skipped
            hbest = vs[0]["bbox"][3] - vs[0]["bbox"][1]
            vs = [v for v in vs
                  if (v["bbox"][3] - v["bbox"][1]) >= args.rel_h * hbest][:args.max_views]
            inst: collections.Counter = collections.Counter()
            seen_by: collections.defaultdict = collections.defaultdict(set)
            at_xy = pos_at.get((tid, t))
            got = False
            for v in vs:
                if used >= args.read_budget:
                    break
                key = (v["cam"], t)
                img = cache.get(key)
                if img is None:
                    img = cv2.imread(str(raw / v["cam"] / f"t{t:06d}.jpg"))
                    if img is None:
                        continue
                    if len(cache) > 24:
                        cache.clear()
                    cache[key] = img
                crop = torso_from_keypoints(img, v["bbox"], v.get("keypoints"), args.kp_thr)
                if crop is None:
                    crop = torso(img, v["bbox"])
                    n_fixed += 1
                else:
                    n_pose += 1
                used += 1
                n_read += 1
                if crop.size:
                    kits.append(kit_stat(crop))
                face = facing_score(v.get("keypoints"), args.kp_thr)
                for num, conf in read_crop(reader, crop, args.upscale_to, args.min_conf):
                    inst[num] += conf
                    seen_by[num].add(v["cam"])
                    reads.append({"t": int(t), "cam": v["cam"], "n": num,
                                  "c": round(conf, 3), "face": face})
                    got = True
            if got:
                n_inst += 1
            # cross-view agreement bonus: cameras that agree at one instant corroborate each other
            for num, s in inst.items():
                k = len(seen_by[num])
                div = (viewpoint_diversity(sorted(seen_by[num]), centres, at_xy)
                       if (k > 1 and centres and at_xy is not None) else 0.0)
                if k > 1:
                    n_multi += 1
                    div_seen.append(div)
                # the bonus only counts as far as the views were independent
                score[num] += s * (1.0 + args.agree_bonus * div)
        top = score.most_common(2)
        number = top[0][0] if top else None
        s1 = top[0][1] if top else 0.0
        s2 = top[1][1] if len(top) > 1 else 0.0
        margin = float((s1 - s2) / s1) if s1 > 0 else 0.0
        out[str(tid)] = {
            "number": number,
            "margin": round(margin, 3),
            "score": round(float(s1), 3),
            "n_reads": len(reads),
            "n_instants": n_inst,
            "n_multiview_agreements": n_multi,
            "mean_viewpoint_diversity": (round(float(np.mean(div_seen)), 3) if div_seen else None),
            "votes": {k: round(float(v), 3) for k, v in score.most_common(6)},
            "reads": reads,
            "kit": kit_stat_summary(kits),
        }
        print(f"  track {tid:>4}: {number or '--':>3}  margin {margin:4.2f}  "
              f"{len(reads):>3} reads over {n_inst:>3} instants, {n_multi} cross-view agreements")
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    tot = n_pose + n_fixed
    print(f"[jersey_mv] {n_read} crops read in {time.time() - t_start:.0f} s -> {args.out}")
    print(f"[jersey_mv] torso RoI from POSE on {n_pose}/{tot} crops "
          f"({100.0 * n_pose / max(tot, 1):.0f} %), fixed band on the rest")
    conf = [k for k, v in out.items() if v["number"] and v["margin"] >= 0.5]
    print(f"[jersey_mv] {len(conf)}/{len(out)} tracks with a number at margin >= 0.5")


def kit_stat_summary(kits):
    if not kits:
        return None
    a = np.asarray(kits, dtype=np.float32)
    return [round(float(x), 3) for x in a.mean(axis=0)]


if __name__ == "__main__":
    main()
