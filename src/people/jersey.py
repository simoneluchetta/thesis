"""Reads the shirt number off each track, one camera crop at a time.

A shirt number is the same in every camera at every instant, so it is a fixed anchor for
the identity. No single read is trusted: the number of a track is a confidence-weighted vote
over many crops. The crops are bucketed in time and the biggest of each bucket is read in
turn, so every part of the track gets a say and a track holding two people shows a split
vote instead of one confident number. Only the torso band is read (from about a sixth to
just over half of the box height), because the whole person gives the OCR court lines and
the number on the shorts. The margin of the winner over the runner-up is written out too.

The pipeline uses the multi-camera reader in jersey_mv.py, which imports kit_stat and torso
from here. Standalone, from src/:

  uv run python -m people.jersey \\
      --tracked ../work/people/assembled.json \\
      --raw4k   ../work/frames \\
      --out     ../work/people/jersey_numbers.json

Throughput on the CPU is about 6 crops/s, so 120 crops per track on a 30-track clip takes
roughly ten minutes.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import cv2
import numpy as np


def torso(img: np.ndarray, bbox, top=0.16, bottom=0.55, pad=0.05) -> np.ndarray:
    """The band of the box where the number is: from `top` to `bottom` of the box height, padded sideways."""
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    h, w = y1 - y0, x1 - x0
    yA, yB = y0 + int(top * h), y0 + int(bottom * h)
    x0 = max(0, x0 - int(pad * w))
    x1 = min(img.shape[1], x1 + int(pad * w))
    yA, yB = max(0, yA), min(img.shape[0], yB)
    if yB <= yA or x1 <= x0:
        return np.zeros((0, 0, 3), np.uint8)
    return img[yA:yB, x0:x1]


def kit_stat(crop: np.ndarray) -> tuple:
    """Median brightness (V) and saturation (S) of the torso crop, in HSV.

    A rough kit colour. It is noisy, because the crop also contains arms, background and
    other players, so the number read stays the reliable identity signal.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(np.median(hsv[:, :, 2])), float(np.median(hsv[:, :, 1]))


def collect(tracked: dict, roles_filter: set | None, min_h: float, n_buckets: int = 12):
    """Choose which crops to read for each track, covering its whole lifetime.

    Returns {track_id: [(box height, t, cam, bbox)]} in reading order.
    """
    per = collections.defaultdict(list)
    for t in tracked["timestamps"]:
        for p in tracked["frames"][str(t)]:
            tid = p.get("track_id", p.get("id"))
            if roles_filter is not None and p.get("role") not in roles_filter:
                continue
            for v in p.get("views", []):
                b = v.get("bbox")
                if not b:
                    continue
                h = b[3] - b[1]
                if h >= min_h:
                    per[tid].append((h, int(t), v["cam"], b))
    # bucket by time, then take the biggest crop from each bucket in turn, so a big crop is
    # still read first but every part of the track gets a say
    out = {}
    for tid, rows in per.items():
        if not rows:
            continue
        t0 = min(r[1] for r in rows)
        t1 = max(r[1] for r in rows)
        span = max(t1 - t0, 1)
        buckets = collections.defaultdict(list)
        for r in rows:
            k = min(int((r[1] - t0) / span * n_buckets), n_buckets - 1)
            buckets[k].append(r)
        for k in buckets:
            buckets[k].sort(key=lambda r: -r[0])
        order, i = [], 0
        keys = sorted(buckets)
        while any(len(buckets[k]) > i for k in keys):
            for k in keys:
                if len(buckets[k]) > i:
                    order.append(buckets[k][i])
            i += 1
        out[tid] = order
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracked", required=True)
    ap.add_argument("--raw4k", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_track", type=int, default=60,
                    help="crops to read per track, largest first (the budget knob)")
    ap.add_argument("--min_h", type=float, default=150.0, help="skip boxes shorter than this (px)")
    ap.add_argument("--time_buckets", type=int, default=12,
                    help="temporal buckets the reading budget is spread over, so the vote cannot "
                         "be decided by one well-lit stretch of the track")
    ap.add_argument("--min_conf", type=float, default=0.30, help="discard reads below this confidence")
    ap.add_argument("--roles", default="player",
                    help="comma-separated roles to read, or 'all'. Officials have no number.")
    ap.add_argument("--gpu", action="store_true", help="run the recogniser on the GPU")
    ap.add_argument("--upscale_to", type=int, default=160, help="torso band is resized to this height")
    ap.add_argument("--no_ocr", action="store_true",
                    help="skip number reading and only compute the kit split. Needs no OCR "
                         "backend at all, and the kit-mix flag alone already proves an identity "
                         "error whenever one track spans both kits.")
    args = ap.parse_args()

    reader = None
    if not args.no_ocr:
        import easyocr
        # see: https://github.com/JaidedAI/EasyOCR
        reader = easyocr.Reader(["en"], gpu=args.gpu, verbose=False)

    tracked = json.loads(Path(args.tracked).read_text(encoding="utf-8"))
    roles_filter = None if args.roles == "all" else set(args.roles.split(","))
    if roles_filter is not None and not any(
            "role" in p for t in tracked["timestamps"][:1] for p in tracked["frames"][str(t)]):
        print("[jersey] no `role` field on the tracks - reading ALL of them "
              "(run people/roles.py first to skip the officials)")
        roles_filter = None

    raw = Path(args.raw4k)
    per = collect(tracked, roles_filter, args.min_h, args.time_buckets)
    print(f"[jersey] {len(per)} tracks with readable views; "
          f"budget {args.per_track} crops each")

    out, t0, n_read = {}, time.time(), 0
    frame_cache: dict[tuple, np.ndarray] = {}
    for tid in sorted(per, key=lambda k: -len(per[k])):
        votes: collections.Counter = collections.Counter()
        kits: list = []
        reads: list = []          # every individual read, with its timestamp
        n_try = n_hit = 0
        for h, t, cam, b in per[tid][:args.per_track]:
            key = (cam, t)
            img = frame_cache.get(key)
            if img is None:
                img = cv2.imread(str(raw / cam / f"t{t:06d}.jpg"))
                if img is None:
                    continue
                if len(frame_cache) > 24:            # a few frames only; these are 4K
                    frame_cache.clear()
                frame_cache[key] = img
            crop = torso(img, b)
            if crop.size == 0 or crop.shape[0] < 12:
                continue
            s = max(1.0, args.upscale_to / crop.shape[0])
            crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)),
                              interpolation=cv2.INTER_CUBIC)
            n_try += 1
            kits.append(kit_stat(crop))
            if reader is None:
                continue
            try:
                res = reader.readtext(crop, allowlist="0123456789", detail=1,
                                      text_threshold=0.5, low_text=0.3)
            except Exception:
                continue
            best = [(r[2], r[1].strip()) for r in res
                    if r[1].strip().isdigit() and 1 <= len(r[1].strip()) <= 2]
            if not best:
                continue
            best.sort(reverse=True)
            conf, txt = best[0]
            if conf >= args.min_conf:
                votes[txt] += conf          # confidence-weighted, so a confident read counts more
                n_hit += 1
                # keep each read with its time: the reads in time order show where a track
                # stops being one person, which is what jersey_repair.py needs
                reads.append({"t": int(t), "cam": cam, "n": txt,
                              "c": round(float(conf), 3), "h": round(float(h), 1)})
        n_read += n_try
        ranked = votes.most_common()
        top = ranked[0] if ranked else None
        second = ranked[1] if len(ranked) > 1 else None
        # margin of the winner over the runner-up; a narrow margin means the number is
        # unreadable or the track holds two people
        total = sum(votes.values())
        margin = ((top[1] - (second[1] if second else 0.0)) / total) if total else 0.0
        out[str(tid)] = {
            "kit_v": [round(k[1], 1) for k in kits],
            "number": (top[0] if top else None),
            "weight": round(top[1], 2) if top else 0.0,
            "margin": round(margin, 3),
            "n_crops_tried": n_try, "n_reads": n_hit,
            "votes": {k: round(v, 2) for k, v in ranked[:6]},
            "reads": sorted(reads, key=lambda r: r["t"]),
        }
        if top:
            print(f"  track {tid:3d}: #{top[0]:<3} margin {margin:4.2f}  "
                  f"({n_hit}/{n_try} crops read)  votes {dict(list(out[str(tid)]['votes'].items())[:4])}")
        else:
            print(f"  track {tid:3d}: no number  (0/{n_try} crops read)")

    # the two kits, learned from the colours rather than assumed
    allv = np.array([v for r in out.values() for v in r["kit_v"]])
    if len(allv) > 20:
        lo, hi = np.percentile(allv, 15), np.percentile(allv, 85)      # 2-means seeded at the tails
        for _ in range(30):
            m = 0.5 * (lo + hi)
            a, b = allv[allv < m], allv[allv >= m]
            nlo = a.mean() if len(a) else lo
            nhi = b.mean() if len(b) else hi
            if abs(nlo - lo) < 1e-3 and abs(nhi - hi) < 1e-3:
                break
            lo, hi = nlo, nhi
        thr = 0.5 * (lo + hi)
        print("")
        print(f"[jersey] kit split at saturation S={thr:.0f} (unsaturated/white ~{lo:.0f}, saturated/navy ~{hi:.0f})")
        mixed = []
        for tid, r in out.items():
            v = np.array(r["kit_v"])
            if len(v) < 6:
                r["kit"], r["kit_mix"] = None, None
                continue
            frac_light = float((v >= thr).mean())
            r["kit"] = "navy" if frac_light > 0.5 else "white"
            r["kit_mix"] = round(min(frac_light, 1 - frac_light), 3)
            if r["kit_mix"] > 0.15:
                mixed.append((r["kit_mix"], tid, len(v)))
        for r in out.values():          # drop the raw list from the file
            r.pop("kit_v", None)
        mixed.sort(reverse=True)
        print(f"[jersey] kit mix: {len(mixed)}/{len(out)} tracks over the 15 % flag, median mix "
              f"{100*float(np.median([m for m, _, _ in mixed])) if mixed else 0:.0f} %. "
              f"The kit colour is noisy (see kit_stat), so this is a readout, not a warning.")
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    named = sum(1 for v in out.values() if v["number"])
    confident = sum(1 for v in out.values() if v["number"] and v["margin"] >= 0.5)
    print(f"\n[jersey] {named}/{len(out)} tracks got a number, {confident} with margin >= 0.5")
    print(f"[jersey] {n_read} crops read in {time.time()-t0:.0f} s -> {args.out}")

    # two tracks alive at the same instant cannot be the same person, so if they carry the
    # same number one of them is wrong; report it
    live = collections.defaultdict(set)
    for t in tracked["timestamps"]:
        for p in tracked["frames"][str(t)]:
            live[int(t)].add(p.get("track_id", p.get("id")))
    clash = collections.Counter()
    for t, ids in live.items():
        seen = collections.defaultdict(list)
        for i in ids:
            n = out.get(str(i), {}).get("number")
            if n:
                seen[n].append(i)
        for n, who in seen.items():
            if len(who) > 1:
                clash[(n, tuple(sorted(who)))] += 1
    if clash:
        print("[jersey] CONFLICTS - same number on tracks that coexist (instants):")
        for (n, who), c in clash.most_common(10):
            print(f"    #{n} on tracks {who}: {c} instants")
    else:
        print("[jersey] no two coexisting tracks share a number")


if __name__ == "__main__":
    main()
