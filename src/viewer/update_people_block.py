"""Updates just the people in scene.json, leaving the rest of the scene exactly as it was.

scene.json also holds the cameras, the arena layers and the tie points. Those were exported
once, from the research repository, and are not rebuilt here. When only a new tracking run has
happened, patching this one block in place leaves everything else byte-identical.

The 3-D bodies are a separate artifact and are not touched here.

The thumbnail arguments must point at the session the positions were fused from. Point them
at another session and the inset shows a different game, with no error on screen.

run_people.py runs this. By hand, from src/viewer/ :
  uv --project .. run python update_people_block.py \
    --people ../../work/people/roster.json --raw4k ../../work/frames \
    --thumb_subdir game --session game --label "game, roster-locked 10 athletes" \
    --pos_field pos_smooth
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent                 # src/viewer/
SRC = HERE.parent                                      # src/
DEFAULT_RAW4K = SRC.parent / "work" / "frames"
DEFAULT_SCENE = HERE / "public" / "scene.json"
DEFAULT_PEOPLE = SRC.parent / "work" / "people" / "roster.json"


def write_frame_thumbs(raw4k: Path, cam: str, timestamps, out_dir: Path, max_size: int,
                       subdir: str = "") -> dict:
    """One small reference thumbnail per timestamp from raw_4k/<cam>/t<ts>.jpg (long side<=max_size).
    Returns {timestamp: relative_url}."""
    if subdir:
        out_dir = out_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_dir = raw4k / cam
    thumbs = {}
    if not cam_dir.is_dir():
        print(f"  [people-thumbs] reference cam folder missing: {cam_dir}")
        return thumbs
    for t in timestamps:
        src = cam_dir / f"t{t:06d}.jpg"
        if not src.exists():
            continue
        dst = out_dir / f"{cam}_t{t:06d}.jpg"
        rel = f"people_thumbs/{subdir + '/' if subdir else ''}{dst.name}"
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            thumbs[t] = rel
            continue
        img = cv2.imread(str(src))
        if img is None:
            continue
        h, w = img.shape[:2]
        ls = max(h, w)
        if ls > max_size:
            s = max_size / ls
            img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        thumbs[t] = rel
    print(f"  [people-thumbs] {len(thumbs)}/{len(timestamps)} {cam} frame thumbnails")
    return thumbs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=str(DEFAULT_SCENE), help="scene.json to patch in place")
    ap.add_argument("--people", default=str(DEFAULT_PEOPLE),
                    help="people_fused/tracked json (timestamps + frames[{id,pos,n_cams}])")
    ap.add_argument("--thumb_cam", default="cam13", help="reference camera for the per-frame inset")
    ap.add_argument("--thumb_size", type=int, default=400, help="thumbnail long-side cap (px)")
    ap.add_argument("--raw4k", default=str(DEFAULT_RAW4K),
                    help="frame cache the inset thumbnails come from - MUST be the session the "
                         "people were fused from")
    ap.add_argument("--thumb_subdir", default="",
                    help="write the thumbnails under people_thumbs/<subdir>/ so sessions with "
                         "overlapping frame numbers do not overwrite each other")
    ap.add_argument("--session", default="", help="short session id recorded in people.meta")
    ap.add_argument("--label", default="", help="human label shown in the People-map header")
    ap.add_argument("--calibration", default="",
                    help="which camera calibration these positions were fused with (provenance "
                         "only; shown in the viewer)")
    ap.add_argument("--fps", type=float, default=25.0, help="source frame rate, for the seconds readout")
    ap.add_argument("--pos_field", default="pos",
                    help="which position to draw. `pos` is the raw fused measurement; "
                         "`pos_smooth` is people/assemble.py's RTS-smoothed posterior mean, which "
                         "is a strictly better estimate of where the athlete was (the smoother "
                         "sees the whole track, forward and backward) and removes the 0.071 m "
                         "frame-to-frame jitter. Recorded in people.meta either way.")
    ap.add_argument("--no_backup", action="store_true", help="skip the scene.json backup copy")
    args = ap.parse_args()

    scene_path = Path(args.scene)
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    rp = json.loads(Path(args.people).read_text(encoding="utf-8"))

    ts = [int(t) for t in rp.get("timestamps", [])]
    frames = {}
    for t in ts:
        out = []
        for p in rp.get("frames", {}).get(str(t), []):
            pos = p.get(args.pos_field) or p["pos"]
            obs = {"id": int(p["id"]), "pos": [float(v) for v in pos],
                   "n_cams": int(p.get("n_cams", 0))}
            # the tracking stages add these; a fuse-only file has neither. Carry them when
            # present so the map can colour by persistent identity, not a per-frame index.
            if p.get("track_id") is not None:
                obs["track_id"] = int(p["track_id"])
            if p.get("spread_m") is not None:
                obs["spread_m"] = round(float(p["spread_m"]), 3)
            # roles.py labels each track player / peripheral / bystander, so the map can tell
            # the sideline staff from the athletes
            if p.get("role"):
                obs["role"] = p["role"]
            # roles.py found this track's kit far from both team clusters, so the person is on
            # neither team. Referees move like players, so only the kit separates them.
            if p.get("off_team"):
                obs["off_team"] = True
            out.append(obs)
        frames[str(t)] = out
    people_obj = {"timestamps": ts, "frames": frames}
    # roster_lock.py names every slot by team and jersey number, so the map can say "A11"
    # instead of "track 0". Absent on runs that did not lock a roster.
    slots = rp.get("slots")
    if isinstance(slots, dict) and slots:
        labels = {str(int(k)): f"{v['team']}{v.get('number', '')}"
                  for k, v in slots.items()
                  if isinstance(v, dict) and v.get("team") is not None}
        if labels:
            people_obj["slots"] = labels
            print(f"  [people] roster labels carried: "
                  f"{', '.join(labels[k] for k in sorted(labels, key=int))}")
    thumbs = write_frame_thumbs(Path(args.raw4k), args.thumb_cam, ts,
                                scene_path.parent / "people_thumbs", args.thumb_size,
                                args.thumb_subdir)
    people_obj["thumb_cam"] = args.thumb_cam
    people_obj["frame_thumbs"] = {str(t): thumbs[t] for t in ts if t in thumbs}
    n_tracked = sum(1 for v in frames.values() for p in v if "track_id" in p)
    n_players = sum(1 for v in frames.values() for p in v if p.get("role") == "player")
    people_obj["meta"] = {
        "pos_field": args.pos_field,
        "session": args.session or None,
        "label": args.label or None,
        "source": Path(args.people).name,
        "calibration": args.calibration or None,
        "fps": args.fps,
        "tracked": n_tracked > 0,
        "n_tracks": len({p["track_id"] for v in frames.values() for p in v if "track_id" in p}) or None,
        "roles": bool(n_players),
        "baked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }

    old = scene.get("people", {})
    old_n = len(old.get("timestamps", []))
    old_sess = (old.get("meta") or {}).get("session")
    if not args.no_backup and old_n and old_sess != args.session:
        # Switching sessions overwrites a block that cannot be rebuilt from scene.json alone,
        # so keep one copy next to the scene.
        bak = scene_path.with_suffix(f".pre_{args.session or 'people'}.bak.json")
        if not bak.exists():
            shutil.copy2(scene_path, bak)
            print(f"[update_people_block] backed up previous scene to {bak.name}")

    scene["people"] = people_obj
    scene_path.write_text(json.dumps(scene), encoding="utf-8")

    n_inst = sum(len(v) for v in frames.values())
    untouched = [k for k in scene if k != "people"]
    print(f"[update_people_block] {Path(args.people).name}: {old_n} -> {len(ts)} timestamps, "
          f"{n_inst} person-instances, {people_obj['meta']['n_tracks']} tracks")
    print(f"[update_people_block] patched {scene_path}  (other blocks untouched: {untouched})")


if __name__ == "__main__":
    main()
