"""Finds the people in each frame and works out where on the court they are standing.

Feet are on the floor, so one calibrated camera is enough: the ray through the foot pixel
meets the plane z = 0 at one point, in metres. The foot pixel is the midpoint of the two
ankles from a keypoint detector, a single confident ankle if only one is seen, or the bottom
of the bounding box as a last resort (marked "box", because it is worse). fuse_global.py
combines the cameras afterwards.
see: https://en.wikipedia.org/wiki/Line-plane_intersection Each detection also keeps its box, its 17 COCO keypoints,
a ReID appearance vector and a saved crop for the later stages.

Normally run by run_people.py. Standalone, from src/ (the frames must
already be in work/frames/<cam>/tNNNNNN.jpg, or the videos in input_videos/):
    uv run python people/detect.py --timestamps 0,1,2 --cams perimeter --stereo_cams ""

Writes work/people/foot_detections.json, which looks like this:

    {timestamp: [detection, detection, ...]}

    and each detection has these fields:
      cam, person          the camera, and the number of the person within that frame
      foot_uv              [u, v], the foot pixel in the full-size frame
      foot_src             where the foot pixel came from: "ankle" (both ankles), "ankle1" (one) or "box"
      world                [X, Y, 0], the foot on the court, in metres
      cam_xy               [x, y], where the camera stands on the court, in metres
      elev_deg             how steeply the camera looks down at this foot, in degrees
      det_score, kp_score  confidence of the person detection and of the keypoints, 0 to 1
      bbox                 [x0, y0, x1, y1], the person box in pixels
      keypoints            the 17 COCO keypoints as [u, v, score], in full-size pixels
                           (the COCO order: https://cocodataset.org/#keypoints-2020)
      appearance           the ReID vector (unit length), with appearance_backend and appearance_dim
      crop_ref             path of the saved crop: crops/detect/t??????/<cam>_p<i>.jpg

--no_appearance and --no_crops give a purely geometric run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# the helpers live one directory up, and this has to work from wherever it is run
ROOT = Path(__file__).resolve().parent          # src/people/
SRC = ROOT.parent                                # src/
MT = SRC.parent                                  # repository root
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from camera_frames import _placement_to_cam_from_world, extract_frames, _parse_stereo_cams
from _floor_homography_sanity import full_opencv_to_KD, pixel_to_floor, warn_if_folding

PLACER = MT / "placements" / "placed_cameras_redfox_repaired.json"
PNP = MT / "diagnostics" / "court_pnp_cameras_redfox_repaired.json"
VIDEOS = MT / "input_videos"                     # the synchronised recordings out<N>.mp4
RAW_DIR = MT / "work" / "frames"
OUT_DIR = MT / "work" / "people"
CROPS_DIR = ROOT / "crops"               # person crops (git-ignored), foldered crops/detect/t??????/

# which class is a person, and which keypoints are the ankles
PERSON_LABEL = 1
L_ANKLE, R_ANKLE = 15, 16
# grouped by body part, for the sanity test below
UPPER_KP = (0, 1, 2, 3, 4, 5, 6)        # nose, eyes, ears, shoulders
LOWER_KP = (11, 12, 13, 14, 15, 16)     # hips, knees, ankles


def _valid_person(kp_full: np.ndarray, kp_thr: float, min_kp: int) -> bool:
    """Is this a whole person, or a fragment (a pair of legs, a head and shoulders, an empty box)?

    Needs at least `min_kp` confident joints, with at least one above the waist and one below.
    """
    conf = kp_full[:, 2] >= kp_thr
    if int(conf.sum()) < min_kp:
        return False
    return bool(conf[list(UPPER_KP)].any()) and bool(conf[list(LOWER_KP)].any())

# the perimeter cameras, which see people upright and at full resolution
PERIMETER_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]
# Ceiling cameras whose calibration round-trips through the floor correctly. They are extra
# evidence only, never trusted alone. The other three ceiling cameras are left out entirely.
CEILING_TRUST_IDS = [209, 110, 210]  # cam09R, cam10L, cam10R


def id_to_name(cam_id: int) -> str:
    if cam_id < 100:
        return f"cam{cam_id:02d}"
    if cam_id < 200:
        return f"cam{cam_id - 100:02d}L"
    return f"cam{cam_id - 200:02d}R"


def load_detector(device: str):
    from torchvision.models.detection import (
        keypointrcnn_resnet50_fpn,
        KeypointRCNN_ResNet50_FPN_Weights,
    )
    weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
    # see: https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html
    model = keypointrcnn_resnet50_fpn(weights=weights)
    model.eval().to(device)
    return model


@torch.no_grad()
def detect_people(model, img_bgr: np.ndarray, device: str, scale: float,
                  score_thr: float, kp_thr: float, min_valid_kp: int = 4):
    """Run the keypoint detector on one frame and return one dict per person, at full resolution:
      {foot_uv:[u,v], foot_src, det_score, kp_score, bbox:[x0,y0,x1,y1],
       keypoints:(17,3) array of [u,v,score]}.
    The bbox is kept for the appearance crop and the keypoints for the body fitting."""
    H, W = img_bgr.shape[:2]
    if scale != 1.0:
        small = cv2.resize(img_bgr, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = img_bgr
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    ten = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).to(device)
    out = model([ten])[0]
    boxes = out["boxes"].cpu().numpy()
    labels = out["labels"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    kps = out["keypoints"].cpu().numpy()           # (N,17,3) x,y,vis
    kps_sc = out["keypoints_scores"].cpu().numpy()  # (N,17)
    inv = 1.0 / scale

    people = []
    for i in range(len(boxes)):
        if labels[i] != PERSON_LABEL or scores[i] < score_thr:
            continue
        la_s, ra_s = kps_sc[i, L_ANKLE], kps_sc[i, R_ANKLE]
        # both ankles if we have them, one if not, and the bottom of the box only as a last
        # resort, which later stages know to trust less
        if la_s >= kp_thr and ra_s >= kp_thr:
            foot = 0.5 * (kps[i, L_ANKLE, :2] + kps[i, R_ANKLE, :2])
            src, kp_score = "ankle", float(min(la_s, ra_s))
        elif max(la_s, ra_s) >= kp_thr:
            j = L_ANKLE if la_s >= ra_s else R_ANKLE
            foot = kps[i, j, :2]
            src, kp_score = "ankle1", float(max(la_s, ra_s))
        else:
            x0, y0, x1, y1 = boxes[i]
            foot = np.array([0.5 * (x0 + x1), y1])
            src, kp_score = "box", 0.0
        # Scale back to the original frame. The third column is replaced by the per-joint
        # score, because the detector only puts a constant visibility flag there.
        kp_full = kps[i].astype(np.float64).copy()
        kp_full[:, :2] *= inv
        kp_full[:, 2] = kps_sc[i]
        if not _valid_person(kp_full, kp_thr, min_valid_kp):
            continue                       # drop fragments / jointless box-fallback blobs
        bbox = boxes[i].astype(np.float64) * inv
        people.append({
            "foot_uv": foot * inv, "foot_src": src,
            "det_score": float(scores[i]), "kp_score": kp_score,
            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            "keypoints": kp_full,
        })
    return people


def _crop_bbox(img_bgr: np.ndarray, bbox):
    """Cut a person out of the frame. Boxes do run off the edge, hence the clipping."""
    H, W = img_bgr.shape[:2]
    x0 = max(0, int(bbox[0])); y0 = max(0, int(bbox[1]))
    x1 = min(W, int(round(bbox[2]))); y1 = min(H, int(round(bbox[3])))
    if x1 - x0 < 4 or y1 - y0 < 8:
        return None
    return img_bgr[y0:y1, x0:x1]


def reembed(inp: str, out: str, raw4k: str, backend: str, batch: int, min_crop_px: int,
            device: str) -> None:
    """Recompute only the appearance embeddings of an existing detection file with another backend.

    Boxes, keypoints and foot points are copied unchanged, so only the `appearance` vectors
    differ. Each (camera, timestamp) frame is decoded once and every crop is cut from it.
    """
    from people.reid import build_embedder

    rows_by_frame: dict[tuple[str, int], list] = {}
    dets = json.loads(Path(inp).read_text(encoding="utf-8"))
    n_rows = 0
    for t, rows in dets.items():
        for i, r in enumerate(rows):
            rows_by_frame.setdefault((r["cam"], int(t)), []).append((i, r))
            n_rows += 1
    embedder = build_embedder(backend, device=device)
    fallback = embedder if embedder.name.startswith("hsv") else build_embedder("hsv")
    print(f"[reembed] {Path(inp).name}: {n_rows} detections over {len(rows_by_frame)} "
          f"(camera, timestamp) frames -> {embedder.name} (dim {embedder.dim})")

    raw = Path(raw4k)
    done = n_missing = n_small = 0
    for k, (cam, t) in enumerate(sorted(rows_by_frame)):
        f = raw / cam / f"t{t:06d}.jpg"
        img = cv2.imread(str(f)) if f.exists() else None
        if img is None:
            n_missing += len(rows_by_frame[(cam, t)])
            continue
        idxs, crops, small = [], [], []
        for i, r in rows_by_frame[(cam, t)]:
            c = _crop_bbox(img, r["bbox"]) if r.get("bbox") else None
            if c is None:
                n_missing += 1
                continue
            if min(c.shape[:2]) < min_crop_px:
                small.append((i, c))
            else:
                idxs.append(i)
                crops.append(c)
        rows = dets[str(t)]
        for a in range(0, len(crops), batch):
            vecs = embedder.embed_batch(crops[a:a + batch])
            for j, v in enumerate(vecs):
                r = rows[idxs[a + j]]
                r["appearance"] = [round(float(x), 4) for x in v]
                r["appearance_backend"] = embedder.name
                r["appearance_dim"] = int(embedder.dim)
                done += 1
        # Tiny crops take the same colour-histogram fallback the original pass used, so the
        # two files agree about which detections have a real embedding and which do not.
        for i, c in small:
            v = fallback.embed_batch([c])[0]
            r = rows[i]
            r["appearance"] = [round(float(x), 4) for x in v]
            r["appearance_backend"] = fallback.name
            r["appearance_dim"] = int(fallback.dim)
            n_small += 1
        if k % 250 == 0:
            print(f"  [{k + 1}/{len(rows_by_frame)}] {cam} t{t:06d}  {done} embedded")
    Path(out).write_text(json.dumps(dets), encoding="utf-8")
    print(f"[reembed] wrote {out}: {done} re-embedded, {n_small} via the HSV fallback, "
          f"{n_missing} without a usable crop")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamps", default="7500,15000,30000",
                    help="comma-separated raw video frame indices (25 fps; t==frame)")
    ap.add_argument("--cams", default="perimeter",
                    choices=["perimeter", "perimeter+ceiling", "all"],
                    help="which cameras to detect on (ceiling top-down ankles are unreliable)")
    ap.add_argument("--scale", type=float, default=1.0, help="detector input downscale (1.0=full 4K)")
    ap.add_argument("--score_thr", type=float, default=0.70, help="person detection score gate")
    ap.add_argument("--kp_thr", type=float, default=3.0, help="ankle keypoint score gate")
    ap.add_argument("--min_valid_kp", type=int, default=4,
                    help="min confident keypoints (with upper+lower body span) to accept a person; "
                         "rejects half-body fragments and jointless box blobs")
    ap.add_argument("--elev_min", type=float, default=10.0,
                    help="reject foot rays below this elevation (deg). 10deg keeps real athletes "
                         "at the far end of the court (12-14deg); fuse.py down-weights low rays by "
                         "sin(elev) so their noisier floor position is trusted less, but the view "
                         "still feeds multi-view triangulation / SMPL-X")
    ap.add_argument("--stereo_cams", default="9,10,11")
    ap.add_argument("--out", default=str(OUT_DIR / "foot_detections.json"))
    # the extra evidence, on by default
    ap.add_argument("--no_appearance", action="store_true",
                    help="skip the ReID appearance embedding (pure-geometry detections)")
    ap.add_argument("--no_crops", action="store_true",
                    help="don't save person crops to disk")
    ap.add_argument("--backend", default="auto",
                    help="ReID backend for appearance: auto (osnet->cnn->hsv), osnet, cnn, hsv, "
                         "one of the timm aliases (dinov2, dinov2b, clip, siglip), or an explicit "
                         "timm:<model_id>[@<side>]")
    ap.add_argument("--min_crop_px", type=int, default=24,
                    help="crops whose shorter side is below this use the HSV fallback descriptor")
    ap.add_argument("--reembed", action="store_true",
                    help="do NOT detect: re-encode the appearance of an existing --in detection "
                         "file with --backend, leaving boxes and keypoints byte-identical "
                         "")
    ap.add_argument("--in", dest="inp", default="",
                    help="input foot_detections*.json for --reembed")
    ap.add_argument("--raw4k", default=str(RAW_DIR),
                    help="frame directory the crops are cut from, for --reembed")
    ap.add_argument("--batch", type=int, default=64, help="embedding batch size")
    args = ap.parse_args()

    if args.reembed:
        if not args.inp:
            raise SystemExit("--reembed needs --in <foot_detections.json>")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        reembed(args.inp, args.out, args.raw4k, args.backend, args.batch,
                args.min_crop_px, dev)
        return

    timestamps = [int(t) for t in args.timestamps.split(",") if t.strip()]
    stereo = _parse_stereo_cams(args.stereo_cams)
    placer = json.loads(PLACER.read_text(encoding="utf-8"))
    pnp = json.loads(PNP.read_text(encoding="utf-8"))

    if args.cams == "perimeter":
        want_ids = set(PERIMETER_IDS)
    elif args.cams == "perimeter+ceiling":
        want_ids = set(PERIMETER_IDS) | set(CEILING_TRUST_IDS)
    else:
        want_ids = None  # all detected eyes

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[people_detect] device={device}  timestamps={timestamps}  cams={args.cams}  scale={args.scale}")
    print("[people_detect] extracting frames (reuses camera_frames.extract_frames)...")
    frames = extract_frames(VIDEOS, timestamps, RAW_DIR, stereo)

    model = load_detector(device)
    print("[people_detect] detector loaded; running...")

    # the real embedder, plus a colour-histogram fallback for crops too small to be worth
    # putting through it
    save_crops = not args.no_crops
    embedder = fallback = None
    if not args.no_appearance:
        from people.reid import build_embedder
        embedder = build_embedder(args.backend, device=device)
        fallback = embedder if embedder.name.startswith("hsv") else build_embedder("hsv")
    if save_crops:
        print(f"[people_detect] saving person crops under {CROPS_DIR / 'detect'}")

    result: dict[str, list] = {}
    for t in timestamps:
        rows = []
        for cam_id, path in sorted(frames[t].items()):
            name = id_to_name(cam_id)
            if want_ids is not None and cam_id not in want_ids:
                continue
            if name not in placer or name not in pnp:
                continue
            img = cv2.imread(str(path))
            if img is None:
                continue
            R_w2c, t_w2c = _placement_to_cam_from_world(placer[name])
            K, dist = full_opencv_to_KD(pnp[name]["params"])
            # Warn once if this camera's distortion folds inside the frame. A detection in the
            # folded band comes back marked invalid and is dropped below.
            warn_if_folding(name, K, dist, img.shape[1], img.shape[0])
            C = (-R_w2c.T @ t_w2c)  # camera center (world) == placer position
            dets = detect_people(model, img, device, args.scale, args.score_thr, args.kp_thr,
                                 args.min_valid_kp)
            for pi, d in enumerate(dets):
                uv = d["foot_uv"]
                P, front = pixel_to_floor(uv.reshape(1, 2), R_w2c, t_w2c, K, dist)
                if not bool(front[0]):
                    continue
                p = P[0]
                horiz = float(np.hypot(C[0] - p[0], C[1] - p[1]))
                elev = float(np.degrees(np.arctan2(C[2], max(horiz, 1e-6))))
                if elev < args.elev_min:
                    continue
                rec = {
                    "cam": name, "person": pi,
                    "foot_uv": [float(uv[0]), float(uv[1])], "foot_src": d["foot_src"],
                    "world": [float(p[0]), float(p[1]), 0.0],
                    "cam_xy": [float(C[0]), float(C[1])],
                    "elev_deg": round(elev, 2),
                    "det_score": round(d["det_score"], 3), "kp_score": round(d["kp_score"], 2),
                    # extra evidence for later stages
                    "bbox": [round(v, 1) for v in d["bbox"]],
                    "keypoints": [[round(float(ku), 1), round(float(kv), 1), round(float(ks), 2)]
                                  for ku, kv, ks in d["keypoints"]],
                }
                # appearance and crop, from the detector's own box at full resolution
                if save_crops or embedder is not None:
                    crop = _crop_bbox(img, d["bbox"])
                    if crop is not None:
                        if save_crops:
                            rel = f"detect/t{t:06d}/{name}_p{pi}.jpg"
                            cp = CROPS_DIR / rel
                            cp.parent.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(cp), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            rec["crop_ref"] = f"crops/{rel}"
                        if embedder is not None:
                            use_fb = min(crop.shape[0], crop.shape[1]) < args.min_crop_px
                            e = fallback if use_fb else embedder
                            rec["appearance"] = [round(float(x), 4) for x in e.embed(crop)]
                            rec["appearance_backend"] = e.name
                            rec["appearance_dim"] = int(e.dim)
                rows.append(rec)
        result[str(t)] = rows
        print(f"  t{t:06d}: {len(rows)} foot detections across cameras")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[people_detect] wrote {out_path}")


if __name__ == "__main__":
    main()
