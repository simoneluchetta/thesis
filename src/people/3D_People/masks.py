"""Cuts each person's silhouette out of the frames, for the silhouette term of the body fit.

Mask R-CNN (torchvision) runs on each full frame in which a trusted camera saw a tracked
person. The instance whose box overlaps the tracked box best (IoU >= iou_thr) is taken as
that person. What is stored is not the mask but a clamped distance transform of it: for every
pixel in a region around the box, how far it is from the mask in full-resolution pixels (0
inside). The fit projects its mesh vertices and reads the distance there, so it needs no
renderer.

Reads work/frames/<cam>/tNNNNNN.jpg and the tracked people. Output (default work/bodies/):
  masks_dense.npz   - one float16 array per "track|t|cam" key: clamped outside-mask DT, values in
                      FULL-RES pixels (the array may be downsampled; values stay full-res px).
  masks_dense.json  - manifest: per-key ROI [x0,y0,w,h] (full-frame px) + global params.

Run from src/:
  uv run python people/3D_People/masks.py --selftest        # 2 timestamps, sanity
  uv run python people/3D_People/masks.py                    # all timestamps
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent
for _p in (str(SRC), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adapter import DEFAULT_TRACKED, iter_person_frames, load_cameras

RAW_DIR = SRC.parent / "work" / "frames"
OUT_DIR = SRC.parent / "work" / "bodies"
COCO_PERSON_LABEL = 1


def _iou(a, b) -> float:
    """IoU of two [x0,y0,x1,y1] boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _frame_path(cam: str, t: int) -> Path:
    return RAW_DIR / cam / f"t{t:06d}.jpg"


def _load_maskrcnn(device):
    from torchvision.models.detection import (maskrcnn_resnet50_fpn,
                                              MaskRCNN_ResNet50_FPN_Weights)
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    # see: https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html
    model = maskrcnn_resnet50_fpn(weights=weights).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _person_instances(model, frame_bgr, device, score_thr: float):
    """Run Mask R-CNN once on a full frame -> list of {box, mask(bool HxW), score} for persons."""
    import torch
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    ten = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        out = model([ten])[0]
    boxes = out["boxes"].cpu().numpy()
    labels = out["labels"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    masks = out["masks"].cpu().numpy()[:, 0]            # (N,H,W) soft
    insts = []
    for i in range(len(boxes)):
        if labels[i] == COCO_PERSON_LABEL and scores[i] >= score_thr:
            insts.append({"box": boxes[i].tolist(),
                          "mask": masks[i] >= 0.5,
                          "score": float(scores[i])})
    return insts


def _outside_dt(mask_bool, roi, clamp_px, max_side):
    """Clamped outside-mask distance transform over the ROI, optionally downsampled.
    Returns (dt float16 array, used_roi). Values are FULL-RES px regardless of array size."""
    from scipy.ndimage import distance_transform_edt
    x0, y0, x1, y1 = roi
    sub = mask_bool[y0:y1, x0:x1]
    if not sub.any():                                   # no person pixels in ROI -> skip
        return None
    # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html
    dt = distance_transform_edt(~sub).astype(np.float32)   # 0 inside mask, dist outside
    np.clip(dt, 0.0, clamp_px, out=dt)
    h, w = dt.shape
    if max(h, w) > max_side:                            # downsample (values stay full-res px)
        s = max_side / max(h, w)
        dt = cv2.resize(dt, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                        interpolation=cv2.INTER_AREA)
    return dt.astype(np.float16)


def run(tracked_path, out_dir, device, score_thr, iou_thr, roi_margin, clamp_px, max_side,
        ts_limit=None):
    cameras = load_cameras()
    tracked = json.loads(Path(tracked_path).read_text(encoding="utf-8"))
    model = _load_maskrcnn(device)

    # collect needed (cam,t) -> list of (track_id, bbox); trusted cams only
    need = {}
    pf_list = list(iter_person_frames(tracked, cameras))
    ts_seen = sorted({pf.t for pf in pf_list})
    if ts_limit:
        keep = set(ts_seen[:ts_limit])
        pf_list = [pf for pf in pf_list if pf.t in keep]
    for pf in pf_list:
        for v in pf.views:
            if not v.camera.trusted or v.bbox is None:
                continue
            need.setdefault((v.cam, pf.t), []).append((int(pf.track_id), list(v.bbox)))

    arrays, manifest = {}, {}
    stats = {"keys": 0, "no_frame": 0, "no_inst": 0, "ious": [], "areas": []}
    t0 = time.time()
    cam_t_list = sorted(need)
    for n, (cam, t) in enumerate(cam_t_list):
        fp = _frame_path(cam, t)
        frame = cv2.imread(str(fp)) if fp.exists() else None
        if frame is None:
            stats["no_frame"] += len(need[(cam, t)]); continue
        H, W = frame.shape[:2]
        insts = _person_instances(model, frame, device, score_thr)
        for track_id, bbox in need[(cam, t)]:
            if not insts:
                stats["no_inst"] += 1; continue
            ious = [_iou(bbox, ins["box"]) for ins in insts]
            j = int(np.argmax(ious))
            if ious[j] < iou_thr:
                stats["no_inst"] += 1; continue
            mask = insts[j]["mask"]
            # expanded ROI around the tracked bbox, clipped to frame
            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            mx, my = roi_margin * bw, roi_margin * bh
            x0 = int(max(0, np.floor(bbox[0] - mx))); y0 = int(max(0, np.floor(bbox[1] - my)))
            x1 = int(min(W, np.ceil(bbox[2] + mx))); y1 = int(min(H, np.ceil(bbox[3] + my)))
            dt = _outside_dt(mask, (x0, y0, x1, y1), clamp_px, max_side)
            if dt is None:
                stats["no_inst"] += 1; continue
            key = f"{track_id}|{t}|{cam}"
            arrays[key] = dt
            manifest[key] = {"roi": [x0, y0, x1 - x0, y1 - y0]}
            stats["keys"] += 1
            stats["ious"].append(ious[j])
            stats["areas"].append(float(mask.sum()) / (W * H))
        if (n + 1) % 20 == 0:
            print(f"  [{n+1}/{len(cam_t_list)}] {time.time()-t0:.0f}s  keys={stats['keys']}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "masks_dense.npz", **arrays)
    (out_dir / "masks_dense.json").write_text(json.dumps({
        "version": "1", "clamp_px": clamp_px, "roi_margin": roi_margin,
        "score_thr": score_thr, "iou_thr": iou_thr, "max_side": max_side,
        "keys": manifest,
    }), encoding="utf-8")
    return stats


def _report(stats):
    iou = np.asarray(stats["ious"]); area = np.asarray(stats["areas"])
    print(f"[masks] stored {stats['keys']} (track,t,cam) silhouette DTs")
    print(f"[masks] no source frame: {stats['no_frame']}   no matching instance: {stats['no_inst']}")
    if iou.size:
        print(f"[masks] mask-vs-bbox IoU: median {np.median(iou):.2f}  p10 {np.percentile(iou,10):.2f}")
        print(f"[masks] person mask area (frac of frame): median {np.median(area):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracked", default=str(DEFAULT_TRACKED))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--score_thr", type=float, default=0.7, help="min Mask R-CNN person score")
    ap.add_argument("--iou_thr", type=float, default=0.20, help="min IoU(view bbox, instance) to accept")
    ap.add_argument("--roi_margin", type=float, default=0.25, help="bbox expansion fraction for the ROI")
    ap.add_argument("--clamp_px", type=float, default=48.0, help="max outside-mask distance (full-res px)")
    ap.add_argument("--max_side", type=int, default=192, help="max DT array side (downsample beyond)")
    ap.add_argument("--selftest", action="store_true", help="first 2 timestamps only")
    args = ap.parse_args()
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"[masks] device {device}; source frames {RAW_DIR}")
    stats = run(args.tracked, args.out_dir, device, args.score_thr, args.iou_thr,
                args.roi_margin, args.clamp_px, args.max_side,
                ts_limit=2 if args.selftest else None)
    _report(stats)
    print(f"[masks] wrote {Path(args.out_dir) / 'masks_dense.npz'} (+ .json)")


if __name__ == "__main__":
    main()
