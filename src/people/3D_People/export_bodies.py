"""Turns the fitted bodies into binary files the viewer can load and play back.

The fit stores SMPL-X parameters, which the browser cannot pose without the body model. So
the meshes are posed here once and written as raw vertex positions. All bodies share one
triangle list, written once; each body instance adds its vertices at a fixed stride, so the
viewer can seek to any frame by arithmetic. Optionally each vertex gets a colour sampled
from the source frames (work/frames/<cam>/tNNNNNN.jpg), gated by the person masks.

The viewer reads the .bin files straight into typed arrays:
https://threejs.org/docs/index.html#api/en/core/BufferAttribute

Reads : the bodies JSON (fit_smplx.py or infill_bodies.py output), the tracked people and
        the cameras.
Writes: viewer/public/bodies.json (manifest) and bodies/verts.bin (float32), faces.bin
        (uint32), colors.bin (uint8 RGB). Coordinates are the world metres of the cameras.

run_bodies.py runs this. Standalone, from src/:
  uv run python people/3D_People/export_bodies.py --bodies ../work/bodies/bodies_full.json \
      --tracked ../work/bodies/tracked_subset.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent
for _p in (str(SRC), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adapter import iter_person_frames, load_cameras
from triangulate import triangulate_person_frame
from fit_smplx import build_smplx, Cfg, load_masks, DEFAULT_MASKS
from _floor_homography_sanity import project_world_to_pixel

DEFAULT_BODIES = SRC.parent / "work" / "bodies" / "bodies_full.json"
DEFAULT_OUT = SRC / "viewer" / "public"
RAW_DIR = SRC.parent / "work" / "frames"   # 4K source frames for texturing

# COCO-17 skeleton edges (for the 3D stick overlay of the raw triangulated joints).
COCO_BONES = [
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],        # shoulders + arms
    [11, 12], [5, 11], [6, 12],                      # torso
    [11, 13], [13, 15], [12, 14], [14, 16],          # legs
    [0, 1], [0, 2], [1, 3], [2, 4],                  # head
    [0, 5], [0, 6],                                  # neck-ish (nose to shoulders)
]


# texturing
def _vertex_normals(V, faces):
    """Outward per-vertex normals (auto-oriented vs the centroid, so winding can't flip them)."""
    n = np.zeros_like(V)
    tri = V[faces]                                            # (F,3,3)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    nl = np.linalg.norm(n, axis=1, keepdims=True); nl[nl == 0] = 1.0
    n = n / nl
    ctr = V.mean(0)
    if (((V - ctr) * n).sum(1) < 0).mean() > 0.5:            # mostly inward -> flip
        n = -n
    return n


def _zbuffer_visible(uv, depth, W, H, infront, cell=3.0, margin=0.05):
    """Per-vertex occlusion test with a coarse z-buffer, without a renderer.

    For each screen cell of `cell` px, the smallest distance of any in-front vertex is stored
    (splatted 3x3 so the sparse points form a closed surface). A vertex is visible when its
    own distance is within `margin` metres of that nearest surface, otherwise it is hidden
    behind another body part. `depth` is the distance to the camera centre in metres.
    """
    gw = int(np.ceil(W / cell)); gh = int(np.ceil(H / cell))
    buf = np.full((gh, gw), np.inf, np.float32)
    inb = infront & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    cx = np.clip((uv[:, 0] / cell).astype(np.int64), 0, gw - 1)
    cy = np.clip((uv[:, 1] / cell).astype(np.int64), 0, gh - 1)
    bx, by, bd = cx[inb], cy[inb], depth[inb]
    for dy in (-1, 0, 1):                                    # 3x3 splat -> watertight surface
        for dx in (-1, 0, 1):
            np.minimum.at(buf, (np.clip(by + dy, 0, gh - 1), np.clip(bx + dx, 0, gw - 1)), bd)
    return inb & (depth <= buf[cy, cx] + margin)


def _mesh_adjacency(faces, nv):
    """Symmetric vertex adjacency (scipy CSR) from the triangle list - for hole-filling."""
    from scipy.sparse import csr_matrix
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
    e = np.concatenate([e, e[:, ::-1]], 0)                   # both directions
    data = np.ones(len(e), np.float32)
    return csr_matrix((data, (e[:, 0], e[:, 1])), shape=(nv, nv))


def _fill_holes(color, seen, A, max_iter=200):
    """Fill the colours of unseen vertices from their seen neighbours across the mesh.

    Repeated neighbour averaging over the edge graph, so unseen regions get the nearest real
    surface colour. `color` (nv,3) float, `seen` (nv,) bool. Returns the filled colours."""
    cur = color.astype(np.float32).copy()
    known = seen.copy()
    cur[~known] = 0.0
    for _ in range(max_iter):
        if known.all():
            break
        num = A @ (cur * known[:, None])                     # (nv,3) sum of known-neighbour colours
        den = A @ known.astype(np.float32)                   # (nv,) count of known neighbours
        newly = (~known) & (den > 0)
        if not newly.any():
            break
        cur[newly] = num[newly] / den[newly, None]
        known[newly] = True
    return cur


def _bilinear_bgr(frame, uv):
    """Bilinearly sample a HxWx3 uint8 frame at (N,2) pixel coords -> (N,3) float BGR."""
    H, W = frame.shape[:2]
    x = np.clip(uv[:, 0], 0, W - 1.001); y = np.clip(uv[:, 1], 0, H - 1.001)
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1; y1 = y0 + 1
    wx = (x - x0)[:, None]; wy = (y - y0)[:, None]
    f = frame.astype(np.float32)
    return (f[y0, x0] * (1 - wx) * (1 - wy) + f[y0, x1] * wx * (1 - wy)
            + f[y1, x0] * (1 - wx) * wy + f[y1, x1] * wx * wy)


def _sample_dt(m, uv):
    """Sample the person's outside-mask DT (px) at full-frame pixels; large where outside the ROI."""
    dt = m["dt"]; x0, y0, W, H = m["roi"]; h, w = dt.shape
    fx = (uv[:, 0] - x0) / max(W, 1); fy = (uv[:, 1] - y0) / max(H, 1)
    val = np.full(len(uv), 999.0, np.float32)
    inside = (fx >= 0) & (fx < 1) & (fy >= 0) & (fy < 1)
    ix = np.clip((fx * w).astype(np.int64), 0, w - 1)
    iy = np.clip((fy * h).astype(np.int64), 0, h - 1)
    val[inside] = dt[iy[inside], ix[inside]]
    return val


def _tracked_origin(tracked_path):
    """Name of the tracking run these bodies came from.

    The subset file is always called `tracked_subset.json`, so run_bodies.py stamps the origin
    into its `source` field. Falls back to the file name when the stamp is absent.
    """
    if not tracked_path:
        return None
    try:
        d = json.loads(Path(tracked_path).read_text(encoding="utf-8"))
        return d.get("source") or Path(tracked_path).name
    except Exception:
        return Path(tracked_path).name


def _slot_labels(tracked_path):
    """{track_id: "A11"} when the tracked file is a roster lock, else None.

    A roster-locked file carries a `slots` block naming each slot's team and jersey
    number, which is what turns "track 0" into "athlete A11" in the viewer.
    """
    try:
        doc = json.loads(Path(tracked_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    slots = doc.get("slots")
    if not isinstance(slots, dict) or not slots:
        return None
    out = {}
    for k, v in slots.items():
        if isinstance(v, dict) and v.get("team") is not None:
            out[str(int(k))] = f"{v['team']}{v.get('number', '')}"
    return out or None


def _person_view_cams(tracked_path, cameras):
    """{(track_id, t): [trusted cam names that saw this person]}."""
    tracked = json.loads(Path(tracked_path).read_text(encoding="utf-8"))
    out = {}
    for pf in iter_person_frames(tracked, cameras):
        out[(int(pf.track_id), int(pf.t))] = [v.cam for v in pf.views if v.camera.trusted]
    return out


def _texture_records(records, V, faces, cameras, cams_by_key, masks, dt_thr,
                     min_facing=0.1, occlusion=True, occ_cell=3.0, occ_margin=0.05,
                     fill_holes=True):
    """Per-vertex RGB (n_records, n_verts, 3) uint8 sampled from the source frames.

    Each body is projected into every trusted view that saw it. A vertex takes colour from a
    view only if it faces the camera (`min_facing`), is in frame, is inside that person's mask
    and is not hidden behind another body part (`_zbuffer_visible`). Samples are blended
    across views, weighted by how much the vertex faces each camera. Vertices no camera saw
    are filled from the nearest seen colour (`_fill_holes`)."""
    n, nv = V.shape[0], V.shape[1]
    colors = np.full((n, nv, 3), 170, np.uint8)              # neutral fallback (RGB)
    A = _mesh_adjacency(faces, nv) if fill_holes else None
    # Records are sorted by timestamp, so the decoded frames of the previous timestamp can be
    # dropped. This keeps the cache at about one frame per camera instead of every frame.
    frame_cache = {}
    cache_t = None
    colored_frac = []
    for r, p in enumerate(records):
        tid, t = int(p["track_id"]), int(p["t"])
        if t != cache_t:
            frame_cache.clear()
            cache_t = t
        Vr = V[r]
        nrm = _vertex_normals(Vr, faces)
        acc = np.zeros((nv, 3), np.float32); wsum = np.zeros(nv, np.float32)
        for cam_name in cams_by_key.get((tid, t), []):
            cam = cameras.get(cam_name)
            if cam is None:
                continue
            ck = (cam_name, t)
            if ck not in frame_cache:
                fp = RAW_DIR / cam_name / f"t{t:06d}.jpg"
                frame_cache[ck] = cv2.imread(str(fp)) if fp.exists() else None
            frame = frame_cache[ck]
            if frame is None:
                continue
            H, W = frame.shape[:2]
            uv = project_world_to_pixel(Vr, cam.R, cam.t, cam.K, cam.dist)
            Xc = Vr @ cam.R.T + cam.t.reshape(3)
            vd = cam.C[None, :] - Vr
            dist = np.linalg.norm(vd, axis=1)
            vd /= (dist[:, None] + 1e-9)
            facing = (nrm * vd).sum(1)
            infront = ((Xc[:, 2] > 0.1)
                       & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1))
            valid = infront & (facing > min_facing)
            if masks is not None:                            # gate to this person's silhouette
                m = masks.get((tid, t, cam_name))
                if m is not None:
                    valid &= _sample_dt(m, uv) < dt_thr
            if occlusion and valid.any():                    # reject self-occluded (hidden) verts
                valid &= _zbuffer_visible(uv, dist.astype(np.float32), W, H,
                                          infront, occ_cell, occ_margin)
            if not valid.any():
                continue
            col = _bilinear_bgr(frame, uv)                   # (nv,3) BGR
            wv = np.where(valid, np.clip(facing, 0, 1) ** 3, 0.0).astype(np.float32)
            acc += wv[:, None] * col
            wsum += wv
        seen = wsum > 1e-6
        if seen.any():
            rgb = np.zeros((nv, 3), np.float32)
            rgb[seen] = (acc[seen] / wsum[seen, None])[:, ::-1]   # BGR -> RGB
            if fill_holes and not seen.all():                     # propagate nearest seen colour
                rgb = _fill_holes(rgb, seen, A)
            elif not seen.all():
                rgb[~seen] = rgb[seen].mean(0)                    # flat fallback (fill disabled)
            colors[r] = np.clip(rgb, 0, 255).astype(np.uint8)
        colored_frac.append(float(seen.mean()))
    # a record no camera saw (an interpolated/infilled body) inherits the nearest-in-time
    # textured record of its own track instead of staying flat grey
    by_track = {}
    for r, p in enumerate(records):
        by_track.setdefault(int(p["track_id"]), []).append(r)
    n_inherit = 0
    for r, p in enumerate(records):
        if colored_frac[r] > 0:
            continue
        sibs = [q for q in by_track[int(p["track_id"])] if colored_frac[q] > 0]
        if sibs:
            src = min(sibs, key=lambda q: abs(int(records[q]["t"]) - int(p["t"])))
            colors[r] = colors[src]
            n_inherit += 1
    if n_inherit:
        print(f"[texture] {n_inherit} view-less records inherited the nearest "
              f"textured frame's colours")
    return colors, np.asarray(colored_frac)


def _rebuild_meshes(records, gender, num_betas, device):
    """records: list of dicts with betas/global_orient/body_pose/transl. Returns (V,3) float32
    vertices stacked (n_records, n_verts, 3) and the shared faces (n_faces,3) uint32."""
    n = len(records)
    model = build_smplx(gender, num_betas=num_betas, batch_size=n).to(device)
    betas = torch.tensor([r["betas"] for r in records], dtype=torch.float32, device=device)
    go = torch.tensor([r["global_orient"] for r in records], dtype=torch.float32, device=device)
    bp = torch.tensor([r["body_pose"] for r in records], dtype=torch.float32, device=device)
    tr = torch.tensor([r["transl"] for r in records], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(betas=betas, global_orient=go, body_pose=bp, transl=tr)
        V = out.vertices.detach().cpu().numpy().astype(np.float32)     # (n,10475,3)
    faces = model.faces.astype(np.uint32)                              # (20908,3)
    return V, faces


def _triangulated_skeletons(tracked_path, cfg: Cfg, cameras):
    """{(track_id, t): joints17x3 (NaN where missing)} - the raw target the fit used."""
    tracked = json.loads(Path(tracked_path).read_text(encoding="utf-8"))
    out = {}
    for pf in iter_person_frames(tracked, cameras):
        joints = triangulate_person_frame(pf, cfg.kp_thr, 25.0, 2)
        sk = np.full((17, 3), np.nan)
        for j in range(17):
            r = joints[j]
            if r:
                sk[j] = r["X"]
        out[(int(pf.track_id), int(pf.t))] = sk
    return out


def run(bodies_path, tracked_path, out_dir, gender, device, texture=True, dt_thr=3.0,
        min_facing=0.1, occlusion=True, occ_margin=0.05, fill_holes=True):
    doc = json.loads(Path(bodies_path).read_text(encoding="utf-8"))
    num_betas = len(next(iter(doc["tracks"].values()))["betas"])
    cfg = Cfg()
    cameras = load_cameras()
    skels = _triangulated_skeletons(tracked_path, cfg, cameras) if tracked_path else {}

    # flatten to a stable record order (timestamp, then track) so rec index is deterministic
    timestamps = sorted(int(t) for t in doc["frames"])
    records, frames_manifest = [], {}
    for t in timestamps:
        people = doc["frames"][str(t)]
        flist = []
        for p in people:
            rec = len(records)
            records.append(p)
            sk = skels.get((int(p["track_id"]), int(t)))
            joints = (None if sk is None
                      else [(None if not np.isfinite(sk[j]).all() else [float(v) for v in sk[j]])
                            for j in range(17)])
            entry = {"track_id": int(p["track_id"]), "rec": rec, "joints": joints}
            # records added by infill_bodies.py are interpolated; the viewer shows them differently
            if p.get("interp"):
                entry["interp"] = True
            flist.append(entry)
        frames_manifest[str(t)] = flist

    print(f"[export] rebuilding {len(records)} posed SMPL-X meshes ({gender}) on {device} ...")
    V, faces = _rebuild_meshes(records, gender, num_betas, device)
    n_verts, n_faces = V.shape[1], faces.shape[0]

    out_dir = Path(out_dir)
    (out_dir / "bodies").mkdir(parents=True, exist_ok=True)
    V.reshape(-1).tofile(out_dir / "bodies" / "verts.bin")       # float32 LE
    faces.reshape(-1).tofile(out_dir / "bodies" / "faces.bin")   # uint32 LE

    has_color = False
    if texture:
        masks = load_masks(DEFAULT_MASKS) if Path(DEFAULT_MASKS).exists() else None
        cams_by_key = _person_view_cams(tracked_path, cameras)
        print(f"[export] texturing {len(records)} bodies from 4K frames "
              f"({'mask-gated' if masks else 'geometric-only'}"
              f"{', occlusion-tested' if occlusion else ''}) ...")
        colors, colored_frac = _texture_records(records, V, faces, cameras, cams_by_key, masks,
                                                dt_thr, min_facing=min_facing,
                                                occlusion=occlusion, occ_margin=occ_margin,
                                                fill_holes=fill_holes)
        colors.reshape(-1).tofile(out_dir / "bodies" / "colors.bin")   # uint8 RGB
        has_color = True
        print(f"[export]   colors.bin ({colors.nbytes / 1e6:.1f} MB)  "
              f"camera-seen verts/body median {np.median(colored_frac) * 100:.0f}% "
              f"(min {colored_frac.min() * 100:.0f}%){'; holes filled from nearest seen' if fill_holes else ''}")

    # The viewer checks that the 2D People map and these bodies come from the same recording,
    # so the source files are recorded here.
    manifest = {
        "version": "1",
        "source": Path(bodies_path).name,
        "tracked_source": _tracked_origin(tracked_path),
        "n_verts": int(n_verts),
        "n_faces": int(n_faces),
        "stride": int(n_verts * 3),
        "verts_url": "bodies/verts.bin",
        "faces_url": "bodies/faces.bin",
        "color_url": "bodies/colors.bin" if has_color else None,
        "has_color": has_color,
        "n_records": len(records),
        "tracks": sorted(int(k) for k in doc["tracks"]),
        "slots": _slot_labels(tracked_path),
        "timestamps": timestamps,
        "frames": frames_manifest,
        "bones": COCO_BONES,
    }
    (out_dir / "bodies.json").write_text(json.dumps(manifest), encoding="utf-8")

    vb = (out_dir / "bodies" / "verts.bin").stat().st_size / 1e6
    print(f"[export] wrote {out_dir / 'bodies.json'}")
    print(f"[export]   verts.bin {vb:.1f} MB  ({len(records)} bodies x {n_verts} verts)")
    print(f"[export]   faces.bin ({n_faces} tris, shared)")
    print(f"[export]   timeline {len(timestamps)} frames, tracks {manifest['tracks']}, "
          f"{len(records)} body-instances")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bodies", default=str(DEFAULT_BODIES), help="fit output (default: work/bodies/bodies_full.json)")
    ap.add_argument("--tracked", default=None,
                    help="the tracked people, for the raw skeleton overlay (default: adapter.DEFAULT_TRACKED)")
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT), help="viewer/public dir")
    ap.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--no_texture", action="store_true", help="skip per-vertex texturing (flat meshes)")
    ap.add_argument("--texture_dt_thr", type=float, default=3.0,
                    help="max outside-mask distance (px) to accept a vertex colour sample")
    ap.add_argument("--min_facing", type=float, default=0.1,
                    help="back-face cull threshold (normal.viewdir); higher drops grazing samples")
    ap.add_argument("--no_occlusion", action="store_true",
                    help="disable the self-occlusion z-buffer test (debug / A-B)")
    ap.add_argument("--occ_margin", type=float, default=0.05,
                    help="depth tolerance (m) for the occlusion z-buffer; smaller = stricter")
    ap.add_argument("--no_fill", action="store_true",
                    help="don't hole-fill unseen verts from nearest seen colour (flat mean instead)")
    args = ap.parse_args()
    if args.tracked is None:
        from adapter import DEFAULT_TRACKED
        args.tracked = str(DEFAULT_TRACKED)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    run(args.bodies, args.tracked, args.out_dir, args.gender, device,
        texture=not args.no_texture, dt_thr=args.texture_dt_thr, min_facing=args.min_facing,
        occlusion=not args.no_occlusion, occ_margin=args.occ_margin, fill_holes=not args.no_fill)


if __name__ == "__main__":
    main()
