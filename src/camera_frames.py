"""Two camera helpers shared by the scripts in this project.

`_placement_to_cam_from_world` turns one viewer camera entry, a position in metres plus three
Euler angles (yaw about Z, then pitch about X, then roll about Y, the order three.js
decomposes in), into the world-to-camera rotation and translation OpenCV expects.

`extract_frames` pulls the requested frames out of the recordings in input_videos/ at full
4K. The three overhead units record two eyes side by side in one frame, so those frames are
cut in half and the halves become separate cameras from here on (camNNL / camNNR, encoded
ids 100+N / 200+N).

The pipeline extracts its frames in run_people.py and uses only the pose helper from here.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

N_CAMERAS = 13


def _parse_stereo_cams(s: str) -> set[int]:
    return {int(x) for x in s.split(",") if x.strip()}


def _split_stereo_frame(src: Path, dst_l: Path, dst_r: Path) -> None:
    """Split a ceiling unit's frame down the middle into its two views."""
    img = cv2.imread(str(src))
    if img is None:
        raise RuntimeError(f"Could not read {src} for stereo split")
    h, w = img.shape[:2]
    half = w // 2
    dst_l.parent.mkdir(parents=True, exist_ok=True)
    dst_r.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_l), img[:, :half], [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(dst_r), img[:, half : half * 2], [cv2.IMWRITE_JPEG_QUALITY, 95])


def extract_frames(
    videos_dir: Path,
    timestamps: list[int],
    raw_dir: Path,
    stereo_cams: set[int],
    cam_ids: list[int] | None = None,
) -> dict[int, dict[int, Path]]:
    """Pull the requested frames out of every clip, at full resolution.

    For each cam in `stereo_cams` we also write the left/right halves to
    raw_dir/cam{c:02d}L/ and raw_dir/cam{c:02d}R/, and the returned mapping
    contains those split eyes (encoded ids 100+c, 200+c) in place of the
    full-frame entry. The original full-frame jpgs stay on disk for
    archival / debugging.

    Returns {timestamp -> {encoded_cam_id -> path}}.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    out: dict[int, dict[int, Path]] = {t: {} for t in timestamps}
    for c in (cam_ids if cam_ids is not None else range(1, N_CAMERAS + 1)):
        cam_dir = raw_dir / f"cam{c:02d}"
        cam_dir.mkdir(exist_ok=True)
        is_stereo = c in stereo_cams
        if is_stereo:
            (raw_dir / f"cam{c:02d}L").mkdir(exist_ok=True)
            (raw_dir / f"cam{c:02d}R").mkdir(exist_ok=True)
        video_path = videos_dir / f"out{c}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        try:
            n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            wrote_full = 0
            wrote_split = 0
            for t in timestamps:
                if t < 0 or t >= n_total:
                    raise ValueError(f"timestamp {t} out of range for {video_path} (n_frames={n_total})")
                dst_full = cam_dir / f"t{t:06d}.jpg"
                if not dst_full.exists():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, t)
                    ret, frame = cap.read()
                    if not ret:
                        raise RuntimeError(f"Could not read frame {t} from {video_path}")
                    cv2.imwrite(str(dst_full), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    wrote_full += 1

                if is_stereo:
                    dst_l = raw_dir / f"cam{c:02d}L" / f"t{t:06d}.jpg"
                    dst_r = raw_dir / f"cam{c:02d}R" / f"t{t:06d}.jpg"
                    if not (dst_l.exists() and dst_r.exists()):
                        _split_stereo_frame(dst_full, dst_l, dst_r)
                        wrote_split += 1
                    out[t][100 + c] = dst_l
                    out[t][200 + c] = dst_r
                else:
                    out[t][c] = dst_full
        finally:
            cap.release()
        suffix = " (stereo, split L/R)" if is_stereo else ""
        print(f"  cam{c:02d}: {wrote_full} full frames + {wrote_split} stereo splits{suffix}")
    return out


def _euler_zxy_to_rotation_matrix(yaw_z: float, pitch_x: float, roll_y: float) -> np.ndarray:
    """The viewer's three angles, back into a rotation matrix.

    The viewer decomposes with three.js in ZXY order and stores them in that order, so this
    must use the same convention.
    """
    from scipy.spatial.transform import Rotation as ScipyRotation

    # capital axes mean intrinsic rotations
    # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html
    return ScipyRotation.from_euler("ZXY", [yaw_z, pitch_x, roll_y]).as_matrix()


def _placement_to_cam_from_world(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    """A placement entry as a pose, in the world-to-camera form everything else expects."""
    position = np.asarray(entry["position"], dtype=np.float64)
    rot_zxy = np.asarray(
        entry.get("rotation") or entry.get("rotationYXZ") or [0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    R_c2w = _euler_zxy_to_rotation_matrix(rot_zxy[0], rot_zxy[1], rot_zxy[2])
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ position
    return R_w2c, t_w2c


if __name__ == "__main__":
    #   uv run python camera_frames.py --timestamps 100,5000,12000,18000,25000,32000
    # Frames land in --raw_dir as cam01/t000100.jpg, and existing ones are kept, so the
    # command can be re-run to add timestamps.
    import argparse

    ap = argparse.ArgumentParser(description="Extract 4K frames from the session-1 recordings; "
                                             "the overhead units are split into their two eyes.")
    ap.add_argument("--videos_dir", default=str(Path(__file__).resolve().parent.parent / "input_videos"),
                    help="folder holding the synchronised recordings out1.mp4 .. out13.mp4 "
                         "(default: input_videos/ at the top of the project)")
    ap.add_argument("--timestamps", required=True, help="comma-separated frame indices, e.g. 100,5000,12000")
    ap.add_argument("--raw_dir", default="../work/extracted_frames", help="the frame cache to write into")
    ap.add_argument("--stereo_cams", default="9,10,11", help="cameras recorded as two eyes side by side")
    ap.add_argument("--cams", default=",".join(str(c) for c in range(1, N_CAMERAS + 1)),
                    help="recorder numbers to read (out<N>.mp4); e.g. 1,2,3,4,5,6,7,8,12,13 for a rig "
                         "without the overhead units, with --stereo_cams \"\"")
    args = ap.parse_args()

    timestamps = [int(x) for x in args.timestamps.split(",") if x.strip()]
    frames = extract_frames(Path(args.videos_dir), timestamps, Path(args.raw_dir),
                            _parse_stereo_cams(args.stereo_cams),
                            cam_ids=[int(x) for x in args.cams.split(",") if x.strip()])
    n = sum(len(v) for v in frames.values())
    print(f"{n} frame files for {len(timestamps)} timestamp(s) under {args.raw_dir}")
