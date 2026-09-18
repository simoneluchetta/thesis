# src

The main Python project (one uv environment): the athletes in 2D and 3D, and the web viewer.

| Part | What it does |
|---|---|
| `run_people.py` | videos to the People map: frames, detection, tracking, shirt numbers, the ten-athlete roster |
| `run_bodies.py` | the roster to 3D SMPL-X bodies in the People map |
| `people/` | the steps `run_people.py` runs |
| `people/3D_People/` | the steps `run_bodies.py` runs |
| `viewer/` | the web viewer |
| `camera_frames.py`, `_floor_homography_sanity.py` | small camera helpers shared by the scripts |

`pipeline.py` at the repository root runs both scripts. By hand, from the repository root:

```
uv run --project src python src/run_people.py --work work
uv run --project src python src/run_bodies.py --work work
```

The calibration they use is `../placements/placed_cameras_redfox_repaired.json` with
`../diagnostics/court_pnp_cameras_redfox_repaired.json`.

Two model files are needed and are already in place:

- `people/models/osnet_ain_x1_0.pth`, the re-identification network that tells people apart by
  their appearance;
- `people/3D_People/models/smplx/SMPLX_NEUTRAL.npz`, the SMPL-X body model. Its licence does not
  allow redistribution: keep this project private. It can be downloaded, after registering, from
  https://smpl-x.is.tue.mpg.de.

The detection and silhouette networks (Keypoint R-CNN, Mask R-CNN) and the EasyOCR models are
downloaded automatically on the first run.
