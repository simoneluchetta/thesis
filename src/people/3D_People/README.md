# 3D_People

The athletes in 3D. `../../run_bodies.py` runs these in order:

| Script | What it does |
|---|---|
| `adapter.py` | loads the cameras and, for each athlete and instant, the camera views that saw them |
| `triangulate.py` | a 3D skeleton from the 2D joints seen by several cameras |
| `masks.py` | the outline of each person in each view (Mask R-CNN) |
| `fit_smplx.py` | fits a SMPL-X body: one body shape per athlete, one pose per instant (uses `geom.py`) |
| `infill_bodies.py` | fills the instants the fit had to leave out, following the athlete on the court |
| `export_bodies.py` | colours the bodies from the video frames and writes them for the viewer |

`vposer.py` is an optional pose prior, not used by default.

`models/smplx/SMPLX_NEUTRAL.npz` is the SMPL-X body model. Its licence does not allow
redistribution; it comes from https://smpl-x.is.tue.mpg.de after registering.
