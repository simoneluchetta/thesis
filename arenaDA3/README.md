# arenaDA3

Builds the 3D arena shown in the viewer's NeRF tab from the game videos, using the depth maps
that Depth Anything 3 (DA3) predicts for each camera.

| Part | What it does |
|---|---|
| `run_world.py` | runs the steps below in order; `pipeline.py` calls it |
| `da3_prior/` | runs DA3 on the ten cameras and saves a depth map for each (its own Python environment) |
| `da3_fusion/` | combines the depth maps into one point cloud of the arena |
| `da3_4d/` | does the same every 5th frame, so the arena can be played back with the athletes in it |
| `export_webapp_assets.py` | copies the results into the viewer |
| `arena_cameras.py` | loads the camera calibration, used by the scripts above |

DA3 needs older library versions than the rest of the project (numpy below 2), so it has its own
uv environment in `da3_prior/`. Everything here runs in that environment:

```
uv run --project arenaDA3/da3_prior python arenaDA3/run_world.py --work work
```

The model weights (1.6 GB) are downloaded on the first run. They are licensed CC BY-NC 4.0, for
non-commercial use.
