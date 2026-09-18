# NeRF

A neural radiance field of the arena, trained earlier in the project on the empty hall. It is
not trained again: this folder only turns the saved model into the point cloud shown in the
viewer's NeRF tab.

| File | What it is |
|---|---|
| `model/nerf.pt` | the trained network (8 layers, 256 wide), trained with a Depth Anything 3 depth prior |
| `model/camera_poses.json` | the camera positions the model was trained with |
| `model/floor_points.ply` | the floor from an earlier NeRF of the same arena, which is sharper than this one |
| `nerf_model.py` | the network, rebuilt so the saved weights can be loaded |
| `export_nerf_cloud.py` | samples the network and writes `src/viewer/public/nerf/nerf_cloud.ply` |

The pipeline runs the export by itself when the cloud is missing. To run it by hand, from the
repository root:

```
uv run --project src python NeRF/export_nerf_cloud.py
```

It takes a few seconds on the GPU.
