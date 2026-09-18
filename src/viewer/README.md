# viewer

The web viewer (React and three.js). It has no server: it reads the files in `public/`.

```
npm install        (first time only)
npm run dev
```

then open http://127.0.0.1:5173.

The two tabs the pipeline fills:

| Tab | Shows | Files in `public/` |
|---|---|---|
| People map | the athletes on the court seen from above, with team and shirt number; "3D bodies" switches to the SMPL-X bodies | the `people` part of `scene.json`, `people_thumbs/`, `bodies.json`, `bodies/` |
| NeRF | the Depth Anything 3 world (still and over time) and the NeRF point cloud | `nerf/` |

The other tabs (3D viewer, Camera placer, Court calibrator, PnP result, Annotate) were the tools
used to calibrate the cameras and check the results. They still open, with the data they had.

`update_people_block.py` writes the People map data into `scene.json`; the pipeline runs it.
`update_pnp_overlay.py` adds a set of camera poses to `scene.json` for the PnP result tab:

```
cd src/viewer
uv run --project .. python update_pnp_overlay.py --placer ../../placements/placed_cameras_redfox_repaired.json
```

Do not delete `public/scene.json`: it holds the cameras and the arena for all the tabs.
