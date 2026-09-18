# placements/

`placed_cameras_redfox_repaired.json` holds the position and rotation of each of the
ten perimeter cameras (cam01 to cam08, cam12, cam13) in the court's world frame, in metres.

This is the calibration the 3D results were made with. It starts from the checkerboard
calibration of the second recording session and fixes two things: the lens distortion of
cam07, cam12 and cam13, which folded back on itself near the image corners, and the poses of
cam02 and cam08, which were not at their best fit.

Use it only with `../diagnostics/court_pnp_cameras_redfox_repaired.json`, which holds the
matching lens parameters. Poses and lens parameters were solved together, and mixing them with
another calibration gives badly wrong projections without any error.
