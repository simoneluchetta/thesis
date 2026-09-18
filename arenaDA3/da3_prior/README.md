# da3_prior

`export_depth_priors_da3.py` runs Depth Anything 3 on one picture per camera and saves a depth
map and a confidence map for each. The model is told exactly where every camera is, which makes
its depth consistent between cameras.

At the end it prints two checks: how well the depth agrees with the known flat floor, and
whether points of known height (hoops, bench) come out in the right order of distance.

This folder is also the uv project of the DA3 environment (`pyproject.toml`, `uv.lock`), with
Depth Anything 3 installed from GitHub at a fixed commit.
