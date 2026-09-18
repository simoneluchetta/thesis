# da3_fusion

`fuse_da3_tsdf.py` turns the ten depth maps into one point cloud of the arena.

Each depth map is first scaled so that the floor lands exactly on the real floor. Then all maps
are fused with a TSDF volume (5 cm cells): every camera votes on where the surfaces are, and the
surfaces end up where the votes agree. It writes the point cloud, a mesh, two top and side views
and a short report.

Thin things such as hoop rims are too small for this and do not appear.
