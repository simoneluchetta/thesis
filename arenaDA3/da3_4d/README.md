# da3_4d

`fuse_4d.py` repeats the fusion of `../da3_fusion/` on the real video frames, every 5th frame.
Since the cameras are synchronised, the ten frames of one instant show the same moment, so the
players are captured like any other object. The result is one point cloud per instant, which the
NeRF tab plays back as the "sliding DA3 world".

The floor scaling is redone at every instant. When it looks unreliable (too few floor pixels, for
example) the camera's usual value is used instead, so the playback does not jump.

About 15 seconds per instant on an RTX 3080, so about 40 minutes for 30 seconds of video. An
interrupted run skips the instants already done.
