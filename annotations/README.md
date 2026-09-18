# annotations/

`court_tiepoints_triangulated.json` lists 20 landmarks of the arena that are mostly off the floor:
the corners and rims of both basketball backboards, four corners of the volleyball net, and the
corners of a bench. Each one was clicked by hand in every camera that saw it on the empty arena,
and those clicks were combined into a single 3D position in metres.

For each landmark the file stores its pixel clicks per camera, the 3D position, and how far
that position lands from the clicks when projected back into the images (`point_rms_px`,
median about 9 px on 4K frames).

The Depth Anything 3 step (`arenaDA3/`) uses these points only for a printed quality check: does
the predicted depth put them in the right order of distance. During a game some of these objects
may have been moved (the hoops are often pushed aside), so a lower score there is expected.
