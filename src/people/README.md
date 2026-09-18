# people

The athletes in 2D. `../run_people.py` runs these in order:

| Script | What it does |
|---|---|
| `detect.py` | finds every person in every camera frame (Keypoint R-CNN), puts their feet on the court, and describes their look with OSNet (`reid.py`) |
| `fuse_global.py` | decides which detections in different cameras are the same person, frame by frame (uses helpers from `fuse.py`) |
| `assemble.py` | links those people through time into tracks |
| `roles.py` | tells players from referees and bystanders, by where they stand and what they wear |
| `jersey_mv.py` | reads the shirt numbers with EasyOCR, in every camera that sees the player (uses `jersey.py`) |
| `jersey_repair.py` | splits a track where the shirt number shows it switched from one player to another |
| `roster_lock.py` | forces the result to exactly five athletes per team for the whole clip, learning the two team colours from the footage, and names each one by team and number |

`models/osnet_ain_x1_0.pth` holds the re-identification weights used by `reid.py`.
