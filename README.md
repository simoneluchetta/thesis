# Multi-camera 3D reconstruction of a sports arena and its athletes

> This work was developed together with a friend of mine who graduated from university a couple
> of years ago, and as a project I carried out while teaching I.T. Technology at "Marconi Rovereto", along with my colleagues and my students.

This was for my master's thesis *"Exploring Implicit and Parametric Models for 3D Human 
Reconstruction in Basketball"* (Simone Luchetta).

We have 10 synchronised 4K cameras filming a basketball game. 
From those videos this project finds the athletes, tracks them, reads their shirt numbers and fits a 3D body to each of them. 
It also rebuilds the arena itself in 3D. 
Everything is shown in a web viewer:

- **People map**: the athletes on the court seen from above, with team and shirt number, and
  the same moment as 3D bodies;
- **NeRF**: the arena as a 3D point cloud, still or played back over time with the athletes in
  it, and the point cloud of a NeRF of the empty arena.

The other tabs within the webapp were used for a painful process of camera hand placement (for an initial camera guess for COLMAP), 
that later became a "PnP clicking game" needed for a more precise camera calibration.

## What you need

- A PC with an NVIDIA GPU (it was made on an RTX 3080 with 10 GB) and about 40 GB of free disk
  space for a 30 second window.
- [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs Python and all
  the libraries by itself.
- [Node.js](https://nodejs.org/) 20 or newer, for the viewer.
- [Git](https://git-scm.com/), which uv uses to download Depth Anything 3.
- Libraries and networks weights (a few GB).
- On Windows, keep this folder at a short path (My example: `C:\projects\LuchettaSimoneCodes`). 
  Windows provided a great feature that will break loading libraries if the path is longer than 200 characters...

## How to run it

1. Put the synchronised videos in `input_videos/`, named `out1.mp4` ... `out8.mp4`,
   `out12.mp4`, `out13.mp4` (see `input_videos/README.md`).

2. From this folder, run:

   ```
   uv run pipeline.py
   ```

   This processes the first 30 seconds. To choose another part of longer videos:
   `uv run pipeline.py --start 120 --seconds 30`.

   If it stops for any reason, run the same command again: it continues where it left off. (Fingers crossed)

3. Start the viewer webapp:

   ```
   cd src/viewer
   npm install
   npm run dev
   ```

   (`npm install` only the first time; on Windows, if PowerShell refuses `npm`, type `npm.cmd`.)
   (Nel mio caso ho dovuto optare per la seconda `npm.cmd`, e si va sul sicuro)
   Then open http://127.0.0.1:5173.

## Notes

For 30 seconds of video footage on an RTX 3080:

| Part | Time |
|---|---|
| people (frames, detection, tracking, shirt numbers, roster) | about 50 minutes |
| bodies (3D SMPL-X bodies) | 3 to 4.5 hours |
| world (Depth Anything 3 worlds) | about 45 minutes |

(The first run also spends some time installing the two Python environments)

## Folder organization

```
pipeline.py               the one command that runs everything
input_videos/             where the videos go
src/                      the athletes in 2D and 3D, and the web viewer (viewer/)
arenaDA3/                 the arena in 3D with Depth Anything 3
NeRF/                     the trained NeRF of the empty arena
placements/, diagnostics/ the camera calibration (where each camera is, and its lens)
annotations/              measured points above the floor, used to check the 3D arena
work/                     created by the pipeline: frames and intermediate files (about 17 GB for
                          30 s). Safe to delete once the results are in the viewer; the next run
                          then starts from the beginning.
```
