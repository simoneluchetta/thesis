# input_videos

Put the synchronised videos of the cameras here, one file per camera, named after the recorder
number:

```
out1.mp4  out2.mp4  out3.mp4  out4.mp4  out5.mp4
out6.mp4  out7.mp4  out8.mp4  out12.mp4 out13.mp4
```

These ten are the cameras around the court. Recorders 9, 10 and 11 (the ceiling units) are not
used; leave them out or keep them here, it makes no difference.

The videos must start at the same moment and have the same frame rate. They can be longer than
the part you want: `uv run pipeline.py --start 120 --seconds 30` processes 30 seconds starting
two minutes in. Without options the first 30 seconds are used.
