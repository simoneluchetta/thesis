# Annotate tab

The same text is in the tab's **Instructions** panel.

## What this is

You say who is who at 30 moments of the game. The result, `gt_answers.json`, was used during the
thesis to measure how well the tracking works. The pipeline does not need it.

## Important

The team and number already filled in are the tracker's guesses. Always check the shirt in the
photos. If you cannot read the number, leave it empty.

## How to do it

Go through the moments one by one. For each one:

- Click a person on the map or in a photo. Click a photo to zoom in.
- Set the team and the number. The label is copied to the same person in the other moments.
- If the label is wrong only in this moment, tick **only this instant** and fix it.
- If nobody is really there, mark it as a **ghost**.
- If someone is missing, double-click the map where they stand, or use **+ Add person**.
- Only drag a marker if it is more than about half a metre off.
- Tick **done**.

A red warning means the same player is in two places, so one label is wrong. A yellow note means
the same player appears twice close together; that is usually fine.

## Keys

With a person selected: `a` `b` team, `r` referee, `x` other, `g` ghost, `0-9` then `Enter`
number, `o` only this instant.

Any time: `n` next, `p` previous, `d` done.

## When you finish

Click **Save gt_answers.json** and keep the file. Your work is also saved in the browser as you
go, so you can close the tab and come back.
