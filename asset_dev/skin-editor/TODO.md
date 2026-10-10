# Auto-Dunmer TODO

- **Reddish hair greyed as skin** on a few skins (e.g. blacksmithmale1_w, crafterfemale1_d,
  smeltermale1_w, noblemale3_w). Pale red hair and skin overlap in colour, so no automatic rule
  separates them safely. Options:
  - a click-to-mark page: click hair pixels on a zoomed head, export `hair_colours` lists for
    `overrides.json`;
  - or fix those few skins by hand in the editor after the full run.
- **128x128 and other non-standard textures** (e.g. aristocratfemale2_a) aren't processed yet.
- **cookfemale1_a / cookfemale1_b** in `3.mcolonies_skins_p2/grey_export` have hair pixels erased to
  transparent (73 and 106 pixels), which will show as holes in game.
- **Blonde hair against grey skin** still looks a bit off (golden blonde sits opposite the
  blue-teal skin and is much lighter than the face). Tried ash blonde hair, neutral-violet skin for
  blondes, and both (`asset_dev/4.blonde_options`); none was liked enough. Revisit.
