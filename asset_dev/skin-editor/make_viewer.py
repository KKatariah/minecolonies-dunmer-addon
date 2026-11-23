"""Build a self-contained HTML page for flipping through skins on 3D citizens.

Usage (from the repo root):
    python asset_dev/skin-editor/make_viewer.py                      # newest 6.ears*, else 5.auto_full*
    python asset_dev/skin-editor/make_viewer.py --skins asset_dev/5.auto_full_v9

Writes asset_dev/_viewer.html, the same page for every run (open it in a browser and keep it
bookmarked: browsers like Firefox keep a separate store for each local file, so one page means
your reviews from every run stay readable). Each skin is shown next to its
original from 1.medieval-original, on the same job model MineColonies uses in game
(hats, hair, chest, skirts, aprons, bags, 3px arms...), read from
citizen_models.json (make that with extract_models.py). The 3D view loads three.js
from cdnjs, so it needs an internet connection; the flat texture views don't.

Each skin can be marked Approved or Edits needed (with common-issue checkboxes, see ISSUES,
and a comment); the sidebar filter can list skins by issue.

Skins are compared with the previous run of the same series (or --previous). A new run starts
from that run's reviews: skins that look the same keep their verdict, changed ones are flagged
"re-check" with the old verdict shown, and the changed pixels can be highlighted. A skin
re-rendered into the same run after you reviewed it is flagged "re-check" the same way. Reviews are kept
in the browser; Export saves them as _reviews.json, and if that file is in the
skins folder when the viewer is rebuilt, its reviews are loaded back in.
"""

import argparse
import base64
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from build_mod import REPO, newest_finished_run, skin_files
from extract_models import HUMANOID, RENDERED_ROOTS
import suspects

HERE = Path(__file__).resolve().parent
ORIGINALS = REPO / "asset_dev" / "1.medieval-original"
MODELS = HERE / "citizen_models.json"
THREE = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
# Common review issues, shown as checkboxes on "Edits needed" (from the v10 review comments).
# Keys are saved in _reviews.json, so rename labels freely but keep keys stable.
ISSUES = [
    ("headwear", "Headband, hat or hood greyed"),
    ("brow_band", "Dark/grey band above eyes"),
    ("accessory", "Jewellery or accessory greyed (ring, piercing, hair tie, buckle)"),
    ("clothing", "Clothing greyed"),
    ("skin_missed", "Skin not greyed"),
    ("blush", "Cheek blush not greyed"),
    ("hair_uneven", "Hair toned unevenly"),
    ("hair_none", "Hair toning on a bald/turbaned character"),
    ("beard", "Beard or beard overlay off"),
    ("blackened", "Clothing turned black"),
    ("contrast", "Skin blends with clothing/hair"),
    ("looks_off", "Correct but looks off (recolour item)"),
    ("uneven_detail", "Detail half-greyed (make uniform)"),
    ("underside_split", "Leg tops / torso underside: back half from the original torso underside, front half from the original leg tops"),
    ("underside_torso", "Leg tops: use the original torso underside's texture, and put the torso underside back to the original"),
    ("dark_shadows", "Greyed skin has super dark pixels (lighten them)"),
    ("brows_flat", "Eyebrows flattened to one colour (check the shading)"),
]
# The issue the u key toggles (marking the skin Edits needed).
SPLIT_UNDERSIDE = "underside_split"
# The issue the t key toggles.
TORSO_UNDERSIDE = "underside_torso"
# The issue the l key toggles.
DARK_SHADOWS = "dark_shadows"


def data_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skins", type=Path, help="folder of finished skins (default: newest 6.ears*, else 5.auto_full*)")
    parser.add_argument("--originals", type=Path, default=ORIGINALS)
    parser.add_argument("--out", type=Path, help="output file (default: _viewer.html next to the run folders)")
    parser.add_argument("--previous", type=Path,
                        help="earlier run to compare against; changed skins are flagged (default: the run before this one)")
    parser.add_argument("--previous-reviews", type=Path,
                        help="reviews to carry over from the earlier run (default: <previous>/_reviews.json)")
    args = parser.parse_args()

    if not MODELS.exists():
        raise SystemExit(f"no {MODELS.name}; run extract_models.py first")
    models = json.loads(MODELS.read_text())
    # Plain body for textures no MineColonies model uses (e.g. the old deliveryman ones).
    models["humanoid"] = {
        "tex": None,
        "parts": [{"name": n, "parent": None, "cubes": c, "pose": p} for n, (c, p) in HUMANOID.items()],
        "rendered": RENDERED_ROOTS, "hidden": [], "rules": [], "lean": "0",
    }

    skins_dir = (args.skins or newest_finished_run()).resolve()
    skins = skin_files(skins_dir)
    if not skins:
        raise SystemExit(f"no skins in {skins_dir}")

    previous = (args.previous or previous_run(skins_dir))
    previous = previous.resolve() if previous else None
    earlier = earlier_runs(skins_dir)
    if previous and previous not in earlier:
        earlier.append(previous)

    # Likely mistakes (see suspects.py), also saved as _suspects.json in the run folder.
    report = suspects.find(skins_dir)
    (skins_dir / "_suspects.json").write_text(json.dumps(report, indent=1))
    print(f"{len(report)} skins with suspected issues")

    # Notes on the fixes made in each run ({skin: [note]} in the run folder's _changes.json, a note
    # being a string or {"at": ISO time, "note": text}), shown on skins to re-check next to the pixel
    # summary. Dated notes let a re-check within the same run show only the fixes since the review.
    notes = {run.name: json.loads((run / "_changes.json").read_text())
             for run in [*earlier, skins_dir] if (run / "_changes.json").exists()}

    entries = []
    for skin in skins:
        orig = args.originals / skin.name
        with Image.open(skin) as im:
            size = im.size
        changed, diff = compare(skin, previous / skin.name) if previous else (None, None)
        _, diff_orig = compare(skin, orig)
        with Image.open(skin) as im:
            pixels = np.array(im.convert("RGBA"))
        entries.append({
            "name": skin.stem,
            # So a review made before the skin was re-rendered in this same run asks for a re-check.
            "hash": hashlib.sha1(pixels.tobytes()).hexdigest()[:12],
            "updated": datetime.fromtimestamp(skin.stat().st_mtime, timezone.utc).isoformat(),
            "size": size,
            "skin": data_uri(skin),
            "orig": data_uri(orig) if orig.exists() else None,
            "changed": changed,
            "diff": diff,
            "diffOrig": diff_orig,
            "same": [run.name for run in earlier if same_pixels(pixels, run / skin.name)],
            # What changed since each earlier run, for the "what changed" note on skins to re-check.
            "changes": {run.name: c for run in earlier if (c := change_summary(pixels, run / skin.name))},
            "notes": {run: n[skin.stem] for run, n in notes.items() if n.get(skin.stem)},
            "suspects": [{"check": i["check"], "label": i["label"], "n": len(i["pixels"])}
                         for i in report.get(skin.stem, [])],
            "suspectMark": mark_pixels(size, [p for i in report.get(skin.stem, []) for p in i["pixels"]]),
        })

    # Reviews exported from earlier runs. The page carries them over together with any it finds in
    # the browser for earlier runs' viewers: kept on skins that look the same as in that run, shown
    # as the last verdict with a re-check flag on the ones that changed.
    sources = {run.name: json.loads((run / "_reviews.json").read_text()).get("reviews", {})
               for run in earlier if (run / "_reviews.json").exists()}
    if args.previous_reviews and previous:
        sources[previous.name] = json.loads(args.previous_reviews.read_text()).get("reviews", {})
    reviews_file = skins_dir / "_reviews.json"
    reviews = json.loads(reviews_file.read_text()).get("reviews", {}) if reviews_file.exists() else {}
    # Skins Claude re-rendered for a specific issue, {name: [issue key]}: shown as "re-check for ..."
    # until the skin gets a new verdict.
    flags_file = skins_dir / "_flags.json"
    flags = json.loads(flags_file.read_text()) if flags_file.exists() else {}
    if sources:
        print("exported reviews from " + ", ".join(f"{run} ({len(r)})" for run, r in sources.items()))

    out = args.out or skins_dir.parent / "_viewer.html"
    html = (TEMPLATE
            .replace("__TITLE__", f"Dunmer skins: {skins_dir.name}")
            .replace("__RUN__", json.dumps(skins_dir.name))
            .replace("__REVIEWS__", json.dumps(reviews))
            .replace("__EARLIER__", json.dumps([run.name for run in earlier]))
            .replace("__SOURCES__", json.dumps(sources))
            .replace("__PREV__", json.dumps(previous.name if previous else None))
            .replace("__ISSUES__", json.dumps(ISSUES))
            .replace("__FLAGS__", json.dumps(flags))
            .replace("__SPLIT__", json.dumps(SPLIT_UNDERSIDE))
            .replace("__TORSO_UNDER__", json.dumps(TORSO_UNDERSIDE))
            .replace("__DARK__", json.dumps(DARK_SHADOWS))
            .replace("__CHECKS__", json.dumps(list(suspects.CHECKS.items())))
            .replace("__THREE__", THREE)
            .replace("__MODELS__", json.dumps(models, separators=(",", ":")))
            .replace("__DATA__", json.dumps(entries)))
    out.write_text(html)
    if previous:
        print(f"{sum(1 for e in entries if e['changed'])} skins changed since {previous.name}")
    print(f"{len(entries)} skins -> {out}")


def previous_run(run):
    """The highest-numbered run before `run` in the same series (6.ears_v3 -> 6.ears_v2 -> 6.ears)."""
    base = re.sub(r"_v\d+$", "", run.name)
    version = lambda p: int(m.group(1)) if (m := re.search(r"_v(\d+)$", p.name)) else 1
    runs = [p for p in run.parent.glob(base + "*")
            if p.is_dir() and re.fullmatch(re.escape(base) + r"(_v\d+)?", p.name) and version(p) < version(run)]
    return max(runs, key=version) if runs else None


def earlier_runs(run):
    """Every run before `run` in the same series, oldest first."""
    base = re.sub(r"_v\d+$", "", run.name)
    version = lambda p: int(m.group(1)) if (m := re.search(r"_v(\d+)$", p.name)) else 1
    runs = [p.resolve() for p in run.parent.glob(base + "*")
            if p.is_dir() and re.fullmatch(re.escape(base) + r"(_v\d+)?", p.name) and version(p) < version(run)]
    return sorted(runs, key=version)


# The torso's underside (base and outer layer) on a 128-wide texture: hardly ever seen, so a change
# there alone doesn't ask for a re-check.
UNDERSIDE = ((28, 16, 8, 4), (28, 32, 8, 4))


def same_pixels(pixels, before):
    """Whether a skin looks the same as in an earlier run, ignoring the torso's underside."""
    if not before.exists():
        return False
    with Image.open(before) as b:
        b = np.array(b.convert("RGBA"))
    if b.shape != pixels.shape:
        return False
    differ = (b != pixels).any(-1)
    k = pixels.shape[1] // 128
    for x, y, w, h in UNDERSIDE if k else ():
        differ[y * k : (y + h) * k, x * k : (x + w) * k] = False
    return not differ.any()


def mark_pixels(size, pixels):
    """PNG data URI marking these (x, y) texture pixels."""
    if not pixels:
        return None
    w, h = size
    mark = np.zeros((h, w, 4), np.uint8)
    for x, y in pixels:
        if x < w and y < h:
            mark[y, x] = (255, 50, 0, 255)
    buf = io.BytesIO()
    Image.fromarray(mark).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def compare(skin, before):
    """(pixels that differ, PNG data URI marking them) against the same skin in an earlier run."""
    if not before.exists():
        return None, None
    with Image.open(skin) as a, Image.open(before) as b:
        a, b = np.array(a.convert("RGBA")), np.array(b.convert("RGBA"))
    if a.shape != b.shape:
        return int(a.shape[0] * a.shape[1]), None
    changed = (a != b).any(-1)
    if not changed.any():
        return 0, None
    mark = np.zeros(a.shape, np.uint8)
    mark[changed] = (255, 0, 220, 255)
    buf = io.BytesIO()
    Image.fromarray(mark).save(buf, "PNG")
    return int(changed.sum()), "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# Body parts on a 128-wide texture as (label, x0, y0, x1, y1), inclusive; the first that holds a pixel
# names it in the change summary.
PARTS = [
    ("head top", 8, 0, 15, 7), ("under chin", 16, 0, 23, 7), ("face", 8, 8, 15, 15),
    ("head sides", 0, 8, 7, 15), ("head sides", 16, 8, 23, 15), ("back of head", 24, 8, 31, 15),
    ("hair/hat layer", 32, 0, 63, 15),
    ("leg tops", 4, 16, 7, 19), ("leg tops", 20, 48, 23, 51), ("soles", 8, 16, 11, 19), ("soles", 24, 48, 27, 51),
    ("torso top", 20, 16, 27, 19), ("torso underside", 28, 16, 35, 19),
    ("torso front", 20, 20, 27, 31), ("torso back", 32, 20, 39, 31), ("torso sides", 16, 20, 31, 31),
    ("hand bottoms", 48, 16, 51, 19), ("hand bottoms", 40, 48, 43, 51), ("arm tops", 44, 16, 47, 19),
    ("arm tops", 36, 48, 39, 51), ("arms", 40, 20, 55, 31), ("arms", 32, 52, 47, 63),
    ("legs", 0, 20, 15, 31), ("legs", 16, 52, 31, 63),
    ("torso layer", 16, 32, 39, 47), ("arm layer", 40, 32, 55, 47), ("arm layer", 48, 48, 63, 63),
    ("leg layer", 0, 32, 15, 47), ("leg layer", 0, 48, 15, 63), ("bust", 64, 48, 83, 55),
    ("extra parts (skirt, apron...)", 64, 0, 127, 63),
]


def change_summary(pixels, before):
    """[[part, pixels changed, tone shift only]] against the same skin in an earlier run, None if it
    is missing there or looks the same. A tone shift only is a change of at most 8 per channel, the
    nudge every grey pixel gets when the skin's grey ramp moves."""
    if not before.exists():
        return None
    with Image.open(before) as b:
        b = np.array(b.convert("RGBA")).astype(int)
    if b.shape != pixels.shape:
        return [["whole skin (new size)", int(pixels.shape[0] * pixels.shape[1]), False]]
    delta = np.abs(b - pixels.astype(int)).max(-1)
    if not delta.any():
        return None
    k = max(pixels.shape[1] // 128, 1)
    counts = {}
    for y, x in zip(*np.nonzero(delta)):
        label = next((p[0] for p in PARTS if p[1] <= x // k <= p[3] and p[2] <= y // k <= p[4]), "elsewhere")
        n, slight = counts.get(label, (0, True))
        counts[label] = (n + 1, slight and bool(delta[y, x] <= 8))
    return [[label, n, slight] for label, (n, slight) in sorted(counts.items(), key=lambda c: -c[1][0])]


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #f4f2ef; --panel: #ffffff; --text: #1d1b19; --muted: #6b655e;
  --line: #ddd8d1; --accent: #a8322d; --sel: #f3dcd9; --stage: #e9e5df;
  --ok: #2e7d4f; --warn: #b8610f;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #161514; --panel: #201e1c; --text: #ece8e3; --muted: #9a938b;
          --line: #34302c; --accent: #e0645c; --sel: #3a2421; --stage: #2a2724;
          --ok: #3f9a66; --warn: #d9822b; }
}
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.4 system-ui, sans-serif; background: var(--bg); color: var(--text);
       display: grid; grid-template-columns: 260px 1fr; height: 100vh; }
aside { border-right: 1px solid var(--line); background: var(--panel); display: flex; flex-direction: column; min-height: 0;
        min-width: 0; overflow: hidden; }
aside header > * { min-width: 0; }
aside header { padding: 12px; border-bottom: 1px solid var(--line); display: grid; gap: 8px; }
input[type=search] { width: 100%; padding: 6px 8px; border: 1px solid var(--line); border-radius: 6px;
                     background: var(--bg); color: var(--text); font: inherit; }
.count { color: var(--muted); font-size: 12px; }
ul { list-style: none; margin: 0; padding: 4px 0; overflow-y: auto; flex: 1; }
li { display: flex; align-items: center; gap: 8px; padding: 3px 12px; cursor: pointer; }
li:hover { background: var(--bg); }
li.sel { background: var(--sel); font-weight: 600; }
li.grp { cursor: default; padding: 8px 12px 2px; font-size: 12px; font-weight: 600; color: var(--muted, #888); text-transform: uppercase; letter-spacing: .04em; }
li .mark { margin-left: auto; font-weight: 700; }
li .warn { color: #e0a000; font-size: 12px; }
li .warn + .mark { margin-left: 4px; }
.suspects { flex-basis: 100%; font-size: 13px; color: var(--text); border-left: 3px solid #e0a000;
            padding: 2px 8px; }
.suspects[hidden] { display: none; }
.suspects b { color: #e0a000; }
li .mark.approved, .status.approved { color: var(--ok); }
li .mark.edits, .status.edits { color: var(--warn); }
li .mark.recheck, .status.recheck, .changed { color: var(--accent); }
.changed { font-size: 12px; font-weight: 600; }
.previous { flex-basis: 100%; font-size: 13px; color: var(--muted); border-left: 3px solid var(--accent);
            padding: 2px 8px; }
.previous[hidden] { display: none; }
.flatwrap { position: relative; }
.flat img.diff { position: absolute; inset: 0; pointer-events: none; display: none;
                 animation: pulse 1.2s ease-in-out infinite; }
body.showchanges .flat img.diff { display: block; }
@keyframes pulse { 50% { opacity: .2; } }
body.suspectmode .flat img.diff { animation: flash .5s steps(1) infinite;
                                  filter: drop-shadow(0 0 2px #ff3000) drop-shadow(0 0 2px #ff3000); }
@keyframes flash { 50% { opacity: 0; } }
select { font: inherit; padding: 4px 6px; border: 1px solid var(--line); border-radius: 6px;
         background: var(--bg); color: var(--text); }
.tools { display: flex; gap: 6px; flex-wrap: wrap; }
.tools button { padding: 3px 10px; font-size: 12px; }
.review { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; padding: 12px 16px 0; }
.review button { font-weight: 600; }
#approve.on { background: var(--ok); border-color: var(--ok); color: #fff; }
#flag.on { background: var(--warn); border-color: var(--warn); color: #fff; }
.status { font-size: 13px; font-weight: 600; }
.issues { flex-basis: 100%; display: flex; flex-wrap: wrap; gap: 4px 14px; }
.issues[hidden] { display: none; }
.issues label:has(input:checked) { color: var(--text); font-weight: 600; }
#comment { flex-basis: 100%; min-height: 64px; padding: 8px; font: inherit; resize: vertical;
           border: 1px solid var(--warn); border-radius: 8px; background: var(--panel); color: var(--text); }
.face { width: 24px; height: 24px; flex: none; image-rendering: pixelated;
        background-position: -24px -24px; border-radius: 3px; }
main { display: flex; flex-direction: column; min-height: 0; overflow-y: auto; }
.bar { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; padding: 12px 16px;
       border-bottom: 1px solid var(--line); background: var(--panel); position: sticky; top: 0; z-index: 1; }
.bar h1 { font-size: 18px; margin: 0; }
.bar .model { color: var(--muted); font-size: 12px; margin-right: auto; }
button { font: inherit; padding: 5px 12px; border: 1px solid var(--line); border-radius: 6px;
         background: var(--bg); color: var(--text); cursor: pointer; }
button:hover { border-color: var(--accent); }
label { display: flex; align-items: center; gap: 4px; color: var(--muted); }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px; min-width: 0; }
.card.dunmer { border-color: var(--accent); }
.card.dunmer h2 { color: var(--accent); }
.card h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin: 0 0 8px; }
.stage { background: var(--stage); border-radius: 8px; display: flex; justify-content: center;
         cursor: grab; touch-action: none; }
.stage:active { cursor: grabbing; }
.stage canvas { display: block; }
.flat { margin-top: 10px; background: var(--stage)
        repeating-conic-gradient(rgba(127,127,127,.15) 0 25%, transparent 0 50%) 0 0 / 16px 16px;
        border-radius: 8px; padding: 8px; }
.flat img { width: 100%; image-rendering: pixelated; display: block; }
.missing { color: var(--muted); padding: 40px; text-align: center; }
.note { color: var(--muted); font-size: 12px; padding: 0 16px 16px; }
@media (max-width: 760px) {
  body { grid-template-columns: 1fr; grid-template-rows: 40vh 1fr; }
  aside { border-right: 0; border-bottom: 1px solid var(--line); }
  .pair { gap: 8px; padding: 8px; }
  .card { padding: 6px; }
}
</style>
</head>
<body>
<aside>
  <header>
    <input id="q" type="search" placeholder="Filter, e.g. baker, female, _d">
    <select id="filter">
      <option value="all">All skins</option>
      <option value="none">Not reviewed</option>
      <option value="approved">Approved</option>
      <option value="edits">Edits needed</option>
    </select>
    <select id="group" title="Sort the list into groups by the issue in the skin's last review (its current one if flagged, otherwise the one before the re-render)">
      <option value="none">No grouping</option>
      <option value="issue">Group by last issue</option>
    </select>
    <label class="count" title="Leave approved skins out of the list, whatever the filter"><input type="checkbox" id="hideApproved"> Hide approved</label>
    <div class="count" id="count"></div>
    <div class="count" id="progress"></div>
    <div class="tools">
      <button id="export" title="Save reviews as _reviews.json; put it in the skins folder">Export reviews</button>
      <button id="import">Import</button>
      <input type="file" id="importFile" accept=".json,application/json" hidden>
    </div>
  </header>
  <ul id="list"></ul>
</aside>
<main>
  <div class="bar">
    <h1 id="title"></h1>
    <span class="changed" id="changedNote"></span>
    <span class="model" id="model"></span>
    <button id="prev" title="Previous (up arrow)">&uarr; Prev</button>
    <button id="next" title="Next (down arrow)">Next &darr;</button>
    <label><input type="checkbox" id="rotate" autocomplete="off"> Rotate</label>
    <label><input type="checkbox" id="walk" autocomplete="off" checked> Walk</label>
    <label title="Job clothes and tools that only show while working"><input type="checkbox" id="work" checked> At work</label>
    <label><input type="checkbox" id="hat" checked> Hat</label>
    <label title="Carried items: logs, fish, books, torches..."><input type="checkbox" id="props"> Items</label>
    <label title="Highlight changed pixels on the Dunmer texture (c)">Highlight
      <select id="highlight">
        <option value="none">Off</option>
        <option value="orig" selected>Changes from original</option>
        <option value="suspects">Suspected issues</option>
      </select></label>
  </div>
  <div class="review">
    <button id="approve" title="Looks good (y or Enter)">&#10003; Approve</button>
    <button id="flag" title="Needs fixing (n)">&#9873; Edits needed</button>
    <span class="status" id="status"></span>
    <div class="suspects" id="suspects" hidden></div>
    <div class="previous" id="previous" hidden></div>
    <div class="previous" id="changes" hidden></div>
    <div class="issues" id="issues" hidden></div>
    <textarea id="comment" hidden placeholder="What needs fixing? e.g. beard still brown, arm patch missed, eyes not red"></textarea>
  </div>
  <div class="pair">
    <section class="card"><h2>Original</h2><div class="stage" id="stage0"></div><div class="flat"><img id="flat0" alt=""></div></section>
    <section class="card dunmer"><h2>Dunmer</h2><div class="stage" id="stage1"></div><div class="flat"><div class="flatwrap"><img id="flat1" alt=""><img id="diff1" class="diff" alt=""></div></div></section>
  </div>
  <p class="note">Models are MineColonies' own job models, so hats, hair, chests, skirts and aprons show the way they do in game.
    Drag to turn, scroll to zoom. Keys: &uarr;/&darr; or j/k to step, y or Enter approve (and go to the next unapproved skin), n edits needed, u needs the split leg-top / torso-underside treatment (back half from the original torso underside, front half from the original leg tops), t needs the leg tops from the original torso underside (and the torso underside put back to the original), l lighten super dark greyed pixels, s / Shift+S next / previous skin with &#9888; suspected issues, r rotate, w walk, a at work, h hat, i items, c cycle the highlight (off / changes from original / suspected issues / since the previous run). &#9888; marks skins with suspected issues (see suspects.py).</p>
</main>
<script id="models" type="application/json">__MODELS__</script>
<script id="data" type="application/json">__DATA__</script>
<script id="reviews" type="application/json">__REVIEWS__</script>
<script id="issueList" type="application/json">__ISSUES__</script>
<script id="flagList" type="application/json">__FLAGS__</script>
<script id="checkList" type="application/json">__CHECKS__</script>
<script id="prevRun" type="application/json">__PREV__</script>
<script id="earlierRuns" type="application/json">__EARLIER__</script>
<script id="reviewSources" type="application/json">__SOURCES__</script>
<script src="__THREE__"></script>
<script>
const skins = JSON.parse(document.getElementById("data").textContent);
const MODELS = JSON.parse(document.getElementById("models").textContent);
const $ = id => document.getElementById(id);
let shown = skins.map((_, i) => i), prevShown = shown.slice(), cur = 0;

const images = new Map();
function loadImage(src) {
  if (!images.has(src)) images.set(src, new Promise((ok, fail) => {
    const im = new Image(); im.onload = () => ok(im); im.onerror = fail; im.src = src;
  }));
  return images.get(src);
}

// Texture name -> MineColonies model, e.g. bakerfemale1_d -> FemaleBakerModel.
function modelFor(name) {
  const m = name.match(/^(.*?)(female|male)\d+/);
  const type = m && MODELS.types[m[1]];
  if (type && MODELS.models[type[m[2]]]) return type[m[2]];
  return "humanoid";
}

// ---- geometry: vanilla ModelPart.Cube / Polygon box UVs ----
// Shading stands in for Minecraft's entity lighting: lit top, darker sides and bottom.
const SHADE = { down: 1.0, up: 0.5, west: 0.6, north: 0.8, east: 0.6, south: 0.8 };
function partGeometry(cubes, texW, texH) {
  const pos = [], uv = [], col = [], idx = [];
  for (const [u, v, x, y, z, w, h, d, [gx, gy, gz], mirror] of cubes) {
    let x0 = x - gx, x1 = x + w + gx;
    const y0 = y - gy, y1 = y + h + gy, z0 = z - gz, z1 = z + d + gz;
    if (mirror) [x0, x1] = [x1, x0];
    const v7 = [x0, y0, z0], v0 = [x1, y0, z0], v1 = [x1, y1, z0], v2 = [x0, y1, z0],
          v3 = [x0, y0, z1], v4 = [x1, y0, z1], v5 = [x1, y1, z1], v6 = [x0, y1, z1];
    const f4 = u, f5 = u + d, f6 = u + d + w, f7 = u + d + w + w, f8 = u + d + w + d, f9 = u + d + w + d + w;
    const f10 = v, f11 = v + d, f12 = v + d + h;
    const faces = [
      [[v4, v3, v7, v0], f5, f10, f6, f11, SHADE.down],
      [[v1, v2, v6, v5], f6, f11, f7, f10, SHADE.up],
      [[v7, v3, v6, v2], f4, f11, f5, f12, SHADE.west],
      [[v0, v7, v2, v1], f5, f11, f6, f12, SHADE.north],
      [[v4, v0, v1, v5], f6, f11, f8, f12, SHADE.east],
      [[v3, v4, v5, v6], f8, f11, f9, f12, SHADE.south],
    ];
    for (const [vs, u1, v1_, u2, v2_, s] of faces) {
      const base = pos.length / 3;
      const uvs = [[u2, v1_], [u1, v1_], [u1, v2_], [u2, v2_]];
      vs.forEach((p, k) => { pos.push(...p); uv.push(uvs[k][0] / texW, uvs[k][1] / texH); col.push(s, s, s); });
      idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  g.setIndex(idx);
  return g;
}

function buildCitizen(cls, img) {
  const def = MODELS.models[cls];
  // The plain body's UVs are in 64-wide units; high-res textures (256x128) are the same layout scaled.
  const [texW, texH] = def.tex || [64, 64 * img.height / img.width];
  const tex = new THREE.Texture(img);
  tex.magFilter = tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false; tex.flipY = false; tex.needsUpdate = true;
  // entityCutoutNoCull: no backface culling, pixels under 0.1 alpha cut out
  const mat = new THREE.MeshBasicMaterial({ map: tex, vertexColors: true, side: THREE.DoubleSide, alphaTest: 0.1 });

  // Model space is y-down with the face towards -z; turn it upright, facing the camera.
  const root = new THREE.Group();
  root.rotation.x = Math.PI;
  root.position.y = 4;
  const parts = {};
  for (const p of def.parts) {
    const g = new THREE.Group();
    g.userData.pose = p.pose;
    if (p.cubes.length) g.add(new THREE.Mesh(partGeometry(p.cubes, texW, texH), mat));
    parts[p.name] = g;
  }
  for (const p of def.parts) {
    if (p.parent) parts[p.parent]?.add(parts[p.name]);
    else if (def.rendered.includes(p.name)) root.add(parts[p.name]);
  }
  const fn = js => { try { return new Function("S", "return (" + js + ");"); } catch { return () => true; } };
  return {
    root, parts, mat, tex,
    hidden: def.hidden.map(path => path[path.length - 1]),
    rules: def.rules.map(([path, js]) => [path[path.length - 1], fn(js)]),
    lean: fn(def.lean),
  };
}

function disposeCitizen(c) {
  c.root.traverse(o => o.geometry?.dispose());
  c.mat.dispose(); c.tex.dispose();
}

function setPose(g, x, y, z, rx, ry, rz) {
  g.position.set(x, y, z);
  g.rotation.set(rx, ry, rz, "ZYX");  // ModelPart: rotationZYX(zRot, yRot, xRot)
}

// What CitizenModel/HumanoidModel.setupAnim does each frame, minus head turning.
function animate(c, S, t, walk) {
  for (const [name, g] of Object.entries(c.parts)) {
    const [x, y, z, rx, ry, rz] = g.userData.pose;
    setPose(g, x, y, z, rx, ry, rz);
    g.visible = true;
  }
  const P = c.parts, age = t * 20, swing = t * 8, amt = walk ? 0.7 : 0;
  const pose = n => P[n].userData.pose;
  const bobZ = Math.cos(age * 0.09) * 0.05 + 0.05, bobX = Math.sin(age * 0.067) * 0.05;
  setPose(P.head, pose("head")[0], 0, pose("head")[2], 0, 0, pose("head")[5]);
  setPose(P.hat, P.head.position.x, 0, P.head.position.z, 0, 0, P.head.rotation.z);
  // Keep the model's own body height: courier bodies hang their cubes above a pivot at y=12
  // (every other model's is 0), and forcing 0 put the torso inside the head.
  setPose(P.body, pose("body")[0], pose("body")[1], pose("body")[2], Number(c.lean(S)) || 0, 0, pose("body")[5]);
  setPose(P.right_arm, -5, 2, 0, Math.cos(swing * 0.6662 + Math.PI) * amt + bobX, 0, bobZ);
  setPose(P.left_arm, 5, 2, 0, Math.cos(swing * 0.6662) * amt - bobX, 0, -bobZ);
  setPose(P.right_leg, pose("right_leg")[0], 12, 0.1, Math.cos(swing * 0.6662) * 1.4 * amt, 0.005, 0.005);
  setPose(P.left_leg, pose("left_leg")[0], 12, 0.1, Math.cos(swing * 0.6662 + Math.PI) * 1.4 * amt, -0.005, -0.005);
  for (const n of c.hidden) if (P[n]) P[n].visible = false;
  for (const [n, f] of c.rules) if (P[n]) P[n].visible = !!f(S);
}

// ---- viewers ----
const has3d = typeof THREE !== "undefined";
const view = { yaw: 0, pitch: 0.12, dist: 95 };
const viewers = [];
if (has3d) {
  for (const n of [0, 1]) {
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(devicePixelRatio);
    const scene = new THREE.Scene();
    const pivot = new THREE.Group(); scene.add(pivot);
    const camera = new THREE.PerspectiveCamera(30, 0.77, 1, 1000);
    $("stage" + n).appendChild(renderer.domElement);
    viewers.push({ renderer, scene, pivot, camera, citizen: null });
  }
  // Make the 3D view short enough that it and the flat texture below both fit in the window.
  const resize = () => viewers.forEach((v, n) => {
    const stage = $("stage" + n), flat = stage.nextElementSibling;
    const top = stage.getBoundingClientRect().top + document.querySelector("main").scrollTop;
    const flatH = flat.offsetHeight + 10, room = innerHeight - top - flatH - 32;
    const w = Math.min(stage.clientWidth, 420, Math.max(room, 240) / 1.3), h = Math.round(w * 1.3);
    v.renderer.setSize(w, h); v.camera.aspect = w / h; v.camera.updateProjectionMatrix();
  });
  addEventListener("resize", resize);
  new ResizeObserver(resize).observe(document.querySelector(".review"));
  document.querySelectorAll(".flat img").forEach(img => img.addEventListener("load", resize));
  resize();

  // drag to turn (both views together), wheel to zoom
  let drag = null;
  for (const n of [0, 1]) {
    const el = $("stage" + n);
    el.addEventListener("pointerdown", e => { drag = { x: e.clientX, y: e.clientY }; el.setPointerCapture(e.pointerId); });
    el.addEventListener("pointermove", e => {
      if (!drag) return;
      view.yaw += (e.clientX - drag.x) * 0.01;
      view.pitch = Math.max(-1.2, Math.min(1.2, view.pitch + (e.clientY - drag.y) * 0.01));
      drag = { x: e.clientX, y: e.clientY };
    });
    el.addEventListener("pointerup", () => drag = null);
    el.addEventListener("wheel", e => { view.dist = Math.max(30, Math.min(200, view.dist * (1 + e.deltaY * 0.001))); e.preventDefault(); }, { passive: false });
  }

  let last = performance.now();
  const tick = now => {
    const dt = (now - last) / 1000; last = now;
    if ($("rotate").checked && !drag) view.yaw += dt * 0.6;
    const S = { work: $("work").checked, hat: $("hat").checked, props: $("props").checked };
    for (const v of viewers) {
      if (v.citizen) animate(v.citizen, S, now / 1000, $("walk").checked);
      v.pivot.rotation.y = view.yaw;
      v.camera.position.set(0, Math.sin(view.pitch) * view.dist, Math.cos(view.pitch) * view.dist);
      v.camera.lookAt(0, 0, 0);
      v.renderer.render(v.scene, v.camera);
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
} else {
  for (const n of [0, 1]) $("stage" + n).innerHTML = '<div class="missing">3D view needs internet (three.js didn\'t load).</div>';
}

// ---- reviews: {name: {status: "approved" | "edits", issues: [key], comment, at, previous}} ----
// `previous` is the review from an earlier run, kept when the skin changed since: until it gets a
// new status the skin is flagged for re-checking.
const RUN = __RUN__;
const STORE = "dunmer-reviews:" + RUN;
const EARLIER = JSON.parse(document.getElementById("earlierRuns").textContent);
// Reviews made for this run (exported file + this browser) win over ones carried from earlier runs.
const own = JSON.parse(document.getElementById("reviews").textContent);
const fileHash = Object.fromEntries(JSON.parse(document.getElementById("data").textContent).map(s => [s.name, s.hash]));
try {
  for (const [name, r] of Object.entries(JSON.parse(localStorage.getItem(STORE) || "{}"))) {
    // A bare "re-check this run" entry the browser saved is dropped when the exported file has a
    // verdict for the skin as it looks now (it was marked stale by mistake, not re-rendered).
    const f = own[name];
    const wrongRecheck = !r.status && r.previous?.run === RUN && f?.status && f.hash === fileHash[name];
    if (!wrongRecheck) own[name] = r;
  }
} catch {}
const lastVerdict = r => r.status || !r.previous ? r : r.previous;
// A review made in this run before the skin was re-rendered (its saved hash differs, or for reviews
// saved before hashes were kept, it is older than the skin file) goes back to asking for a re-check.
for (const [name, r] of Object.entries(own)) {
  const s = skins.find(s => s.name === name);
  if (!s || !(r.status || r.comment || r.issues?.length)) continue;
  const stale = r.hash ? r.hash !== s.hash : !!r.at && new Date(r.at) < new Date(s.updated);
  if (stale) own[name] = { previous: { run: RUN, ...lastVerdict(r) } };
}
let reviews = Object.assign(carryReviews(), own);
const ownNames = new Set(Object.keys(own));
// Reviews from earlier runs, oldest first so newer ones win: exported files, then what this browser
// kept for each earlier run's viewer.
// A review from an earlier run: kept as is if the skin looks the same as it did then, otherwise
// kept as `previous` so the skin asks for a re-check.
function carryInto(out, run, revs) {
  Object.entries(revs || {}).forEach(([name, r]) => {
    const s = skins.find(s => s.name === name);
    if (!s || !(r.status || r.comment || r.issues?.length || r.previous)) return;
    out[name] = s.same.includes(run) ? r : { previous: { run, ...lastVerdict(r) } };
  });
}
function carryReviews() {
  const sources = JSON.parse(document.getElementById("reviewSources").textContent);
  const out = {};
  for (const run of EARLIER) {
    carryInto(out, run, sources[run]);
    try { carryInto(out, run, JSON.parse(localStorage.getItem("dunmer-reviews:" + run) || "{}")); } catch {}
  }
  return out;
}
function saveReviews() {
  const mine = Object.fromEntries(Object.entries(reviews).filter(([name]) => ownNames.has(name)));
  try { localStorage.setItem(STORE, JSON.stringify(mine)); } catch {}
}
function keepReview(name, r) {
  // An emptied review stays as {} so a verdict carried from an earlier run doesn't come back.
  ownNames.add(name);
  const s = skins.find(s => s.name === name);
  if (s) r.hash = s.hash;
  reviews[name] = r.status || r.comment || r.issues?.length || r.previous ? r : {};
}
const MARK = { approved: "\u2713", edits: "\u2691", recheck: "\u27f3" };
const PREV = JSON.parse(document.getElementById("prevRun").textContent);
const ISSUES = JSON.parse(document.getElementById("issueList").textContent);
const ISSUE_LABEL = Object.fromEntries(ISSUES);
// Issues a skin was re-rendered for (from _flags.json), until it gets a new verdict.
const FLAGS = JSON.parse(document.getElementById("flagList").textContent);
const flagsOf = name => reviews[name]?.status ? [] : FLAGS[name] || [];
const CHECKS = JSON.parse(document.getElementById("checkList").textContent);
const recheck = r => !!r?.previous && !r.status;
if (PREV) $("highlight").insertAdjacentHTML("beforeend", `<option value="prev">Changes since ${PREV}</option>`);
function applyHighlight() {
  const mode = $("highlight").value, s = skins[cur];
  const src = mode === "orig" ? s.diffOrig : mode === "prev" ? s.diff : mode === "suspects" ? s.suspectMark : null;
  $("diff1").src = src || "";
  $("diff1").style.visibility = src ? "visible" : "hidden";
  document.body.classList.toggle("showchanges", mode !== "none");
  document.body.classList.toggle("suspectmode", mode === "suspects");
}
$("highlight").addEventListener("change", applyHighlight);
if (PREV) $("filter").insertAdjacentHTML("beforeend",
  `<option value="recheck">Re-check (changed since reviewed)</option><option value="changed">Changed since ${PREV}</option>`);
$("issues").innerHTML = ISSUES.map(([k, label]) => `<label><input type="checkbox" value="${k}"> ${label}</label>`).join("");
$("filter").insertAdjacentHTML("beforeend", `<option value="suspects">\u26a0 Suspected issues</option>` +
  `<optgroup label="Suspected">${CHECKS.filter(([k]) => skins.some(s => s.suspects.some(i => i.check === k)))
    .map(([k, label]) => `<option value="check:${k}">\u26a0 ${label}</option>`).join("")}</optgroup>`);
$("filter").insertAdjacentHTML("beforeend", `<optgroup label="Edits needed for">${
  ISSUES.map(([k, label]) => `<option value="issue:${k}">${label}</option>`).join("")}</optgroup>`);

function setStatus(status) {
  const name = skins[cur].name, r = reviews[name] || {};
  if (r.status === status) delete r.status; else r.status = status;
  r.at = new Date().toISOString();
  keepReview(name, r);
  saveReviews(); showReview(); markItem(cur); updateProgress();
}

// Turn an issue on or off for the current skin; turning it on also marks the skin Edits needed.
function toggleIssue(key) {
  const name = skins[cur].name, r = reviews[name] || {};
  const issues = new Set(r.issues || []);
  if (issues.has(key)) issues.delete(key); else { issues.add(key); r.status = "edits"; }
  r.issues = [...issues];
  if (!r.issues.length) delete r.issues;
  r.at = new Date().toISOString();
  keepReview(name, r);
  saveReviews(); showReview(); markItem(cur); updateProgress();
}

function showSuspects() {
  const list = skins[cur].suspects;
  $("suspects").hidden = !list.length;
  $("suspects").innerHTML = list.length ? "<b>\u26a0 Suspected:</b> " + list.map(i =>
    `${i.label} (${i.n} px)`).join(" \u00b7 ") + " \u2014 highlight \u201cSuspected issues\u201d to see them" : "";
}

function showReview() {
  showSuspects();
  const r = reviews[skins[cur].name] || {};
  $("approve").classList.toggle("on", r.status === "approved");
  $("flag").classList.toggle("on", r.status === "edits");
  $("status").className = "status " + (r.status || (recheck(r) ? "recheck" : ""));
  $("status").textContent = r.status === "approved" ? "Approved" : r.status === "edits" ? "Flagged for edits"
    : recheck(r) ? "Changed since you reviewed it: re-check" : "Not reviewed";
  const flags = flagsOf(skins[cur].name);
  if (flags.length) $("status").textContent += " \u2014 check: " + flags.map(k => ISSUE_LABEL[k] || k).join(", ");
  const p = r.previous;
  $("previous").hidden = !p;
  if (p) $("previous").textContent = `Last review${p.run ? ` (${p.run})` : ""}: ` + [
    p.status === "approved" ? "Approved" : p.status === "edits" ? "Edits needed" : "No verdict",
    ...(p.issues || []).map(k => ISSUE_LABEL[k] || k), p.comment ? `“${p.comment}”` : ""].filter(Boolean).join(" · ");
  showChanges(skins[cur], p);
  const open = r.status === "edits" || !!r.comment || !!r.issues?.length;
  $("comment").hidden = $("issues").hidden = !open;
  $("comment").value = r.comment || "";
  $("issues").querySelectorAll("input").forEach(c => { c.checked = !!r.issues?.includes(c.value); });
}

// What changed since the last review: the fix notes of the runs after it (and of its own run when
// the skin was re-rendered within that run), then which body parts' pixels changed since it.
const RUNS = [...EARLIER, RUN];
function showChanges(s, p) {
  const box = $("changes");
  box.hidden = !p;
  if (!p) return;
  const parts = s.changes[p.run], from = RUNS.indexOf(p.run);
  // A note is a string or {at, note}. Within the review's own run, dated notes newer than the review
  // are the fixes since it; if none are dated, all of that run's notes are shown.
  const text = n => typeof n === "string" ? n : n.note;
  const since = run => {
    const all = s.notes[run] || [];
    if (run !== p.run || !p.at) return all;
    const newer = all.filter(n => typeof n !== "string" && n.at > p.at);
    return all.some(n => typeof n !== "string") ? newer : all;
  };
  const notes = RUNS.filter((run, k) => k > from || (run === p.run && !parts)).flatMap(since).map(text);
  const esc = t => t.replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
  box.innerHTML = "<b>Changed since that review:</b> " +
    (notes.length ? notes.map(esc).join(" · ") : "no fix notes recorded") +
    (parts ? "<br>Pixels changed: " + parts.map(([label, n, slight]) =>
      `${label} ${n}${slight ? " (tone shift only)" : ""}`).join(" · ") : "");
}

function markItem(i) {
  const li = document.querySelector(`li[data-i="${i}"] .mark`);
  if (!li) return;
  const r = reviews[skins[i].name];
  const st = r?.status || (recheck(r) || flagsOf(skins[i].name).length ? "recheck" : "");
  li.className = "mark " + st; li.textContent = MARK[st] || "";
}

function updateProgress() {
  const vals = Object.values(reviews);
  const ok = vals.filter(r => r.status === "approved").length, ed = vals.filter(r => r.status === "edits").length;
  const re = skins.filter(s => recheck(reviews[s.name]) || flagsOf(s.name).length).length;
  $("progress").textContent = `${ok} approved \u00b7 ${ed} edits needed \u00b7 ` +
    (re ? `${re} to re-check \u00b7 ` : "") + `${skins.length - ok - ed} to go`;
}

// Walks the list in its shown order (grouped or not) from the current skin, d = 1 down or -1 up,
// wrapping round; returns the first skin after it that `ok` accepts, or -1. If the current skin
// isn't listed, starts from where it would sit alphabetically.
function findListed(d, ok) {
  const n = shown.length;
  if (!n) return -1;
  let at = shown.indexOf(cur);
  if (at < 0) {
    const p = shown.findIndex(i => i > cur);
    at = (p < 0 ? n : p) - (d > 0 ? 1 : 0);
  }
  for (let k = 1; k <= n; k++) {
    const i = shown[((at + d * k) % n + n) % n];
    if (i !== cur && ok(i)) return i;
  }
  return -1;
}
// The next listed skin without an approval, after the current one and wrapping round to the top.
function nextUnapproved() {
  return findListed(1, i => reviews[skins[i].name]?.status !== "approved");
}
// The next (d = 1) or previous (d = -1) listed skin with suspected issues, wrapping round.
function stepSuspect(d) {
  const i = findListed(d, i => skins[i].suspects.length > 0);
  if (i >= 0) show(i);
}
$("approve").onclick = () => {
  const was = reviews[skins[cur].name]?.status;
  setStatus("approved");
  if (was === "approved") return;
  const next = nextUnapproved();
  if (next >= 0) show(next);
  if ($("hideApproved").checked) renderList();
};
$("flag").onclick = () => { setStatus("edits"); if (!$("comment").hidden) $("comment").focus(); };
$("comment").addEventListener("input", () => {
  const name = skins[cur].name, r = reviews[name] || {};
  r.comment = $("comment").value; r.at = new Date().toISOString();
  keepReview(name, r);
  saveReviews();
});
$("issues").addEventListener("change", () => {
  const name = skins[cur].name, r = reviews[name] || {};
  r.issues = [...$("issues").querySelectorAll("input:checked")].map(c => c.value);
  if (!r.issues.length) delete r.issues;
  r.at = new Date().toISOString();
  keepReview(name, r);
  saveReviews();
});
$("export").onclick = () => {
  const blob = new Blob([JSON.stringify({ run: RUN, exported: new Date().toISOString(), reviews }, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "_reviews.json"; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};
$("import").onclick = () => $("importFile").click();
$("importFile").onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  try {
    const data = JSON.parse(await f.text());
    if (data.run && data.run !== RUN && EARLIER.includes(data.run)) {
      // An export from an earlier run: carry it over like the page does on load, without
      // touching skins you've already reviewed here.
      const carried = {};
      carryInto(carried, data.run, data.reviews);
      for (const [name, r] of Object.entries(carried)) if (!ownNames.has(name)) reviews[name] = r;
    } else {
      Object.entries(data.reviews || data).forEach(([name, r]) => keepReview(name, r));
    }
    saveReviews(); renderList(); showReview(); updateProgress();
  } catch { alert("That file isn't a reviews export."); }
  e.target.value = "";
};

let loading = 0;
async function show(i) {
  cur = i;
  const s = skins[i], cls = modelFor(s.name);
  $("title").textContent = s.name;
  $("model").textContent = cls === "humanoid" ? "no MineColonies model uses this texture; plain body" : cls;
  $("changedNote").textContent = s.changed ? `${s.changed} px changed since ${PREV}` : "";
  applyHighlight();
  location.hash = s.name;
  showReview();
  document.querySelectorAll("li").forEach(li => li.classList.toggle("sel", +li.dataset.i === i));
  document.querySelector("li.sel")?.scrollIntoView({ block: "nearest" });
  const srcs = [s.orig, s.skin];
  srcs.forEach((src, n) => {
    $("flat" + n).src = src || "";
    $("flat" + n).style.visibility = src ? "visible" : "hidden";
  });
  if (!has3d) return;
  const token = ++loading;
  const imgs = await Promise.all(srcs.map(src => src ? loadImage(src) : null));
  if (token !== loading) return;
  viewers.forEach((v, n) => {
    if (v.citizen) { v.pivot.remove(v.citizen.root); disposeCitizen(v.citizen); v.citizen = null; }
    if (imgs[n]) { v.citizen = buildCitizen(cls, imgs[n]); v.pivot.add(v.citizen.root); }
  });
}

// Groups for "Group by last issue": the first one whose issue keys or comment words match wins.
const GROUPS = [
  ["Leg tops / torso underside", ["underside_split", "underside_torso"], /\bleg|underside of (the )?torso|torso underside|bottom of (the )?torso|torso bottom/],
  ["Eyebrows / beard", ["beard", "brow_band", "brows_flat"], /brow|beard|mo?ustache|stubble/],
  ["Hats, hoods & accessories", ["headwear", "accessory", "clothing", "blackened"], /\bhat\b|hood|headband|head ?band|turban|scarf|habit|necklace|\brings?\b|piercing|strap|hair ?tie|\bbow\b|belt|buckle|glove|shoe|sleeve|accessor|shirt|dress|apron|cuff/],
  ["Hair toning & colour", ["hair_uneven", "hair_none"], /hair|fringe|braid|ginger|redder|bald/],
  ["Lighten / darken skin", ["dark_shadows", "contrast"], /lighten|darken|\bdark|\blight|shadow|pale|drool/],
  ["Skin greyed or not", ["skin_missed", "blush"], /grey|gray|skin|lips?\b|mouth|blush/],
  ["Recolour / looks off", ["looks_off", "uneven_detail"], /colou?r|tint|purple|gold|green|weird|looks|remove|replace|flat|texture/],
];
// The review whose issue a skin is grouped by: its own if it has one, else the one before the re-render.
function lastIssue(r) {
  if (!r) return null;
  if (r.comment || r.issues?.length) return r;
  return r.previous ? lastIssue(r.previous) : null;
}
function groupOf(s) {
  const flags = flagsOf(s.name);
  const r = flags.length ? { issues: flags } : lastIssue(reviews[s.name]);
  if (!r) return GROUPS.length + 1;
  const text = (r.comment || "").toLowerCase(), keys = r.issues || [];
  const g = GROUPS.findIndex(([, ks, re]) => keys.some(k => ks.includes(k)) || re.test(text));
  return g < 0 ? GROUPS.length : g;
}
const GROUP_NAME = k => k < GROUPS.length ? GROUPS[k][0] : k === GROUPS.length ? "Other comments" : "No issue noted";

function renderList() {
  const words = $("q").value.toLowerCase().split(/\s+/).filter(Boolean);
  const want = $("filter").value;
  const wants = (s, r) => want === "all" ||
    (want.startsWith("issue:") ? (r?.status === "edits" && !!r.issues?.includes(want.slice(6)))
        || flagsOf(s.name).includes(want.slice(6))
     : want === "recheck" ? recheck(r) || flagsOf(s.name).length > 0
     : want === "suspects" ? s.suspects.length > 0
     : want.startsWith("check:") ? s.suspects.some(i => i.check === want.slice(6))
     : want === "changed" ? !!s.changed
     : (r?.status || "none") === want);
  const hide = $("hideApproved").checked;
  shown = skins.map((s, i) => i).filter(i => words.every(w => skins[i].name.includes(w)) &&
    wants(skins[i], reviews[skins[i].name]) && !(hide && reviews[skins[i].name]?.status === "approved"));
  const grouping = $("group").value === "issue";
  const before = prevShown;
  let heads = new Map();
  if (grouping) {
    const g = new Map(shown.map(i => [i, groupOf(skins[i])]));
    shown.sort((a, b) => g.get(a) - g.get(b) || a - b);
    shown.forEach(i => { if (![...heads.values()].includes(g.get(i))) heads.set(i, g.get(i)); });
    var counts = {}; g.forEach(k => { counts[k] = (counts[k] || 0) + 1; });
  }
  $("list").innerHTML = shown.map(i => (heads.has(i) ? `<li class="grp">${GROUP_NAME(heads.get(i))} (${counts[heads.get(i)]})</li>` : "") +
    `<li data-i="${i}"><span class="face" style="background-image:url(${skins[i].skin});background-size:${skins[i].size[0] * 3}px ${skins[i].size[1] * 3}px"></span>${skins[i].name}${skins[i].suspects.length ? `<span class="warn" title="${
      skins[i].suspects.map(x => x.label).join("\n")}">\u26a0</span>` : ""}<span class="mark"></span></li>`).join("");
  shown.forEach(markItem);
  $("count").textContent = `${shown.length} of ${skins.length} skins`;
  prevShown = shown.slice();
  // The current skin left the list: go to the one that followed it in the old list order
  // (grouped or not), wrapping round.
  if (shown.length && !shown.includes(cur)) {
    const at = before.indexOf(cur), now = new Set(shown);
    const after = at < 0 ? undefined : [...before.slice(at + 1), ...before.slice(0, at)].find(i => now.has(i));
    show(after ?? shown.find(i => i > cur) ?? shown[0]);
  }
  else document.querySelectorAll("li").forEach(li => li.classList.toggle("sel", +li.dataset.i === cur));
}

function step(d) {
  if (!shown.length) return;
  const at = shown.indexOf(cur);
  show(shown[(at + d + shown.length) % shown.length]);
}

$("list").addEventListener("click", e => { const li = e.target.closest("li[data-i]"); if (li) show(+li.dataset.i); });
try { $("group").value = localStorage.getItem("dunmer-group") || "none"; } catch {}
$("group").addEventListener("change", () => {
  try { localStorage.setItem("dunmer-group", $("group").value); } catch {}
  renderList();
});
$("q").addEventListener("input", renderList);
$("filter").addEventListener("change", renderList);
try { $("hideApproved").checked = localStorage.getItem("dunmer-hide-approved") === "1"; } catch {}
$("hideApproved").addEventListener("change", () => {
  try { localStorage.setItem("dunmer-hide-approved", $("hideApproved").checked ? "1" : ""); } catch {}
  renderList();
});
$("prev").onclick = () => step(-1);
$("next").onclick = () => step(1);
const KEYS = { r: "rotate", w: "walk", a: "work", h: "hat", i: "props" };
addEventListener("keydown", e => {
  if (e.target === $("q")) { if (e.key === "Enter") $("q").blur(); return; }
  if (e.target === $("comment")) { if (e.key === "Escape") $("comment").blur(); return; }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key;
  if (k === "ArrowDown" || k === "j") { step(1); e.preventDefault(); }
  else if (k === "ArrowUp" || k === "k") { step(-1); e.preventDefault(); }
  else if (k === "/") { $("q").focus(); e.preventDefault(); }
  else if (k === "y" || k === "Enter") { $("approve").click(); e.preventDefault(); }
  else if (k === "n") { $("flag").click(); e.preventDefault(); }
  else if (k === "u") { toggleIssue(__SPLIT__); e.preventDefault(); }
  else if (k === "t") { toggleIssue(__TORSO_UNDER__); e.preventDefault(); }
  else if (k === "l") { toggleIssue(__DARK__); e.preventDefault(); }
  else if (k === "s" || k === "S") { stepSuspect(k === "s" ? 1 : -1); e.preventDefault(); }
  else if (k === "c") {
    const sel = $("highlight");
    sel.selectedIndex = (sel.selectedIndex + 1) % sel.options.length;
    applyHighlight();
  }
  else if (KEYS[k]) { const c = $(KEYS[k]); c.checked = !c.checked; c.dispatchEvent(new Event("change")); }
});

// Back to the skin in the address (read before the list is drawn: drawing it can move to another skin).
const start = skins.findIndex(s => s.name === decodeURIComponent(location.hash.slice(1)));
cur = start >= 0 ? start : 0;
renderList();
updateProgress();
if (shown.includes(cur) || !shown.length) show(cur);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
