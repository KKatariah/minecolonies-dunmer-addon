"""Build a self-contained HTML page for flipping through skins on 3D citizens.

Usage (from the repo root):
    python asset_dev/skin-editor/make_viewer.py                      # newest 6.ears*, else 5.auto_full*
    python asset_dev/skin-editor/make_viewer.py --skins asset_dev/5.auto_full_v9

Writes <skins>/_viewer.html (open it in a browser). Each skin is shown next to its
original from 1.medieval-original, on the same job model MineColonies uses in game
(hats, hair, chest, skirts, aprons, bags, 3px arms...), read from
citizen_models.json (make that with extract_models.py). The 3D view loads three.js
from cdnjs, so it needs an internet connection; the flat texture views don't.

Each skin can be marked Approved or Edits needed (with a comment). Reviews are kept
in the browser; Export saves them as _reviews.json, and if that file is in the
skins folder when the viewer is rebuilt, its reviews are loaded back in.
"""

import argparse
import base64
import json
from pathlib import Path

from PIL import Image

from build_mod import REPO, newest_finished_run
from extract_models import HUMANOID, RENDERED_ROOTS

HERE = Path(__file__).resolve().parent
ORIGINALS = REPO / "asset_dev" / "1.medieval-original"
MODELS = HERE / "citizen_models.json"
THREE = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"


def data_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skins", type=Path, help="folder of finished skins (default: newest 6.ears*, else 5.auto_full*)")
    parser.add_argument("--originals", type=Path, default=ORIGINALS)
    parser.add_argument("--out", type=Path, help="output file (default: <skins>/_viewer.html)")
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
    skins = sorted(p for p in skins_dir.glob("*.png") if not p.name.startswith("_"))
    if not skins:
        raise SystemExit(f"no skins in {skins_dir}")

    entries = []
    for skin in skins:
        orig = args.originals / skin.name
        with Image.open(skin) as im:
            size = im.size
        entries.append({
            "name": skin.stem,
            "size": size,
            "skin": data_uri(skin),
            "orig": data_uri(orig) if orig.exists() else None,
        })

    reviews_file = skins_dir / "_reviews.json"
    reviews = json.loads(reviews_file.read_text()).get("reviews", {}) if reviews_file.exists() else {}

    out = args.out or skins_dir / "_viewer.html"
    html = (TEMPLATE
            .replace("__TITLE__", f"Dunmer skins: {skins_dir.name}")
            .replace("__RUN__", json.dumps(skins_dir.name))
            .replace("__REVIEWS__", json.dumps(reviews))
            .replace("__THREE__", THREE)
            .replace("__MODELS__", json.dumps(models, separators=(",", ":")))
            .replace("__DATA__", json.dumps(entries)))
    out.write_text(html)
    print(f"{len(entries)} skins -> {out}")


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
aside { border-right: 1px solid var(--line); background: var(--panel); display: flex; flex-direction: column; min-height: 0; }
aside header { padding: 12px; border-bottom: 1px solid var(--line); display: grid; gap: 8px; }
input[type=search] { width: 100%; padding: 6px 8px; border: 1px solid var(--line); border-radius: 6px;
                     background: var(--bg); color: var(--text); font: inherit; }
.count { color: var(--muted); font-size: 12px; }
ul { list-style: none; margin: 0; padding: 4px 0; overflow-y: auto; flex: 1; }
li { display: flex; align-items: center; gap: 8px; padding: 3px 12px; cursor: pointer; }
li:hover { background: var(--bg); }
li.sel { background: var(--sel); font-weight: 600; }
li .mark { margin-left: auto; font-weight: 700; }
li .mark.approved, .status.approved { color: var(--ok); }
li .mark.edits, .status.edits { color: var(--warn); }
select { font: inherit; padding: 4px 6px; border: 1px solid var(--line); border-radius: 6px;
         background: var(--bg); color: var(--text); }
.tools { display: flex; gap: 6px; flex-wrap: wrap; }
.tools button { padding: 3px 10px; font-size: 12px; }
.review { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; padding: 12px 16px 0; }
.review button { font-weight: 600; }
#approve.on { background: var(--ok); border-color: var(--ok); color: #fff; }
#flag.on { background: var(--warn); border-color: var(--warn); color: #fff; }
.status { font-size: 13px; font-weight: 600; }
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
    <span class="model" id="model"></span>
    <button id="prev" title="Previous (up arrow)">&uarr; Prev</button>
    <button id="next" title="Next (down arrow)">Next &darr;</button>
    <label><input type="checkbox" id="rotate" checked> Rotate</label>
    <label><input type="checkbox" id="walk"> Walk</label>
    <label title="Job clothes and tools that only show while working"><input type="checkbox" id="work" checked> At work</label>
    <label><input type="checkbox" id="hat" checked> Hat</label>
    <label title="Carried items: logs, fish, books, torches..."><input type="checkbox" id="props"> Items</label>
  </div>
  <div class="review">
    <button id="approve" title="Looks good (y)">&#10003; Approve</button>
    <button id="flag" title="Needs fixing (n)">&#9873; Edits needed</button>
    <span class="status" id="status"></span>
    <textarea id="comment" hidden placeholder="What needs fixing? e.g. beard still brown, arm patch missed, eyes not red"></textarea>
  </div>
  <div class="pair">
    <section class="card"><h2>Original</h2><div class="stage" id="stage0"></div><div class="flat"><img id="flat0" alt=""></div></section>
    <section class="card dunmer"><h2>Dunmer</h2><div class="stage" id="stage1"></div><div class="flat"><img id="flat1" alt=""></div></section>
  </div>
  <p class="note">Models are MineColonies' own job models, so hats, hair, chests, skirts and aprons show the way they do in game.
    Drag to turn, scroll to zoom. Keys: &uarr;/&darr; or j/k to step, y approve (and go to next), n edits needed, r rotate, w walk, a at work, h hat, i items.</p>
</main>
<script id="models" type="application/json">__MODELS__</script>
<script id="data" type="application/json">__DATA__</script>
<script id="reviews" type="application/json">__REVIEWS__</script>
<script src="__THREE__"></script>
<script>
const skins = JSON.parse(document.getElementById("data").textContent);
const MODELS = JSON.parse(document.getElementById("models").textContent);
const $ = id => document.getElementById(id);
let shown = skins.map((_, i) => i), cur = 0;

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
  const [texW, texH] = def.tex || [img.width, img.height];
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
  setPose(P.body, pose("body")[0], 0, pose("body")[2], Number(c.lean(S)) || 0, 0, pose("body")[5]);
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
  const resize = () => viewers.forEach((v, n) => {
    const w = Math.min($("stage" + n).clientWidth, 420), h = Math.round(w * 1.3);
    v.renderer.setSize(w, h); v.camera.aspect = w / h; v.camera.updateProjectionMatrix();
  });
  addEventListener("resize", resize); resize();

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

// ---- reviews: {name: {status: "approved" | "edits", comment, at}} ----
const RUN = __RUN__;
const STORE = "dunmer-reviews:" + RUN;
let reviews = JSON.parse(document.getElementById("reviews").textContent);
try { Object.assign(reviews, JSON.parse(localStorage.getItem(STORE) || "{}")); } catch {}
function saveReviews() {
  try { localStorage.setItem(STORE, JSON.stringify(reviews)); } catch {}
}
const MARK = { approved: "\u2713", edits: "\u2691" };

function setStatus(status) {
  const name = skins[cur].name, r = reviews[name] || {};
  if (r.status === status) delete r.status; else r.status = status;
  r.at = new Date().toISOString();
  if (r.status || r.comment) reviews[name] = r; else delete reviews[name];
  saveReviews(); showReview(); markItem(cur); updateProgress();
}

function showReview() {
  const r = reviews[skins[cur].name] || {};
  $("approve").classList.toggle("on", r.status === "approved");
  $("flag").classList.toggle("on", r.status === "edits");
  $("status").className = "status " + (r.status || "");
  $("status").textContent = r.status === "approved" ? "Approved" : r.status === "edits" ? "Flagged for edits" : "Not reviewed";
  $("comment").hidden = r.status !== "edits" && !r.comment;
  $("comment").value = r.comment || "";
}

function markItem(i) {
  const li = document.querySelector(`li[data-i="${i}"] .mark`);
  if (!li) return;
  const st = reviews[skins[i].name]?.status;
  li.className = "mark " + (st || ""); li.textContent = MARK[st] || "";
}

function updateProgress() {
  const vals = Object.values(reviews);
  const ok = vals.filter(r => r.status === "approved").length, ed = vals.filter(r => r.status === "edits").length;
  $("progress").textContent = `${ok} approved \u00b7 ${ed} edits needed \u00b7 ${skins.length - ok - ed} to go`;
}

$("approve").onclick = () => {
  const was = reviews[skins[cur].name]?.status;
  setStatus("approved");
  if (was !== "approved") step(1);
};
$("flag").onclick = () => { setStatus("edits"); if (!$("comment").hidden) $("comment").focus(); };
$("comment").addEventListener("input", () => {
  const name = skins[cur].name, r = reviews[name] || {};
  r.comment = $("comment").value; r.at = new Date().toISOString();
  if (r.status || r.comment) reviews[name] = r; else delete reviews[name];
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
    Object.assign(reviews, data.reviews || data);
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

function renderList() {
  const words = $("q").value.toLowerCase().split(/\s+/).filter(Boolean);
  const want = $("filter").value;
  shown = skins.map((s, i) => i).filter(i => words.every(w => skins[i].name.includes(w)) &&
    (want === "all" || (reviews[skins[i].name]?.status || "none") === want));
  $("list").innerHTML = shown.map(i =>
    `<li data-i="${i}"><span class="face" style="background-image:url(${skins[i].skin});background-size:${skins[i].size[0] * 3}px ${skins[i].size[1] * 3}px"></span>${skins[i].name}<span class="mark"></span></li>`).join("");
  shown.forEach(markItem);
  $("count").textContent = `${shown.length} of ${skins.length} skins`;
  if (shown.length && !shown.includes(cur)) show(shown[0]);
  else document.querySelectorAll("li").forEach(li => li.classList.toggle("sel", +li.dataset.i === cur));
}

function step(d) {
  if (!shown.length) return;
  const at = shown.indexOf(cur);
  show(shown[(at + d + shown.length) % shown.length]);
}

$("list").addEventListener("click", e => { const li = e.target.closest("li"); if (li) show(+li.dataset.i); });
$("q").addEventListener("input", renderList);
$("filter").addEventListener("change", renderList);
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
  else if (k === "y") $("approve").click();
  else if (k === "n") { $("flag").click(); e.preventDefault(); }
  else if (KEYS[k]) { const c = $(KEYS[k]); c.checked = !c.checked; }
});

renderList();
updateProgress();
const start = skins.findIndex(s => s.name === decodeURIComponent(location.hash.slice(1)));
show(start >= 0 ? start : 0);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
