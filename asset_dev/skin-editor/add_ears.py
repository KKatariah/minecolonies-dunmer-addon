#!/usr/bin/env python3
"""Paint pointed elf ears on the sides of the head where the ear would be seen.

Usage (from the repo root):
    python asset_dev/skin-editor/add_ears.py                          # newest asset_dev/5.auto_full*
    python asset_dev/skin-editor/add_ears.py --skins asset_dev/5.auto_full_v10

Reads a finished run and writes a copy with ears to a new versioned folder
(asset_dev/6.ears, then 6.ears_v2, ...); the run itself is never changed.

An ear goes on a side of the head only when:
  - what shows there is skin or hair, by the Dunmer pass's own skin and hair masks
    (not a hood or hat painted on the head). Where the head's overlay layer has hair
    over the spot, the ear is painted on the overlay too, so it sits over the hair;
  - nothing else covers it in game: a ray from the side through the citizen's own
    model (job hats, hoods, hair pieces, using the texture's transparency) has to
    reach the head without hitting an opaque pixel.
Recomputing the masks retrains the pipeline's models, so a full run takes a minute or two.
overrides.json can force it per skin or outfit: "ears": true / false / "left" / "right".
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

import auto_dunmer as ad
from build_mod import newest_full_run

HERE = Path(__file__).resolve().parent
MODELS = HERE / "citizen_models.json"
OVERLAY_DX = 32  # the head overlay layer is the head's layout shifted right by 32

# The ear on the head's side face, 8x8, rows top to bottom, columns from the face
# (front) to the back of the head. S = skin, D = shadow inside the ear.
EAR = [
    "........",
    "........",
    "........",
    "....S...",
    "...SS...",
    "..SDS...",
    "..SD....",
    "..S.....",
]
SHADE = {"S": 1.0, "D": 0.72}
CHANGED = 12        # RGB change that marks a pixel the Dunmer pass recoloured
MIN_HEAD = 0.75     # share of the ear's pixels that must show skin or hair
MATCH = 30          # RGB distance at which an overlay colour counts as the head's hair/skin
PALE_MIN, PALE_SPREAD = 150, 30  # "hair" lighter than this and this grey is cloth
OPAQUE = 25         # alpha above this hides the ear
STATE = {"work": True, "hat": True, "props": False}  # how the citizen is pictured


def ear_pixels(side):
    """Texture (x, y) and shade letter for each ear pixel. The citizen's right side
    is the head's west face (x 0-7, front at x=7); the left is the east face (x 16-23,
    front at x=16)."""
    out = []
    for row, line in enumerate(EAR):
        for col, ch in enumerate(line):
            if ch != ".":
                x = 7 - col if side == "right" else 16 + col
                out.append((x, 8 + row, ch))
    return out


# ---- model geometry (same rules as the viewer / vanilla ModelPart) ----

def rot(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def js_bool(expr):
    """Evaluate the viewer's visibility expressions for STATE."""
    e = expr.replace("&&", " and ").replace("||", " or ")
    e = re.sub(r"!(?!=)", " not ", e)
    e = e.replace("true", "True").replace("false", "False")
    e = re.sub(r"S\.(\w+)", lambda m: repr(STATE[m.group(1)]), e)
    return bool(eval(e, {"__builtins__": {}}))


def cube_faces(c):
    """The six quads of a box as (corners, uvs) in part space, like ModelPart.Cube."""
    u, v, x, y, z, w, h, d, (gx, gy, gz), mirror = c
    x0, x1 = x - gx, x + w + gx
    y0, y1, z0, z1 = y - gy, y + h + gy, z - gz, z + d + gz
    if mirror:
        x0, x1 = x1, x0
    v7, v0, v1, v2 = (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)
    v3, v4, v5, v6 = (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)
    f4, f5, f6, f7, f8, f9 = u, u + d, u + d + w, u + d + w + w, u + d + w + d, u + d + w + d + w
    f10, f11, f12 = v, v + d, v + d + h
    faces = [
        ((v4, v3, v7, v0), f5, f10, f6, f11), ((v1, v2, v6, v5), f6, f11, f7, f10),
        ((v7, v3, v6, v2), f4, f11, f5, f12), ((v0, v7, v2, v1), f5, f11, f6, f12),
        ((v4, v0, v1, v5), f6, f11, f8, f12), ((v3, v4, v5, v6), f8, f11, f9, f12),
    ]
    out = []
    for vs, u1, va, u2, vb in faces:
        out.append((np.array(vs, float), np.array([(u2, va), (u1, va), (u1, vb), (u2, vb)], float)))
    return out


class Occluders:
    """Every visible box face of a citizen model, in model space."""

    def __init__(self, model):
        parts = {p["name"]: p for p in model["parts"]}
        hidden = {path[-1] for path in model["hidden"]}
        for path, js in model["rules"]:
            if not js_bool(js):
                hidden.add(path[-1])
            else:
                hidden.discard(path[-1])
        self.tex = model["tex"]
        self.faces = []

        def walk(name, M, t):
            p = parts[name]
            if name in hidden:
                return
            x, y, z, rx, ry, rz = p["pose"]
            if p["parent"] is None and name in ("head", "hat"):
                rx = ry = 0  # setupAnim: looking straight ahead
            R = M @ rot(rx, ry, rz)
            T = t + M @ np.array([x, y, z])
            for c in p["cubes"]:
                if name in ("head", "hat") and c[2:8] == [-4, -8, -4, 8, 8, 8] and c[:2] in ([0, 0], [32, 0]):
                    continue  # the head and its overlay layer: what the ear is painted on
                for corners, uvs in cube_faces(c):
                    self.faces.append(((R @ corners.T).T + T, uvs))
            for child in (n for n, q in parts.items() if q["parent"] == name):
                walk(child, R, T)

        for root in model["rendered"]:
            if root in parts and parts[root]["parent"] is None:
                walk(root, np.eye(3), np.zeros(3))

    def blocked(self, img, start, end):
        """Does the segment start->end pass through an opaque pixel of any face?"""
        h, w = img.shape[:2]
        tw, th = self.tex or (w, h)
        d = end - start
        for corners, uvs in self.faces:
            p1, a, b = corners[1], corners[0] - corners[1], corners[2] - corners[1]
            n = np.cross(a, b)
            denom = n @ d
            if abs(denom) < 1e-9:
                continue
            t = n @ (p1 - start) / denom
            if not 0 < t < 1:
                continue
            q = start + t * d - p1
            aa, bb, ab = a @ a, b @ b, a @ b
            det = aa * bb - ab * ab
            if det < 1e-9:
                continue
            s = (bb * (q @ a) - ab * (q @ b)) / det
            r = (aa * (q @ b) - ab * (q @ a)) / det
            if not (0 <= s <= 1 and 0 <= r <= 1):
                continue
            uv = uvs[1] + s * (uvs[0] - uvs[1]) + r * (uvs[2] - uvs[1])
            px = min(int(uv[0] / tw * w), w - 1)
            py = min(int(uv[1] / th * h), h - 1)
            if img[py, px, 3] > OPAQUE:
                return True
        return False


def model_for(models, name):
    m = re.match(r"(.*?)(female|male)\d+", name)
    cls = m and models["types"].get(m.group(1), {}).get(m.group(2))
    return models["models"].get(cls) if cls else None


def ear_point(x, y):
    """Model-space point on the head's side for texture pixel (x, y) and the
    direction to look from (outside the head)."""
    py = -8 + (y - 8) + 0.5
    if x < 8:   # west face: x = -4, z runs from front (tex x=7) to back (x=0)
        return np.array([-4.0, py, 4 - (x + 0.5)]), np.array([-1.0, 0, 0])
    return np.array([4.0, py, -4 + (x - 16 + 0.5)]), np.array([1.0, 0, 0])


def skin_tone(orig, out):
    """Median recoloured colour in the lower face (cheeks/mouth), as the ear colour."""
    face = out[12:16, 9:15, :3].reshape(-1, 3).astype(int)
    was = orig[12:16, 9:15, :3].reshape(-1, 3).astype(int)
    changed = np.abs(face - was).max(1) > CHANGED
    pick = face[changed] if changed.sum() >= 3 else face
    return np.median(pick, axis=0)


def head_masks(name, model, hair_model, overrides):
    """Skin | hair mask of the original, the way auto_dunmer.py's run decided it."""
    full = ad.load_rgba(ad.ORIGINALS / f"{name}.png")
    override = ad.override_for(overrides, name)
    by_hand = ad.hand_greyed(f"{name}.png", full) if override.get("hand", True) else None
    a = ad.frame(full)
    _, skin, hair = ad.dunmerify_with_masks(a, model, hair_model, override, by_hand)
    # Near-white, colourless "hair" is a hood or veil (healer, beekeeper), not hair.
    rgb = a[..., :3]
    pale = (rgb.min(-1) > PALE_MIN) & (rgb.max(-1) - rgb.min(-1) < PALE_SPREAD)
    head = skin | (hair & ~pale)
    # The hair model only knows the base layer, so overlay pixels count as hair (or
    # skin) when their colour is close to the hair/skin on the base head.
    base = a[:16, :32, :3][head[:16, :32] & (a[:16, :32, 3] > OPAQUE)].reshape(-1, 3).astype(float)
    if len(base):
        layer = a[:16, 32:64, :3].astype(float)
        dist = np.sqrt(((layer[:, :, None, :] - base[None, None]) ** 2).sum(-1)).min(-1)
        head[:16, 32:64] |= (dist <= MATCH) & (a[:16, 32:64, 3] > OPAQUE)
    return head


def decide(out, head, occ, force):
    """Which sides get an ear (and whether on the overlay too), and why the others don't."""
    sides, why = {}, {}
    for side in ("right", "left"):
        if force is False or force in ("left", "right") and force != side:
            why[side] = "override"
            continue
        px = ear_pixels(side)
        over = [out[y, x + OVERLAY_DX, 3] > OPAQUE for x, y, _ in px]
        if force is True or force == side:
            sides[side] = over
            continue
        shows = [head[y, x + OVERLAY_DX] if o else head[y, x] for (x, y, _), o in zip(px, over)]
        if np.mean(shows) < MIN_HEAD:
            why[side] = "hood or hat there" if any(over) else "not skin or hair there"
            continue
        if occ is not None:
            covered = 0
            for x, y, _ in px:
                p, look = ear_point(x, y)
                covered += occ.blocked(out, p + look * 30, p + look * 0.6)
            if covered > 1:
                why[side] = f"covered by the model ({covered}/{len(px)} pixels)"
                continue
        sides[side] = over
    return sides, why


def paint(out, tone, sides):
    for side, over in sides.items():
        for (x, y, ch), o in zip(ear_pixels(side), over):
            colour = np.clip(tone * SHADE[ch], 0, 255)
            out[y, x, :3], out[y, x, 3] = colour, 255
            if o:
                out[y, x + OVERLAY_DX, :3], out[y, x + OVERLAY_DX, 3] = colour, 255


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skins", type=Path, help="finished run (default: newest 5.auto_full*)")
    parser.add_argument("names", nargs="*", help="only these skins (default: all)")
    args = parser.parse_args()

    src = (args.skins or newest_full_run()).resolve()
    models = json.loads(MODELS.read_text())
    overrides = ad.load_overrides()
    print("training the skin and hair models...")
    skin_model, hair_model = ad.SkinModel(set()), ad.HairModel(set())
    out_dir = ad.new_output_dir("6.ears")

    skins = sorted(p for p in src.glob("*.png") if not p.name.startswith("_"))
    if args.names:
        skins = [p for p in skins if p.stem in args.names]
    report, counts = [], {0: 0, 1: 0, 2: 0}
    for path in skins:
        name = path.stem
        out = np.array(Image.open(path).convert("RGBA"))
        orig_path = ad.ORIGINALS / path.name
        if not orig_path.exists():
            shutil.copy2(path, out_dir / path.name)
            report.append(f"{name:28} {'-':12} no original")
            counts[0] += 1
            continue
        orig = np.array(Image.open(orig_path).convert("RGBA"))
        head = head_masks(name, skin_model, hair_model, overrides)
        model = model_for(models, name)
        occ = Occluders(model) if model else None
        force = ad.override_for(overrides, name).get("ears")
        sides, why = decide(out, head, occ, force)
        if sides:
            paint(out, skin_tone(orig, out), sides)
            Image.fromarray(out).save(out_dir / path.name)
        else:
            shutil.copy2(path, out_dir / path.name)
        counts[len(sides)] += 1
        notes = "; ".join(f"{s}: {w}" for s, w in why.items())
        placed = "+".join(s + ("*" if any(o) else "") for s, o in sides.items())
        report.append(f"{name:28} {placed or '-':12} {notes}")

    (out_dir / "_ears.txt").write_text(
        f"Ears added to {src.name}\n"
        f"both sides: {counts[2]}, one side: {counts[1]}, none: {counts[0]}\n"
        f"* = also painted on the head overlay layer, over hair\n\n" + "\n".join(report) + "\n")
    print(f"{len(skins)} skins from {src.name} -> {out_dir.name}: "
          f"{counts[2]} both ears, {counts[1]} one, {counts[0]} none (see _ears.txt)")


if __name__ == "__main__":
    main()
