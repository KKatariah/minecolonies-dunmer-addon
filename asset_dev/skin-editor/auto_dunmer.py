#!/usr/bin/env python3
"""Automatic Dunmer pass: grey skin + red eyes, learned from the hand-edited exports.

The hand-edit tool greys whole colours (hue window minus protected colours), so every
pixel of one colour is always treated the same. The model therefore decides per unique
colour, not per pixel:
  stage 1 - each colour's hue/sat/value relative to the face, and where it sits on the texture;
  stage 2 - stage 1 plus what stage 1 decided for colours of similar hue on the same skin.
Hair is found the same way (per colour, learned from the pass-2 hair edits) and toned down
by family: red keeps its colour, blonde keeps its shading contrast, the rest is muted.
Skin is recoloured on an ash ramp in CIELAB (blue-violet shadows, grey-teal highlights) that keeps
the face's shading and stays readable against hair/clothing; eyes become solid Dunmer red.
One-off misses the model can't learn are listed per skin in overrides.json.

Never writes into an existing folder: output goes to a new versioned folder
(e.g. asset_dev/4.auto_poc, then 4.auto_poc_v2, ...).

Usage:
    python auto_dunmer.py --examples bakermale1_w knightfemale1_d ...
    python auto_dunmer.py --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ASSET_DEV = Path(__file__).resolve().parent.parent
ORIGINALS = ASSET_DEV / "1.medieval-original"
MANUAL_PASS1 = ASSET_DEV / "2.greyed-minecolonies_dunmer_skin" / "grey_export"
MANUAL_PASS2 = ASSET_DEV / "3.mcolonies_skins_p2" / "grey_export"

OVERRIDES = Path(__file__).resolve().parent / "overrides.json"

TEX_SIZE = (64, 128)  # (h, w) of the standard citizen texture
# Ash skin ramp in CIELAB LCh: (hue degrees, chroma) at the darkest and lightest skin pixels.
SKIN_SHADOW = (290, 11)
SKIN_LIGHT = (215, 6)
# Pale skins read as too light blue, so cap the skin's mean lightness (L*); darker skins keep theirs.
SKIN_L_CAP = 40
# Greying flattens faces: stretch skin lightness until its spread is at least this (max 1.6x).
SKIN_MIN_SPREAD = 7
SKIN_MAX_STRETCH = 1.6
# Skin must differ from the pixels around it by this much L*, or it blobs into hair/clothing.
SKIN_MIN_GAP = 12
SKIN_L_RANGE = (22, 58)
TARGET_RED = (0x5A, 0x09, 0x09)
EYE_WHITE_RED = (0xB0, 0x18, 0x18)
EYE_XS = (10, 13)
EYE_ROWS = (11, 12, 13, 14)
# Train only on "normal" hand edits: fewer changed pixels means the skin was (nearly) skipped,
# more means the hue window also caught clothing.
TRAIN_MIN_PIXELS = 100
TRAIN_MAX_PIXELS = 800
FACE_PRIOR = 0.6  # texture positions greyed in >60% of hand edits count as "face/skin" reference
GREY_THRESHOLD = 0.6
# (hue window in degrees, saturation window) for stage 2's "similar colours" votes.
NEIGHBOUR_WINDOWS = ((4, 0.15), (8, 0.25), (15, 1.0), (30, 1.0))
HAIR_THRESHOLD = 0.5
HAIR_MIN_EDITED_PIXELS = 10
# Hair treatments: (saturation factor, brightness factor, contrast around the hair's mean brightness).
HAIR_RED = (0.90, 0.65, 1.0)
HAIR_BLONDE = (0.70, 0.80, 1.3)
HAIR_OTHER = (0.55, 0.75, 1.0)


def load_rgba(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA")).astype(float)


def hsv(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = a[..., 0] / 255, a[..., 1] / 255, a[..., 2] / 255
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn + 1e-9
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    s = np.where(mx > 0, (mx - mn) / (mx + 1e-9), 0)
    return h, s, mx


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h = (h % 360) / 60
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], -1) * 255


_XYZ = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def to_lab(rgb: np.ndarray) -> np.ndarray:
    c = rgb / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ _XYZ.T / _WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def from_lab(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f**3 > 0.008856, f**3, (f - 16 / 116) / 7.787) * _WHITE
    c = np.clip(xyz @ np.linalg.inv(_XYZ).T, 0, 1)
    return np.where(c > 0.0031308, 1.055 * c ** (1 / 2.4) - 0.055, 12.92 * c) * 255


def hue_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a - b) % 360
    return np.minimum(d, 360 - d)


class Colours:
    """The unique colours of one texture, with per-colour features."""

    def __init__(self, a: np.ndarray, prior: np.ndarray) -> None:
        self.vis = a[..., 3] > 0
        rgb = a[self.vis][:, :3]
        self.rgb, inv = np.unique(rgb, axis=0, return_inverse=True)
        self.inv = inv.ravel()
        n = len(self.rgb)
        self.count = np.bincount(self.inv, minlength=n).astype(float)

        def mean(q: np.ndarray) -> np.ndarray:
            return np.bincount(self.inv, q[self.vis], minlength=n) / self.count

        def maximum(q: np.ndarray) -> np.ndarray:
            out = np.zeros(n)
            np.maximum.at(out, self.inv, q[self.vis])
            return out

        h, s, v = hsv(a)
        yy, xx = np.mgrid[0 : TEX_SIZE[0], 0 : TEX_SIZE[1]]
        head = np.zeros(TEX_SIZE)
        head[:16, :32] = 1
        face = (prior > FACE_PRIOR)[self.vis]
        if not face.any():
            face = np.ones_like(face)
        fh_rad = np.radians(h[self.vis][face])
        face_h = np.degrees(np.arctan2(np.sin(fh_rad).mean(), np.cos(fh_rad).mean())) % 360
        face_s, face_v = np.median(s[self.vis][face]), np.median(v[self.vis][face])
        face_rgb = np.median(rgb[face], 0)

        self.h, self.s, self.v = mean(h), mean(s), mean(v)
        self.features = np.column_stack(
            [
                self.h, self.s, self.v, self.rgb,
                hue_dist(self.h, face_h), ((self.h - face_h + 180) % 360) - 180,
                self.s - face_s, self.v - face_v,
                np.sqrt(((self.rgb - face_rgb) ** 2).sum(1)),
                np.log(self.count), mean(prior), maximum(prior), mean(head), mean(xx), mean(yy),
                np.full(n, face_h), np.full(n, face_s), np.full(n, face_v), np.full(n, n),
            ]
        )

    def labels(self, changed: np.ndarray) -> np.ndarray:
        return np.bincount(self.inv, changed[self.vis], minlength=len(self.rgb)) > 0

    def stage2_features(self, p1: np.ndarray) -> np.ndarray:
        votes = []
        dh = hue_dist(self.h[:, None], self.h[None])
        ds = np.abs(self.s[:, None] - self.s[None])
        for wh, ws in NEIGHBOUR_WINDOWS:
            w = ((dh <= wh) & (ds <= ws)) * self.count[None]
            votes.append((w * p1[None]).sum(1) / w.sum(1))
        confident = p1 > 0.8
        if confident.any():
            conf_hue = dh[:, confident].min(1)
            conf_rgb = np.sqrt(((self.rgb[:, None] - self.rgb[confident][None]) ** 2).sum(-1)).min(1)
        else:
            conf_hue, conf_rgb = np.full(len(p1), 180.0), np.full(len(p1), 400.0)
        return np.column_stack([self.features, p1, *votes, conf_hue, conf_rgb])

    def to_mask(self, per_colour: np.ndarray) -> np.ndarray:
        mask = np.zeros(TEX_SIZE, bool)
        mask[self.vis] = per_colour[self.inv]
        return mask


def gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, random_state=0)


def training_set(exclude_groups: set[str]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    items = []
    for edited in sorted(MANUAL_PASS1.glob("*.png")):
        group = edited.stem[:-2]
        orig = ORIGINALS / edited.name
        if group in exclude_groups or not orig.exists():
            continue
        a, e = load_rgba(orig), load_rgba(edited)
        if a.shape != e.shape or a.shape[:2] != TEX_SIZE:
            continue
        changed = (np.abs(e[..., :3] - a[..., :3]).max(-1) > 0) & (a[..., 3] > 0)
        if TRAIN_MIN_PIXELS <= changed.sum() <= TRAIN_MAX_PIXELS:
            items.append((group, a, changed))
    return items


class SkinModel:
    def __init__(self, exclude_groups: set[str]) -> None:
        items = training_set(exclude_groups)
        self.prior = np.mean([c for _, _, c in items], 0)
        cols = [(g, Colours(a, self.prior), c) for g, a, c in items]

        def fit_stage1(subset):
            x = np.concatenate([col.features for _, col, _ in subset])
            y = np.concatenate([col.labels(c) for _, col, c in subset])
            w = np.concatenate([col.count for _, col, _ in subset])
            return gbm().fit(x, y, sample_weight=w)

        # Stage 2 learns from out-of-fold stage-1 guesses, so it sees realistic stage-1 mistakes.
        groups = np.array(sorted({g for g, _, _ in cols}))
        x2, y2, w2 = [], [], []
        for fit_idx, _ in GroupKFold(3).split(groups, groups=groups):
            fit_groups = set(groups[fit_idx])
            m1 = fit_stage1([t for t in cols if t[0] in fit_groups])
            for g, col, c in cols:
                if g in fit_groups:
                    continue
                x2.append(col.stage2_features(m1.predict_proba(col.features)[:, 1]))
                y2.append(col.labels(c))
                w2.append(col.count)
        self.stage2 = gbm().fit(np.concatenate(x2), np.concatenate(y2), sample_weight=np.concatenate(w2))
        self.stage1 = fit_stage1(cols)
        print(f"trained on {len(items)} hand-edited skins")

    def skin_mask(self, a: np.ndarray) -> np.ndarray:
        col = Colours(a, self.prior)
        p1 = self.stage1.predict_proba(col.features)[:, 1]
        p2 = self.stage2.predict_proba(col.stage2_features(p1))[:, 1]
        return col.to_mask(p2 > GREY_THRESHOLD)


def hair_features(a: np.ndarray, skin: np.ndarray):
    """Per-colour features for non-skin colours: colour, and how much of it sits on the head."""
    vis = (a[..., 3] > 0) & ~skin
    cols, inv = np.unique(a[vis][:, :3], axis=0, return_inverse=True)
    inv = inv.ravel()
    count = np.bincount(inv, minlength=len(cols)).astype(float)

    def mean(q: np.ndarray) -> np.ndarray:
        return np.bincount(inv, q[vis], minlength=len(cols)) / count

    h, s, v = hsv(a)
    yy, xx = np.mgrid[0 : TEX_SIZE[0], 0 : TEX_SIZE[1]]
    head = ((yy < 16) & (xx < 64)).astype(float)
    head_top = ((yy < 8) & (xx >= 8) & (xx < 16)).astype(float)
    f = np.column_stack(
        [mean(h), mean(s), mean(v), cols, mean(head), mean(xx), mean(yy), np.log(count), mean(head_top)]
    )
    return f, vis, inv, count


class HairModel:
    """Learned from pass 2: pixels changed there that were neither skin (pass 1) nor the red eyes."""

    def __init__(self, exclude_groups: set[str]) -> None:
        pass2_src = MANUAL_PASS2.parent
        xs, ys, ws = [], [], []
        for edited in sorted(MANUAL_PASS2.glob("*.png")):
            orig = ORIGINALS / edited.name
            if edited.stem[:-2] in exclude_groups or not orig.exists():
                continue
            a, p1, p2 = load_rgba(orig), load_rgba(pass2_src / edited.name), load_rgba(edited)
            if not (a.shape == p1.shape == p2.shape) or a.shape[:2] != TEX_SIZE:
                continue
            skin = np.abs(p1 - a)[..., :3].max(-1) > 0
            eyes = (p2[..., 0] == TARGET_RED[0]) & (p2[..., 1] == TARGET_RED[1]) & (p2[..., 2] == TARGET_RED[2])
            hair = (np.abs(p2 - p1)[..., :3].max(-1) > 0) & ~skin & ~eyes & (a[..., 3] > 0)
            if hair.sum() < HAIR_MIN_EDITED_PIXELS:
                continue
            f, vis, inv, count = hair_features(a, skin)
            xs.append(f)
            ys.append(np.bincount(inv, hair[vis], minlength=len(count)) > 0)
            ws.append(count)
        self.model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(
            np.concatenate(xs), np.concatenate(ys), sample_weight=np.concatenate(ws)
        )
        print(f"trained hair model on {len(xs)} hand-edited skins")

    def hair_mask(self, a: np.ndarray, skin: np.ndarray) -> np.ndarray:
        f, vis, inv, _ = hair_features(a, skin)
        mask = np.zeros(TEX_SIZE, bool)
        if len(f):
            mask[vis] = (self.model.predict_proba(f)[:, 1] > HAIR_THRESHOLD)[inv]
        return mask


def tone_hair(out: np.ndarray, hair: np.ndarray) -> None:
    """Tone hair down in place; one treatment per skin, picked from the hair's overall colour."""
    if not hair.any():
        return
    h, s, v = (c[hair] for c in hsv(out))
    hr = np.radians(h)
    mean_h = np.degrees(np.arctan2(np.sin(hr).mean(), np.cos(hr).mean())) % 360
    mean_s, mean_v = np.median(s), np.median(v)
    if (mean_h < 25 or mean_h >= 330) and mean_s >= 0.55 and mean_v >= 0.4:
        sat_f, val_f, contrast = HAIR_RED
    elif 25 <= mean_h < 70 and mean_v > 0.5:
        sat_f, val_f, contrast = HAIR_BLONDE
    else:
        sat_f, val_f, contrast = HAIR_OTHER
    v = v * val_f
    v = v.mean() + (v - v.mean()) * contrast
    out[..., :3][hair] = hsv_to_rgb(h, np.clip(s * sat_f, 0, 1), np.clip(v, 0, 1))


def find_eye_row(a: np.ndarray) -> int:
    """Row where the eye pixels are darkest relative to the pixel just outside them (the eye white)."""
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]

    def contrast(y: int) -> float:
        return (lum[y, EYE_XS[0] - 1] - lum[y, EYE_XS[0]]) + (lum[y, EYE_XS[1] + 1] - lum[y, EYE_XS[1]])

    return max(EYE_ROWS, key=contrast)


def ash_skin(a: np.ndarray, out: np.ndarray, skin: np.ndarray) -> None:
    """Recolour skin in place onto the ash ramp; call after hair is toned (it checks against it)."""
    if not skin.any():
        return
    src = to_lab(a[..., :3])[..., 0][skin]
    mean, spread = src.mean(), src.std() + 1e-6
    stretch = min(SKIN_MAX_STRETCH, max(1.0, SKIN_MIN_SPREAD / spread))
    light = min(mean, SKIN_L_CAP) + (src - mean) * stretch

    near = np.zeros_like(skin)
    for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        near |= np.roll(skin, shift, (0, 1))
    border = near & ~skin & (a[..., 3] > 0)
    if border.any():
        around, mid = np.median(to_lab(out[..., :3])[..., 0][border]), np.median(light)
        if abs(mid - around) < SKIN_MIN_GAP:
            # Move away from the surroundings, keeping the current side if it stays in range.
            side = 1 if mid >= around else -1
            target = around + side * SKIN_MIN_GAP
            if not SKIN_L_RANGE[0] <= target <= SKIN_L_RANGE[1]:
                target = around - side * SKIN_MIN_GAP
            light += float(np.clip(target, *SKIN_L_RANGE)) - mid

    lo, hi = np.percentile(light, [5, 95])
    t = np.clip((light - lo) / max(hi - lo, 1e-6), 0, 1)
    hue = np.radians(SKIN_SHADOW[0] + (SKIN_LIGHT[0] - SKIN_SHADOW[0]) * t)
    chroma = SKIN_SHADOW[1] + (SKIN_LIGHT[1] - SKIN_SHADOW[1]) * t
    lab = np.stack([np.clip(light, 0, 100), chroma * np.cos(hue), chroma * np.sin(hue)], -1)
    out[..., :3][skin] = from_lab(lab)


def load_overrides() -> dict:
    return json.loads(OVERRIDES.read_text()) if OVERRIDES.exists() else {}


def colour_mask(a: np.ndarray, hexes: list[str]) -> np.ndarray:
    mask = np.zeros(TEX_SIZE, bool)
    for h in hexes:
        rgb = [int(h.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
        mask |= (a[..., :3] == rgb).all(-1) & (a[..., 3] > 0)
    return mask


def dunmerify(a: np.ndarray, model: SkinModel, hair_model: HairModel, override: dict | None = None) -> np.ndarray:
    override = override or {}
    out = a.copy()
    skin = model.skin_mask(a) | colour_mask(a, override.get("grey", []))
    skin &= ~colour_mask(a, override.get("keep", []))
    hair = hair_model.hair_mask(a, skin) if override.get("hair", True) else np.zeros(TEX_SIZE, bool)
    tone_hair(out, hair)
    ash_skin(a, out, skin)
    y = find_eye_row(a)
    for x in EYE_XS:
        if a[y, x, 3] > 0:
            out[y, x, :3] = TARGET_RED
    # The eye whites just outside the pupils go bright red, so the eyes read as solid red.
    for x in (EYE_XS[0] - 1, EYE_XS[1] + 1):
        if a[y, x, 3] > 0 and a[y, x, :3].min() >= 150:
            out[y, x, :3] = EYE_WHITE_RED
    return out


def new_output_dir(base: str) -> Path:
    candidate = ASSET_DEV / base
    n = 2
    while candidate.exists():
        candidate = ASSET_DEV / f"{base}_v{n}"
        n += 1
    candidate.mkdir(parents=True)
    return candidate


def to_img(a: np.ndarray) -> Image.Image:
    return Image.fromarray(np.round(a).astype(np.uint8), "RGBA")


def comparison_sheet(names: list[str], out_dir: Path, scale: int = 4) -> None:
    """Rows = examples; columns = original | auto | your manual edit (pass 2 if done, else pass 1)."""
    w, h = TEX_SIZE[1] * scale, TEX_SIZE[0] * scale
    gap = 8
    sheet = Image.new("RGBA", (3 * w + 4 * gap, len(names) * (h + gap) + gap), (45, 45, 45, 255))
    for row, name in enumerate(names):
        manual = MANUAL_PASS2 / name if (MANUAL_PASS2 / name).exists() else MANUAL_PASS1 / name
        for col, path in enumerate((ORIGINALS / name, out_dir / name, manual)):
            tile = Image.new("RGBA", (w, h), (255, 255, 255, 255))
            if path.exists():
                img = Image.open(path).convert("RGBA").resize((w, h), Image.Resampling.NEAREST)
                tile.alpha_composite(img)
            sheet.paste(tile, (gap + col * (w + gap), gap + row * (h + gap)))
    sheet.save(out_dir / "_comparison.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", nargs="*", help="texture names without .png, e.g. bakermale1_w")
    ap.add_argument("--all", action="store_true", help="process every original")
    ap.add_argument("--out", default="4.auto_poc", help="base name of the new output folder")
    args = ap.parse_args()

    if args.all:
        names = sorted(p.name for p in ORIGINALS.glob("*.png"))
        holdout: set[str] = set()
    else:
        names = [f"{n}.png" for n in args.examples or []]
        # Hold the examples' whole outfit group out of training so the demo is honest.
        holdout = {n[:-6] for n in names}

    overrides = load_overrides()
    model = SkinModel(holdout)
    hair_model = HairModel(holdout)
    out_dir = new_output_dir(args.out)
    for name in names:
        a = load_rgba(ORIGINALS / name)
        if a.shape[:2] != TEX_SIZE:
            print(f"skip {name}: size {a.shape[1]}x{a.shape[0]} not supported yet")
            continue
        to_img(dunmerify(a, model, hair_model, overrides.get(name[:-4]))).save(out_dir / name)
    if not args.all:
        comparison_sheet(names, out_dir)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
