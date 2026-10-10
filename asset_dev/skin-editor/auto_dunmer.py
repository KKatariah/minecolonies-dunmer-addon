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
Skins you already greyed by hand (2.greyed-.../grey_export) use your greying to decide what is skin;
the model only decides for skins you skipped (use --no-hand to test the model everywhere).

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
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ASSET_DEV = Path(__file__).resolve().parent.parent
ORIGINALS = ASSET_DEV / "1.medieval-original"
MANUAL_PASS1 = ASSET_DEV / "2.greyed-minecolonies_dunmer_skin" / "grey_export"
MANUAL_PASS2 = ASSET_DEV / "3.mcolonies_skins_p2" / "grey_export"
# Your originals and hand edits (and their backups) are only ever read, never written.
PROTECTED = (ORIGINALS, MANUAL_PASS1.parent, MANUAL_PASS2.parent)

OVERRIDES = Path(__file__).resolve().parent / "overrides.json"

# Backup check against your hand edits: a skin is flagged for review when the tool and your pass 1
# disagree on more than this many visibly-changed pixels. Skins you barely edited by hand
# (fewer than TRAIN_MIN_PIXELS changes) aren't checked, since there's nothing to compare with.
REVIEW_MIN_DIFF = 40
REVIEW_VISIBLE = 15  # RGB change that counts as visible

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
# Below the head, colours much darker (L*) than the face's skin are clothing (dark skirts, boots,
# belts), even when they share its hue.
SKIN_MAX_DARKER = 35
TARGET_RED = (0x5A, 0x09, 0x09)
EYE_WHITE_RED = (0xA8, 0x2C, 0x27)
EYE_XS = (10, 13)
EYE_ROWS = (11, 12, 13, 14)
# Child (and a few adult) faces have eyes two pixels tall: a row counts as eye if its contrast is
# at least this, and at least EYE_ROW_RATIO of the strongest row's.
EYE_MIN_CONTRAST = 100
EYE_ROW_RATIO = 0.5
# A second eye row also needs real eye whites beside it (bright, near-grey), so skin next to a
# dark mask or scarf doesn't count as an eye.
EYE_WHITE_MAX_SAT = 0.15
EYE_WHITE_MIN_VALUE = 0.6
# Two-row eyes: the bottom row gets the base red and the top row is lifted by the original's
# lightness difference between the rows, capped so it doesn't wash out. Never darker than the
# base, or the eye merges into dark lashes above it.
EYE_MAX_LIFT = 12
# Train only on "normal" hand edits: fewer changed pixels means the skin was (nearly) skipped,
# more means the hue window also caught clothing.
TRAIN_MIN_PIXELS = 100
TRAIN_MAX_PIXELS = 800
FACE_PRIOR = 0.6  # texture positions greyed in >60% of hand edits count as "face/skin" reference
GREY_THRESHOLD = 0.6
# (hue window in degrees, saturation window) for stage 2's "similar colours" votes.
NEIGHBOUR_WINDOWS = ((4, 0.15), (8, 0.25), (15, 1.0), (30, 1.0))
HAIR_THRESHOLD = 0.5
# Hair below the head hangs in connected locks; specks this small there are dress fabric that
# happens to match the hair colour.
HAIR_MAX_SPECK = 3
# The hair model decides colour by colour and misses some shades of the same hair. On the inner
# head layer, hair grows into touching untouched pixels within this colour distance (CIELAB) of
# hair already found. Not on the outer layer (mostly caps and hats), and never into skin: reddish
# hair and skin are too close to tell apart by colour (use the hair_colours override for those).
HAIR_GROW_DIST = 10
# Reddish hair can be greyed as skin. Where the hand edits almost never greyed (crown, back and
# upper sides of the head: below this share of skins), small greyed patches touching hair on at
# least 2 sides go back to hair. Bald scalps are one big patch, so they stay skin; beards sit lower
# where skin is common, so greyed beard hair stays grey and reads as stubble.
HAIR_ZONE_MAX_SKIN = 0.17
HAIR_ZONE_MAX_PATCH = 20
# On skins greyed by hand only stray specks go back (the hue-window tool caught some hair shades);
# bigger greyed areas there are your call (e.g. a bald crown).
HAIR_ZONE_MAX_PATCH_HAND = 6
# Very dark hair often isn't toned at all, so stray greyed shades in it can't be found by touching
# hair. In the hair area, small greyed patches this much darker (L*) than the face go back to their
# original colour; bald patches and ears are about as light as the face, so they stay grey, and a
# whole area of hair you greyed by hand stays as you did it.
HAIR_ZONE_DARKER = 20
# The middle of the face (below the eyes). If skin covers less than this share of it, the face was
# missed: a hand edit that misses it is ignored for that skin, and on a model-decided skin the warm
# skin colours found there are greyed (only when the eyes are clear, so masked faces stay as they are).
FACE_MIN_COVER = 0.4
# Model-decided skins: on the head and neck, in places that are usually skin (prior >=
# SKIN_CLAIM_PRIOR, e.g. under the chin, the face, the neck), pixels the hair model took or left alone become skin when they're
# within this colour distance (CIELAB) of the face's skin.
SKIN_CLAIM_PRIOR = 0.4
# Dark hair on dark skins often gets greyed wholesale, by the model or the hand-edit hue window.
# In the hair area (prior < 0.3), a greyed pixel is dark hair if it's this much darker (L*) than the
# face AND has under GREYED_HAIR_CHROMA of the face's colour strength (black/very dark brown hair is
# nearly colourless; a shaded bald scalp keeps ~75-85% of the face's colour). When most greyed pixels
# there look like that, they go back to hair. Mid-brown hair on brown skin looks just like a bald scalp
# by these numbers, so that needs a keep override.
GREYED_HAIR_DARKER = 12
GREYED_HAIR_CHROMA = 0.55
GREYED_HAIR_SHARE = 0.5
SKIN_CLAIM_DIST = 15
# Eyebrows: non-skin pixels in the two rows above the eyes are drawn this much darker (L*) than the
# greyed face around them, with muted colour, so they read against grey skin and red eyes.
BROW_DARKER = 18
BROW_CHROMA = 0.4
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
        body = mask.copy()
        body[:16, :64] = False
        labels, n = ndimage.label(body)
        if n:
            sizes = ndimage.sum(body, labels, range(1, n + 1))
            mask &= ~np.isin(labels, 1 + np.nonzero(sizes <= HAIR_MAX_SPECK)[0])
        return mask


def tone_hair(out: np.ndarray, hair: np.ndarray, sample: np.ndarray | None = None) -> None:
    """Tone hair down in place; one treatment per skin, picked from the hair's overall colour
    (from `sample` if given, so grown-in edge shades don't change which treatment is chosen)."""
    if not hair.any():
        return
    sample = hair if sample is None or not sample.any() else sample
    h, s, v = (c[sample] for c in hsv(out))
    hr = np.radians(h)
    mean_h = np.degrees(np.arctan2(np.sin(hr).mean(), np.cos(hr).mean())) % 360
    mean_s, mean_v = np.median(s), np.median(v)
    h, s, v = (c[hair] for c in hsv(out))
    if (mean_h < 25 or mean_h >= 330) and mean_s >= 0.55 and mean_v >= 0.4:
        sat_f, val_f, contrast = HAIR_RED
    elif 25 <= mean_h < 70 and mean_v > 0.5:
        sat_f, val_f, contrast = HAIR_BLONDE
    else:
        sat_f, val_f, contrast = HAIR_OTHER
    v = v * val_f
    v = v.mean() + (v - v.mean()) * contrast
    out[..., :3][hair] = hsv_to_rgb(h, np.clip(s * sat_f, 0, 1), np.clip(v, 0, 1))


def grow_hair(a: np.ndarray, skin: np.ndarray, hair: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Grow hair on the inner head layer into similar touching shades; skin and `fixed` pixels are never taken."""
    if hair.sum() < 5:
        return hair
    lab = to_lab(a[..., :3])
    inner_head = np.zeros(TEX_SIZE, bool)
    inner_head[:16, :32] = True
    allowed = inner_head & (a[..., 3] > 0) & ~skin & ~fixed
    hair = hair.copy()
    while True:
        near = np.zeros(TEX_SIZE, bool)
        for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            near |= np.roll(hair, shift, (0, 1))
        cand = near & ~hair & allowed
        if not cand.any():
            break
        hair_colours = np.unique(lab[hair].round(1), axis=0)
        to_hair = np.sqrt(((lab[cand][:, None] - hair_colours[None]) ** 2).sum(-1)).min(1)
        take = to_hair < HAIR_GROW_DIST
        if not take.any():
            break
        ys, xs = np.nonzero(cand)
        hair[ys[take], xs[take]] = True
    return hair


def reclaim_hair(a: np.ndarray, skin: np.ndarray, hair: np.ndarray, prior: np.ndarray, fixed: np.ndarray,
                 max_patch: int = HAIR_ZONE_MAX_PATCH):
    """Give small greyed patches in the usual hair area back to the hair (see HAIR_ZONE_MAX_SKIN)."""
    inner_head = np.zeros(TEX_SIZE, bool)
    inner_head[:16, :32] = True
    zone_skin = skin & inner_head & (prior < HAIR_ZONE_MAX_SKIN) & ~fixed
    if hair.sum() < 5 or not zone_skin.any():
        return skin, hair
    hair_sides = sum(np.roll(hair, shift, (0, 1)).astype(int) for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    labels, n = ndimage.label(zone_skin)
    index = range(1, n + 1)
    sizes = ndimage.sum(zone_skin, labels, index)
    most_hair_sides = ndimage.maximum(hair_sides, labels, index)
    patches = [i + 1 for i in range(n) if sizes[i] <= max_patch and most_hair_sides[i] >= 2]
    back = np.isin(labels, patches)
    face = skin & face_middle(a)
    if face.any():
        lightness = to_lab(a[..., :3])[..., 0]
        too_dark = ndimage.mean(lightness, labels, index) < np.median(lightness[face]) - HAIR_ZONE_DARKER
        dark = np.isin(labels, [i + 1 for i in range(n) if sizes[i] <= max_patch and too_dark[i]])
        skin = skin & ~dark
    return skin & ~back, hair | back


def eye_contrast(a: np.ndarray) -> float:
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return max((lum[y, EYE_XS[0] - 1] - lum[y, EYE_XS[0]]) + (lum[y, EYE_XS[1] + 1] - lum[y, EYE_XS[1]]) for y in EYE_ROWS)


def find_eye_rows(a: np.ndarray) -> list[int]:
    """Rows where the eye pixels are dark against the pixel just outside them (the eye white).

    Takes the topmost strong row (a mouth with teeth below can contrast as much as the eyes),
    plus the row under it when the eyes are two pixels tall.
    """
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]

    def contrast(y: int) -> float:
        return (lum[y, EYE_XS[0] - 1] - lum[y, EYE_XS[0]]) + (lum[y, EYE_XS[1] + 1] - lum[y, EYE_XS[1]])

    _, s, v = hsv(a)

    def has_whites(y: int) -> bool:
        return all(s[y, x] <= EYE_WHITE_MAX_SAT and v[y, x] >= EYE_WHITE_MIN_VALUE for x in (EYE_XS[0] - 1, EYE_XS[1] + 1))

    best = max(EYE_ROWS, key=contrast)
    floor = max(EYE_MIN_CONTRAST, EYE_ROW_RATIO * contrast(best))
    strong = [y for y in EYE_ROWS if contrast(y) >= floor]
    if not strong:
        return [best]
    top = strong[0]
    return [top, top + 1] if top + 1 in strong and has_whites(top + 1) else [top]


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


def drop_dark_clothing(a: np.ndarray, skin: np.ndarray) -> np.ndarray:
    face = np.zeros(TEX_SIZE, bool)
    face[8:16, 8:16] = True
    if not (skin & face).any():
        return skin
    lightness = to_lab(a[..., :3])[..., 0]
    too_dark = lightness < np.median(lightness[skin & face]) - SKIN_MAX_DARKER
    too_dark[:16, :64] = False  # head
    return skin & ~too_dark


def load_overrides() -> dict:
    return json.loads(OVERRIDES.read_text()) if OVERRIDES.exists() else {}


def override_for(overrides: dict, name: str) -> dict:
    """Entries for the outfit (e.g. "cookfemale1", all variants) combined with the skin's own."""
    merged: dict = {}
    for entry in (overrides.get(name[:-2], {}), overrides.get(name, {})):
        for key, value in entry.items():
            merged[key] = merged.get(key, []) + value if isinstance(value, list) else value
    return merged


def area_mask(areas: list[list[int]]) -> np.ndarray:
    mask = np.zeros(TEX_SIZE, bool)
    for x0, y0, x1, y1 in areas:
        mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


def colour_mask(a: np.ndarray, hexes: list[str]) -> np.ndarray:
    mask = np.zeros(TEX_SIZE, bool)
    for h in hexes:
        rgb = [int(h.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
        mask |= (a[..., :3] == rgb).all(-1) & (a[..., 3] > 0)
    return mask


def face_middle(a: np.ndarray) -> np.ndarray:
    middle = np.zeros(TEX_SIZE, bool)
    middle[max(find_eye_rows(a)) + 1 : 16, 9:15] = True
    return middle & (a[..., 3] > 0)


def face_cover(a: np.ndarray, skin: np.ndarray) -> float:
    middle = face_middle(a)
    return (skin & middle).sum() / max(middle.sum(), 1)


def skin_like_face(a: np.ndarray) -> np.ndarray:
    """Pixels in the middle of the face with a warm, skin-like colour (not a mask, scarf or beard)."""
    h, s, v = hsv(a)
    return face_middle(a) & ((h < 50) | (h > 330)) & (s > 0.2) & (v > 0.25)


def face_missed(a: np.ndarray, skin: np.ndarray) -> bool:
    """A bare face (clear eyes, mostly skin-coloured middle) that `skin` leaves mostly un-greyed."""
    bare = skin_like_face(a).sum() >= 0.5 * face_middle(a).sum()
    return bare and eye_contrast(a) >= EYE_MIN_CONTRAST and face_cover(a, skin) < FACE_MIN_COVER


def missed_face_colours(a: np.ndarray, skin: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Pixels sharing a warm skin-like colour with the face, if the face was missed; only where skin
    usually is (prior >= 0.2), so hoods and hair of the same colour stay as they are."""
    if not face_missed(a, skin):
        return np.zeros(TEX_SIZE, bool)
    # Whole face from the eyes down (front and sides), not just the middle, so the face comes out even.
    h, s, v = hsv(a)
    face = np.zeros(TEX_SIZE, bool)
    face[min(find_eye_rows(a)) - 1 : 16, :32] = True
    face &= (a[..., 3] > 0) & ((h < 50) | (h > 330)) & (s > 0.2) & (v > 0.25)
    middle_lightness = np.median(to_lab(a[..., :3])[..., 0][skin_like_face(a)])
    face &= np.abs(to_lab(a[..., :3])[..., 0] - middle_lightness) < 25
    colours = {"#%02x%02x%02x" % tuple(c) for c in a[face][:, :3].astype(int)}
    return colour_mask(a, sorted(colours)) & (prior >= 0.2)


def claim_skin(a: np.ndarray, skin: np.ndarray, hair: np.ndarray, prior: np.ndarray, fixed: np.ndarray):
    """Face shades the skin model missed (and the hair model may have taken) in usual-skin places."""
    face = skin & face_middle(a)
    if not face.any():
        return skin, hair
    lab = to_lab(a[..., :3])
    face_colours = np.unique(lab[face].round(1), axis=0)
    eyes = np.zeros(TEX_SIZE, bool)
    for y in find_eye_rows(a):
        eyes[y, EYE_XS[0] - 1 : EYE_XS[1] + 2] = True
    head_and_neck = np.zeros(TEX_SIZE, bool)
    head_and_neck[:16, :32] = True
    head_and_neck[16:20, 16:40] = True  # top of the torso, around the neck
    cand = ~skin & (a[..., 3] > 0) & (prior >= SKIN_CLAIM_PRIOR) & head_and_neck & ~fixed & ~eyes
    if not cand.any():
        return skin, hair
    dist = np.sqrt(((lab[cand][:, None] - face_colours[None]) ** 2).sum(-1)).min(1)
    take = np.zeros(TEX_SIZE, bool)
    take[cand] = dist < SKIN_CLAIM_DIST
    return skin | take, hair & ~take


def ungrey_dark_hair(a: np.ndarray, skin: np.ndarray, prior: np.ndarray) -> np.ndarray:
    face = skin & face_middle(a)
    top = np.zeros(TEX_SIZE, bool)
    top[:16, :] = True  # head and the extra accessory parts beside it
    greyed = skin & top & (prior < 0.3)
    if greyed.sum() < 10 or not face.any():
        return skin
    lab = to_lab(a[..., :3])
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    hair_like = greyed & (lab[..., 0] < np.median(lab[..., 0][face]) - GREYED_HAIR_DARKER) & (
        chroma < GREYED_HAIR_CHROMA * np.median(chroma[face])
    )
    if hair_like.sum() < GREYED_HAIR_SHARE * greyed.sum():
        return skin
    return skin & ~hair_like


def brow_area(a: np.ndarray) -> np.ndarray:
    area = np.zeros(TEX_SIZE, bool)
    top = min(find_eye_rows(a))
    area[max(top - 2, 8) : top, 8:16] = True
    return area


def paint_brows(a: np.ndarray, out: np.ndarray, skin: np.ndarray) -> None:
    """Darken and mute the eyebrows so they stand out from the grey face and the red eyes."""
    brows = brow_area(a) & ~skin & (a[..., 3] > 0)
    face = skin & face_middle(a)
    if not brows.any() or not face.any():
        return
    out_lab = to_lab(out[..., :3])
    limit = np.median(out_lab[..., 0][face]) - BROW_DARKER
    lab = out_lab[brows]
    lab[:, 0] = np.minimum(lab[:, 0], limit)
    lab[:, 1:] *= BROW_CHROMA
    out[..., :3][brows] = np.round(from_lab(lab))


def hand_greyed(name: str, a: np.ndarray) -> np.ndarray | None:
    """Your pass-1 greying for this skin, if you edited it normally (not skipped or over-greyed)."""
    path = MANUAL_PASS1 / name
    if not path.exists():
        return None
    manual = load_rgba(path)
    if manual.shape != a.shape:
        return None
    greyed = (np.abs(manual - a)[..., :3].max(-1) > 0) & (a[..., 3] > 0)
    return greyed if TRAIN_MIN_PIXELS <= greyed.sum() <= TRAIN_MAX_PIXELS else None


def dunmerify(a: np.ndarray, model: SkinModel, hair_model: HairModel, override: dict | None = None,
              by_hand: np.ndarray | None = None) -> np.ndarray:
    return dunmerify_with_masks(a, model, hair_model, override, by_hand)[0]


def dunmerify_with_masks(a: np.ndarray, model: SkinModel, hair_model: HairModel, override: dict | None = None,
                         by_hand: np.ndarray | None = None):
    """Returns (recoloured texture, skin mask, hair mask).

    by_hand: your pass-1 greying (see hand_greyed). When given it decides what is skin instead of
    the model; overrides and the dark-clothing rule still apply on top.
    """
    override = override or {}
    out = a.copy()
    forced_grey = colour_mask(a, override.get("grey", []))
    forced_hair = colour_mask(a, override.get("hair_colours", []))
    keep = colour_mask(a, override.get("keep", []))
    untouched = area_mask(override.get("keep_area", []))
    model_skin = model.skin_mask(a)
    if by_hand is not None and face_missed(a, by_hand):
        by_hand = None  # the hand edit missed the face; let the model decide this skin
    found_skin = model_skin if by_hand is None else by_hand
    if by_hand is None:
        found_skin = found_skin | missed_face_colours(a, found_skin, model.prior)
    if override.get("dark_hair_rule", True):
        found_skin = ungrey_dark_hair(a, found_skin, model.prior)
    skin = (drop_dark_clothing(a, found_skin) | forced_grey) & ~keep & ~forced_hair
    hair = hair_model.hair_mask(a, skin) if override.get("hair", True) else np.zeros(TEX_SIZE, bool)
    found_hair = hair
    if by_hand is None:
        skin, hair = claim_skin(a, skin, hair, model.prior, fixed=forced_hair | keep | untouched)
    max_patch = HAIR_ZONE_MAX_PATCH if by_hand is None else HAIR_ZONE_MAX_PATCH_HAND
    skin, hair = reclaim_hair(a, skin, hair, model.prior, fixed=forced_grey | keep | untouched, max_patch=max_patch)
    hair = grow_hair(a, skin, hair, fixed=forced_grey | keep | untouched) | forced_hair
    skin &= ~untouched
    hair &= ~untouched
    tone_hair(out, hair, sample=found_hair)
    ash_skin(a, out, skin)
    paint_brows(a, out, skin)
    paint_eyes(a, out, find_eye_rows(a))
    return out, skin, hair


def paint_eyes(a: np.ndarray, out: np.ndarray, rows: list[int]) -> None:
    """Pupils go dark red; the eye whites just outside them go bright red, so the eyes read as solid red.

    Two-row eyes keep the original's light/dark difference between their rows (see EYE_MAX_LIFT).
    """
    lightness = to_lab(a[..., :3])[..., 0]
    for xs, red, is_part in (
        (EYE_XS, TARGET_RED, lambda y, x: True),
        ((EYE_XS[0] - 1, EYE_XS[1] + 1), EYE_WHITE_RED, lambda y, x: a[y, x, :3].min() >= 150),
    ):
        red_lab = to_lab(np.array(red, float))
        for x in xs:
            ys = [y for y in rows if a[y, x, 3] > 0 and is_part(y, x)]
            if not ys:
                continue
            ys.sort()
            step = min(abs(lightness[ys[0], x] - lightness[ys[-1], x]), EYE_MAX_LIFT)
            for y in ys:
                lift = step if y != ys[-1] else 0
                if lift == 0:
                    out[y, x, :3] = red
                else:
                    # Scale the red's chroma with its lightness so it stays red rather than greying out.
                    scale = (red_lab[0] + lift) / red_lab[0]
                    out[y, x, :3] = np.round(from_lab(red_lab * [scale, scale, scale]))


def is_protected(path: Path) -> bool:
    path = path.resolve()
    if path.name.startswith("0.hand_edits_backup") or any(p.name.startswith("0.hand_edits_backup") for p in path.parents):
        return True
    return any(path == d.resolve() or d.resolve() in path.parents for d in PROTECTED)


def new_output_dir(base: str) -> Path:
    candidate = ASSET_DEV / base
    n = 2
    while candidate.exists():
        candidate = ASSET_DEV / f"{base}_v{n}"
        n += 1
    if is_protected(candidate):
        raise SystemExit(f"refusing to write into {candidate}: originals and hand edits are read-only")
    candidate.mkdir(parents=True)
    return candidate


def save(a: np.ndarray, path: Path) -> None:
    if is_protected(path) or path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    to_img(a).save(path)


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


def manual_check(name: str, a: np.ndarray, out: np.ndarray, skin: np.ndarray):
    """Compare the tool's skin greying with your pass-1 edit. None if there's no usable hand edit."""
    path = MANUAL_PASS1 / name
    if not path.exists():
        return None
    manual = load_rgba(path)
    if manual.shape != a.shape:
        return None
    vis = a[..., 3] > 0
    eyes = np.zeros(TEX_SIZE, bool)
    eyes[11:16, 9:15] = True
    manual_change = np.abs(manual - a)[..., :3].max(-1)
    greyed_by_hand = (manual_change > 0) & vis & ~eyes
    if greyed_by_hand.sum() < TRAIN_MIN_PIXELS:
        return None
    tool_change = np.abs(out - a)[..., :3].max(-1)
    missed = greyed_by_hand & ~skin & (manual_change > REVIEW_VISIBLE)
    extra = skin & ~eyes & ~greyed_by_hand & (tool_change > REVIEW_VISIBLE)
    return missed, extra, manual


def review_sheet(rows: list, out_dir: Path, scale: int = 3) -> None:
    """Flagged skins: original | differences (magenta = you greyed, tool didn't; cyan = tool greyed,
    you didn't) | your pass 1 | tool."""
    w, h = TEX_SIZE[1] * scale, TEX_SIZE[0] * scale
    gap = 8
    sheet = Image.new("RGBA", (4 * w + 5 * gap, len(rows) * (h + gap) + gap), (45, 45, 45, 255))
    for row, (name, a, out, missed, extra, manual) in enumerate(rows):
        marked = a.copy()
        marked[..., :3] = marked[..., :3] * 0.35 + 255 * 0.65 * (a[..., 3:4] > 0)
        marked[missed, :3] = (255, 0, 255)
        marked[extra, :3] = (0, 200, 255)
        for col, arr in enumerate((a, marked, manual, out)):
            tile = Image.new("RGBA", (TEX_SIZE[1], TEX_SIZE[0]), (255, 255, 255, 255))
            tile.alpha_composite(to_img(arr))
            sheet.paste(tile.resize((w, h), Image.Resampling.NEAREST), (gap + col * (w + gap), gap + row * (h + gap)))
    sheet.save(out_dir / "_review.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", nargs="*", help="texture names without .png, e.g. bakermale1_w")
    ap.add_argument("--all", action="store_true", help="process every original")
    ap.add_argument("--out", default="4.auto_poc", help="base name of the new output folder")
    ap.add_argument("--no-hand", action="store_true",
                    help="ignore your hand edits and let the model decide skin everywhere (for testing it)")
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
    checked, flagged, used_hand = 0, [], 0
    for name in names:
        a = load_rgba(ORIGINALS / name)
        if a.shape[:2] != TEX_SIZE:
            print(f"skip {name}: size {a.shape[1]}x{a.shape[0]} not supported yet")
            continue
        by_hand = None if args.no_hand else hand_greyed(name, a)
        out, skin, _ = dunmerify_with_masks(a, model, hair_model, override_for(overrides, name[:-4]), by_hand)
        save(out, out_dir / name)
        used_hand += by_hand is not None
        check = manual_check(name, a, out, skin)
        if check:
            checked += 1
            missed, extra, manual = check
            if missed.sum() + extra.sum() > REVIEW_MIN_DIFF:
                flagged.append((name, a, out, missed, extra, manual))
    if not args.all:
        comparison_sheet(names, out_dir)
    flagged.sort(key=lambda r: -(r[3].sum() + r[4].sum()))
    lines = [f"{len(flagged)} of {checked} hand-edited skins differ from your pass 1 by more than {REVIEW_MIN_DIFF} visible pixels.",
             "missed = you greyed it, the tool didn't; extra = the tool greyed it, you didn't.", ""]
    lines += [f"{name[:-4]:28s} missed {int(m.sum()):4d}  extra {int(e.sum()):4d}" for name, _, _, m, e, _ in flagged]
    (out_dir / "_review.txt").write_text("\n".join(lines) + "\n")
    if flagged:
        review_sheet(flagged, out_dir)
    print(f"skin taken from your hand edits for {used_hand} skins; the model decided the rest")
    print(lines[0])
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
