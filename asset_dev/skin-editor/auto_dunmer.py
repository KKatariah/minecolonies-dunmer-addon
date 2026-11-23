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
import functools
import json
import re
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
CITIZEN_MODELS = Path(__file__).resolve().parent / "citizen_models.json"  # from extract_models.py
# Model pieces past the standard layout that can show skin; every other piece is a prop or clothing.
SKIN_PARTS = {"breast", "foreArm", "leftForeArm", "rightForeArm", "leftShoulder", "rightShoulder", "shoulderArm"}
# Model pieces that are hair, so may be toned as hair; the rest (hats, potions...) never are.
HAIR_PARTS = re.compile(r"hair|beard|pony", re.IGNORECASE)
# Outer-layer areas (x, y, w, h) of the standard layout and the base-layer (x, y) they sit on.
OVERLAYS = (((32, 0, 32, 16), (0, 0)), ((16, 32, 24, 16), (16, 16)), ((40, 32, 16, 16), (40, 16)),
            ((0, 32, 16, 16), (0, 16)), ((0, 48, 16, 16), (16, 48)), ((48, 48, 16, 16), (32, 48)))
# An outer-layer pixel over clothing is only skin (e.g. a leg through a skirt slit) when its colour
# is within this distance (CIELAB) of the face's skin; otherwise it's a buckle, circlet, trim...
OVERLAY_SKIN_DIST = 4

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
# The darkest a greyed pixel gets: below this the ramp's bluish shadow hue is all that's left and dark
# shading (or a dark brow the skin took in) came out near-black blue.
SKIN_L_FLOOR = 12
# With the soft_shadows override, skin more than SKIN_SHADOW_KNEE (L*) below the median skin only
# goes SKIN_SHADOW_SOFT as much darker.
SKIN_SHADOW_KNEE = 18
SKIN_SHADOW_SOFT = 0.4
# Near-black pixels (L* under SKIN_BLACK_L) more than SKIN_BLACK_GAP darker than the face's darkest
# skin are hair, outlines or clothes, never skin: on the dark-skinned variants they got greyed.
SKIN_BLACK_L = 10
SKIN_BLACK_GAP = 10
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
# Leg faces of the standard layout (right leg and its overlay, left leg and its overlay).
LEGS = np.zeros(TEX_SIZE, bool)
LEGS[16:48, 0:16] = True
LEGS[48:64, 0:32] = True
# The hair model decides colour by colour and misses some shades of the same hair. On the inner
# head layer, hair grows into touching untouched pixels within this colour distance (CIELAB) of
# hair already found. Not on the outer layer (mostly caps and hats), and never into skin: reddish
# hair and skin are too close to tell apart by colour (use the hair_colours override for those).
HAIR_GROW_DIST = 10
HAIR_GROW_STEPS = 2
# Hair keeps only its own shades: the commonest colours making up this share of the hair, plus any
# hair colour chained to them by steps under HAIR_PALETTE_DIST (CIELAB). See keep_hair_palette.
HAIR_PALETTE_SHARE = 0.6
HAIR_PALETTE_DIST = 15
# The commonest colours are counted on these rows of the inner head layer (around the face and nape),
# if at least HAIR_MIN_SEED hair pixels are there; otherwise over all the hair.
HAIR_SEED_ROWS = (10, 16)
# When the hair's commonest shades are clearly coloured (median CIELAB chroma above HAIR_COLOURED),
# black and grey pixels (chroma under HAIR_NEUTRAL) are a hood or hat, not hair.
HAIR_COLOURED = 12
HAIR_NEUTRAL = 4
HAIR_MIN_SEED = 8
# Pixels on a model's hair pieces this close (CIELAB) to a colour of the head's hair are hair too.
HAIR_PIECE_DIST = 12
# A hood: at least HOOD_SHARE of each of the head's top, sides and back (inner layer) is grey or
# white (chroma under HOOD_CHROMA, L* over HOOD_MIN_L). See hood_hair.
HOOD_SHARE = 0.85
HOOD_CHROMA = 6
HOOD_MIN_L = 40
# Under a hood, hair this close (CIELAB) to a colour on the head's top, sides or back is a band.
HOOD_BAND_DIST = 12
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
BROW_DARKER = 12
BROW_CHROMA = 0.6
# ...but never more than this much darker than they were (grey faces are dark already, so a fixed
# offset below the face turned light brows black), and only pixels darker than the original skin
# by BROW_MIN_DARKER count as brows (light face paint in the brow rows isn't a brow).
BROW_MAX_DROP = 15
BROW_MIN_DARKER = 5
# Brow-row pixels within this colour distance (CIELAB) of the non-skin pixel above them carry on from
# above (hair fringe, hat brim, headband, hood), so they aren't brows and keep their own toning.
BROW_SAME_DIST = 12
# brow_style "hair": brows this much darker (L*) than the toned head hair, outer pixel / the rest.
BROW_HAIR_OUTER = 14
BROW_HAIR_INNER = 10.5
HAIR_MIN_EDITED_PIXELS = 10
# Hair treatments: (saturation factor, brightness factor, contrast around the hair's mean brightness).
HAIR_RED = (0.90, 0.65, 1.0)
HAIR_BLONDE = (0.70, 0.80, 1.3)
HAIR_OTHER = (0.55, 0.75, 1.0)


def load_rgba(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA")).astype(float)


def frame(a: np.ndarray) -> np.ndarray:
    """The standard 128x64 citizen layout from a texture of any size.

    Bigger textures (nobles, aristocrats, the female deliveryman) put the usual layout in their
    top-left corner and dress or cart parts outside it; the 64x32 deliveryman is the top-left
    quarter of it. Anything outside the frame is left as it is.
    """
    framed = np.zeros((*TEX_SIZE, 4))
    h, w = min(a.shape[0], TEX_SIZE[0]), min(a.shape[1], TEX_SIZE[1])
    framed[:h, :w] = a[:h, :w]
    return framed


def unframe(a: np.ndarray, framed: np.ndarray) -> np.ndarray:
    out = a.copy()
    h, w = min(a.shape[0], TEX_SIZE[0]), min(a.shape[1], TEX_SIZE[1])
    out[:h, :w] = framed[:h, :w]
    return out


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
        mask &= ~LEGS  # hair never hangs onto the legs; dark soles there can match the hair colour
        body = mask.copy()
        body[:16, :64] = False
        labels, n = ndimage.label(body)
        if n:
            sizes = ndimage.sum(body, labels, range(1, n + 1))
            mask &= ~np.isin(labels, 1 + np.nonzero(sizes <= HAIR_MAX_SPECK)[0])
        return mask


def tone_hair(out: np.ndarray, hair: np.ndarray, sample: np.ndarray | None = None,
              treatment: list[float] | None = None) -> str | None:
    """Tone hair down in place; one treatment per skin, picked from the hair's overall colour
    (from `sample` if given, so grown-in edge shades don't change which treatment is chosen).
    `treatment` ([saturation, value, contrast] factors) overrides the pick.
    Returns the hair kind the colour picked: "red", "blonde" or "other" (None without hair)."""
    if not hair.any():
        return None
    sample = hair if sample is None or not sample.any() else sample
    h, s, v = (c[sample] for c in hsv(out))
    hr = np.radians(h)
    mean_h = np.degrees(np.arctan2(np.sin(hr).mean(), np.cos(hr).mean())) % 360
    mean_s, mean_v = np.median(s), np.median(v)
    h, s, v = (c[hair] for c in hsv(out))
    if (mean_h < 25 or mean_h >= 330) and mean_s >= 0.55 and mean_v >= 0.4:
        kind, (sat_f, val_f, contrast) = "red", HAIR_RED
    elif 25 <= mean_h < 70 and mean_v > 0.5:
        kind, (sat_f, val_f, contrast) = "blonde", HAIR_BLONDE
    else:
        kind, (sat_f, val_f, contrast) = "other", HAIR_OTHER
    if treatment:
        sat_f, val_f, contrast = treatment
    v = v * val_f
    v = v.mean() + (v - v.mean()) * contrast
    out[..., :3][hair] = hsv_to_rgb(h, np.clip(s * sat_f, 0, 1), np.clip(v, 0, 1))
    return kind


def grow_hair(a: np.ndarray, skin: np.ndarray, hair: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Grow hair into similar touching shades; skin and `fixed` pixels are never taken.

    On the inner head layer it grows as far as the shades reach. Elsewhere (the outer head layer, locks
    hanging on the body) only HAIR_GROW_STEPS pixels out, to fill gaps in hair already found without
    flooding into a hat or clothes of a similar colour. Never onto the legs.
    """
    if hair.sum() < 5:
        return hair
    inner_head = np.zeros(TEX_SIZE, bool)
    inner_head[:16, :32] = True
    usable = (a[..., 3] > 0) & ~skin & ~fixed & ~LEGS
    hair = _grow(a, hair, inner_head & usable)
    return _grow(a, hair, usable, steps=HAIR_GROW_STEPS)


def in_hue_range(a: np.ndarray, hues: list[float] | None) -> np.ndarray:
    """Pixels whose CIELAB hue is within [lo, hi] degrees, or that are nearly grey; all pixels if
    `hues` is None (the hair_hue override: a hat or hood of another colour isn't hair)."""
    if not hues:
        return np.ones(TEX_SIZE, bool)
    lab = to_lab(a[..., :3])
    hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
    lo, hi = hues
    return (np.hypot(lab[..., 1], lab[..., 2]) <= 8) | ((hue >= lo) & (hue <= hi) if lo <= hi else (hue >= lo) | (hue <= hi))


# The inner head layer's top, sides and back.
HEAD_AROUND = np.zeros(TEX_SIZE, bool)
HEAD_AROUND[0:8, 8:16] = HEAD_AROUND[8:16, 0:8] = HEAD_AROUND[8:16, 16:32] = True


def hood(a: np.ndarray) -> np.ndarray | None:
    """The grey or white pixels of a hood covering the inner head layer's top, sides and back, or
    None if there's no hood."""
    lab = to_lab(a[..., :3])
    grey = (np.hypot(lab[..., 1], lab[..., 2]) < HOOD_CHROMA) & (lab[..., 0] > HOOD_MIN_L)
    visible = a[..., 3] > 0
    for x0, y0 in ((8, 0), (0, 8), (16, 8), (24, 8)):
        face = visible[y0:y0 + 8, x0:x0 + 8]
        if face.sum() < 8 or grey[y0:y0 + 8, x0:x0 + 8][face].mean() < HOOD_SHARE:
            return None
    return grey


def hood_hair(a: np.ndarray, hair: np.ndarray, grey: np.ndarray | None) -> np.ndarray:
    """Under a hood (`grey`, from hood()), the hood isn't hair, and neither is a band or trim whose
    colours (or close shades) go round the head on it. Hair on the face or hanging over the hood
    on the outer layer stays."""
    if grey is None:
        return hair
    lab = to_lab(a[..., :3])
    band = np.zeros(TEX_SIZE, bool)
    ring = np.unique(lab[HEAD_AROUND & (a[..., 3] > 0) & ~grey], axis=0)
    if len(ring) and hair.any():
        dist = np.linalg.norm(lab[hair][:, None] - ring[None], axis=-1).min(1)
        band[hair] = dist < HOOD_BAND_DIST
    return hair & ~HEAD_AROUND & ~grey & ~band


def keep_hair_palette(a: np.ndarray, hair: np.ndarray) -> np.ndarray:
    """Drop hair pixels whose colour isn't part of the hair's own shades.

    The palette starts from the commonest hair colours (HAIR_PALETTE_SHARE of the hair) and takes in
    every hair colour within HAIR_PALETTE_DIST of it, so shading ramps stay; a gold headband, a hair
    tie or a bow sitting in the hair is a different colour and is left alone.
    """
    if hair.sum() < 5:
        return hair
    lab = to_lab(a[..., :3]).round(1)
    cols, inv = np.unique(lab[hair], axis=0, return_inverse=True)
    inv = inv.ravel()
    # Count shades where hair shows around the face and nape: hats and turbans sit higher up, so a
    # big hat doesn't become "the hair colour".
    seed = np.zeros(TEX_SIZE, bool)
    seed[HAIR_SEED_ROWS[0] : HAIR_SEED_ROWS[1], :32] = True
    in_seed = seed[hair]
    if in_seed.sum() < HAIR_MIN_SEED:
        in_seed[:] = True
    counts = np.bincount(inv[in_seed], minlength=len(cols))
    order = np.argsort(-counts)
    inside = np.zeros(len(cols), bool)
    inside[order[: np.searchsorted(np.cumsum(counts[order]), HAIR_PALETTE_SHARE * counts.sum()) + 1]] = True
    dist = np.sqrt(((cols[:, None] - cols[None]) ** 2).sum(-1))
    while True:
        more = ~inside & (dist[:, inside] < HAIR_PALETTE_DIST).any(1)
        if not more.any():
            break
        inside |= more
    kept = hair.copy()
    kept[hair] = inside[inv]
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    if np.median(chroma[hair][in_seed]) > HAIR_COLOURED:
        kept &= chroma >= HAIR_NEUTRAL
    return kept


def _grow(a: np.ndarray, hair: np.ndarray, allowed: np.ndarray, steps: int | None = None) -> np.ndarray:
    lab = to_lab(a[..., :3])
    hair = hair.copy()
    step = 0
    while steps is None or step < steps:
        step += 1
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


def nearest(lab: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per pixel, the CIELAB distance to the closest colour among the `ref` pixels (inf if none)."""
    colours = np.unique(lab[ref].round(1), axis=0)
    if not len(colours):
        return np.full(lab.shape[:2], np.inf)
    return np.sqrt(((lab[..., None, :] - colours[None, None]) ** 2).sum(-1)).min(-1)


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
        # Only colours closer to the hair than to the face's skin go back: hairlines, sideburn
        # gaps and bald crowns are skin-coloured patches surrounded by hair and stay skin.
        lab = to_lab(a[..., :3])
        back &= nearest(lab, hair & ~back) < nearest(lab, face)
        lightness = lab[..., 0]
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


def ash_skin(a: np.ndarray, out: np.ndarray, skin: np.ndarray, darker: float = 0, soft_shadows: bool | float = False,
             undertone: list[float] | None = None, contrast: float = 1.0, redness: float = 0.0) -> None:
    """Recolour skin in place onto the ash ramp; call after hair is toned (it checks against it).

    darker: L* to take off every skin pixel afterwards (the skin_darker override).
    soft_shadows: keep the deepest shadows dark grey instead of near-black (the soft_shadows override:
    true, or how far below the median skin, in L*, the softening starts).
    undertone: [hue shift in degrees, chroma factor] for the ramp's colour (the skin_undertone override),
    e.g. to set the skin apart from a grey accessory.
    contrast: factor on each pixel's distance from the median skin lightness (the skin_contrast
    override), for skin that comes out flat.
    redness: L* to darken per unit of redness (CIELAB a*) above the skin's mean (the skin_redness
    override), so blush, lips and pink shading in a pale, flat original become shadows."""
    if not skin.any():
        return
    src = to_lab(a[..., :3])[..., 0][skin]
    mean, spread = src.mean(), src.std() + 1e-6
    stretch = min(SKIN_MAX_STRETCH, max(1.0, SKIN_MIN_SPREAD / spread))
    light = min(mean, SKIN_L_CAP) + (src - mean) * stretch
    if redness:
        red = to_lab(a[..., :3])[..., 1][skin]
        light = light - (red - red.mean()) * redness
    if contrast != 1.0:
        light = np.median(light) + (light - np.median(light)) * contrast
    if soft_shadows:
        knee = np.median(light) - (SKIN_SHADOW_KNEE if soft_shadows is True else soft_shadows)
        light = np.where(light < knee, knee - (knee - light) * SKIN_SHADOW_SOFT, light)

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
    hue_shift, chroma_f = undertone or (0, 1)
    hue = np.radians(SKIN_SHADOW[0] + (SKIN_LIGHT[0] - SKIN_SHADOW[0]) * t + hue_shift)
    chroma = (SKIN_SHADOW[1] + (SKIN_LIGHT[1] - SKIN_SHADOW[1]) * t) * chroma_f
    lab = np.stack([np.clip(light - darker, SKIN_L_FLOOR, 100), chroma * np.cos(hue), chroma * np.sin(hue)], -1)
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
    """Entries for every skin ("*"), the outfit (e.g. "cookfemale1", all variants) and the skin's own, combined."""
    merged: dict = {}
    for entry in (overrides.get("*", {}), overrides.get(name[:-2], {}), overrides.get(name, {})):
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


def continues_from_above(a: np.ndarray, area: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """Pixels of `area` that carry on, row by row, from a matching non-skin pixel above the area."""
    lab = to_lab(a[..., :3])
    solid = (a[..., 3] > 0) & ~skin
    same = np.zeros(TEX_SIZE, bool)
    same[1:] = solid[1:] & solid[:-1] & (np.linalg.norm(lab[1:] - lab[:-1], axis=-1) < BROW_SAME_DIST)
    carried = np.zeros(TEX_SIZE, bool)
    for y in sorted(set(np.nonzero(area)[0])):
        above = carried[y - 1] if area[y - 1].any() else np.ones(TEX_SIZE[1], bool)
        carried[y] = area[y] & same[y] & above
    return carried


def mirrored_brows(a: np.ndarray, brows: np.ndarray) -> np.ndarray:
    """Mirror images (across the face) of brow pixels, where they match the brow's colour.

    Faces are drawn symmetric but hairlines often aren't, so one brow can look like fringe
    carrying on from above while the other doesn't; both should be treated the same.
    """
    lab = to_lab(a[..., :3])
    face, flipped = slice(8, 16), slice(15, 7, -1)
    out = np.zeros(TEX_SIZE, bool)
    out[:, face] = brows[:, flipped] & (np.linalg.norm(lab[:, face] - lab[:, flipped], axis=-1) < BROW_SAME_DIST)
    return out


def paint_brows(a: np.ndarray, out: np.ndarray, skin: np.ndarray, untouched: np.ndarray | None = None,
                hair: np.ndarray | None = None, style: str | None = None) -> None:
    """Darken and mute the eyebrows so they stand out from the grey face and the red eyes.

    Pixels already dark enough are left alone, and so are fringes, hat brims and headbands that
    carry on from above: those are toned with the hair or not at all.
    """
    area = brow_area(a) & ~skin & (a[..., 3] > 0)
    if untouched is not None:
        area &= ~untouched
    face = skin & face_middle(a)
    if not area.any() or not face.any():
        return
    out_lab = to_lab(out[..., :3])
    limit = np.median(out_lab[..., 0][face]) - BROW_DARKER
    orig_l = to_lab(a[..., :3])[..., 0]
    area &= orig_l < np.median(orig_l[face]) - BROW_MIN_DARKER
    brows = area & ~continues_from_above(a, area, skin)
    brows |= area & mirrored_brows(a, brows)
    brows &= out_lab[..., 0] > limit
    if not brows.any():
        return
    if style == "hair" and hair is not None:
        hair_brows(out, out_lab, brows, hair)
        return
    # Each brow drops by one amount (enough to bring its lightest pixel down to the limit, capped
    # at BROW_MAX_DROP) so its light/dark shading survives instead of every pixel clamping to the
    # same lightness.
    labels, n = ndimage.label(brows, np.ones((3, 3)))
    lightest = np.asarray(ndimage.maximum(out_lab[..., 0], labels, range(1, n + 1)))
    drop = np.clip(lightest - limit, 0, BROW_MAX_DROP)
    lab = out_lab[brows]
    lab[:, 0] -= drop[labels[brows] - 1]
    lab[:, 1:] *= BROW_CHROMA
    out[..., :3][brows] = np.round(from_lab(lab))


def hair_brows(out: np.ndarray, out_lab: np.ndarray, brows: np.ndarray, hair: np.ndarray) -> None:
    """Brows in the toned head hair's colour (full strength), the outer pixel of each brow darker
    than the inner ones. For blonde hair, where muting the brow's own colour turns it grey-khaki."""
    head = hair.copy()
    head[16:] = False
    head[:, 32:] = False
    if not head.any():
        return
    mean = out_lab[head].mean(0)
    labels, n = ndimage.label(brows, np.ones((3, 3)))
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labels == i)
        side = np.abs(xs - 11.5)
        for y, x, d in zip(ys, xs, side):
            outer = d == side[ys == y].max() and (ys == y).sum() > 1
            lab = np.array([mean[0] - (BROW_HAIR_OUTER if outer else BROW_HAIR_INNER), mean[1], mean[2]])
            out[y, x, :3] = np.round(from_lab(lab[None]))[0]


def hand_greyed(name: str, a: np.ndarray) -> np.ndarray | None:
    """Your pass-1 greying for this skin, if you edited it normally (not skipped or over-greyed)."""
    path = MANUAL_PASS1 / name
    if not path.exists():
        return None
    manual = load_rgba(path)
    if manual.shape != a.shape:
        return None
    manual, a = frame(manual), frame(a)
    greyed = (np.abs(manual - a)[..., :3].max(-1) > 0) & (a[..., 3] > 0)
    return greyed if TRAIN_MIN_PIXELS <= greyed.sum() <= TRAIN_MAX_PIXELS else None


def cube_faces(u: int, v: int, w: float, h: float, d: float) -> np.ndarray:
    """Texture pixels a model cube's six faces use (vanilla box UV layout)."""
    w, h, d = (int(np.ceil(n)) for n in (w, h, d))
    mask = np.zeros(TEX_SIZE, bool)
    mask[v : v + d, u + d : u + d + 2 * w] = True  # top, bottom
    mask[v + d : v + d + h, u : u + 2 * d + 2 * w] = True  # sides
    return mask


@functools.cache
def _citizen_models() -> dict:
    return json.loads(CITIZEN_MODELS.read_text()) if CITIZEN_MODELS.exists() else {"types": {}, "models": {}}


# Texture pixels the standard body (head, torso, arms, legs and their outer layers) uses. Model
# pieces sometimes take their texture from the unused gaps between them, e.g. a hat tier at (58, 16).
STANDARD_LAYOUT = np.zeros(TEX_SIZE, bool)
for _u, _v, _w, _h, _d in ((0, 0, 8, 8, 8), (32, 0, 8, 8, 8), (16, 16, 8, 12, 4), (16, 32, 8, 12, 4),
                           (40, 16, 4, 12, 4), (40, 32, 4, 12, 4), (0, 16, 4, 12, 4), (0, 32, 4, 12, 4),
                           (16, 48, 4, 12, 4), (0, 48, 4, 12, 4), (32, 48, 4, 12, 4), (48, 48, 4, 12, 4)):
    STANDARD_LAYOUT |= cube_faces(_u, _v, _w, _h, _d)


@functools.cache
def _accessory_mask(name: str) -> tuple[np.ndarray, np.ndarray]:
    models = _citizen_models()
    m = re.match(r"^(.*?)(female|male)\d+", name)
    model = models["models"].get((models["types"].get(m.group(1)) or {}).get(m.group(2))) if m else None
    props, body, hair = (np.zeros(TEX_SIZE, bool) for _ in range(3))
    for part in model["parts"] if model else []:
        for u, v, _, _, _, w, h, d, *_ in part["cubes"]:
            (body if part["name"] in SKIN_PARTS else props)[:] |= cube_faces(u, v, w, h, d)
            if HAIR_PARTS.search(part["name"]):
                hair |= cube_faces(u, v, w, h, d)
    props &= ~STANDARD_LAYOUT
    return props & ~body, props & ~body & ~hair, hair & ~STANDARD_LAYOUT


def accessory_mask(name: str | None) -> np.ndarray:
    """Where the texture's citizen model draws its own pieces past the standard layout (capes,
    lockets, potions, aprons, dresses, hats, tools...): never skin, whatever the colour."""
    return _accessory_mask(name)[0] if name else np.zeros(TEX_SIZE, bool)


def prop_mask(name: str | None) -> np.ndarray:
    """The accessory pieces that aren't hair either (hats, potions, tools...): never changed."""
    return _accessory_mask(name)[1] if name else np.zeros(TEX_SIZE, bool)


HATS = Path(__file__).resolve().parent / "hats"  # hat templates: a hat-layer hat (x 32-63, y 0-15), transparent elsewhere
HAT_MATCH = 2  # max RGB difference for a pixel to count as the same hat
HAT_SHARE = 0.5  # share of a template that must match for the skin to be wearing that hat


@functools.lru_cache(maxsize=None)
def _hat_templates() -> tuple:
    return tuple(frame(load_rgba(f)) for f in sorted(HATS.glob("*.png")))


def known_hats(a: np.ndarray) -> np.ndarray:
    """Pixels of hats shared between outfits (templates in hats/), matched pixel for pixel: a hat,
    not hair, so never changed."""
    out = np.zeros(TEX_SIZE, bool)
    for hat in _hat_templates():
        same = (hat[..., 3] > 0) & (a[..., 3] > 0) & (np.abs(a[..., :3] - hat[..., :3]).max(-1) <= HAT_MATCH)
        if same.sum() >= HAT_SHARE * (hat[..., 3] > 0).sum():  # the hat itself, not a few colours that happen to match
            out |= same
    return out


def hair_pieces(a: np.ndarray, hair: np.ndarray, name: str | None) -> np.ndarray:
    """Pixels on the model's own hair pieces (braids, ponytails, hair extensions past the standard
    layout) in the colours of the hair found on the head: the head's hair carries on there."""
    if not name or not hair.any():
        return np.zeros(TEX_SIZE, bool)
    pieces = _accessory_mask(name)[2] & (a[..., 3] > 0) & ~hair
    if not pieces.any():
        return pieces
    lab = to_lab(a[..., :3])
    cols = np.unique(lab[hair], axis=0)
    out = np.zeros(TEX_SIZE, bool)
    out[pieces] = np.linalg.norm(lab[pieces][:, None] - cols[None], axis=-1).min(1) < HAIR_PIECE_DIST
    return out


def black_not_skin(a: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """Near-black pixels much darker than the face's darkest skin (see SKIN_BLACK_L)."""
    light = to_lab(a[..., :3])[..., 0]
    face = skin[8:16, 8:16]
    if face.sum() < 6:
        return np.zeros(TEX_SIZE, bool)
    darkest = np.percentile(light[8:16, 8:16][face], 5)
    return (light < SKIN_BLACK_L) & (light < darkest - SKIN_BLACK_GAP)


def overlay_on_clothes(a: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """Outer-layer skin pixels that sit over clothing in the base layer and aren't the face's skin
    colour: belt buckles, circlets, trims and boot details, not skin."""
    lab = to_lab(a[..., :3])
    face = lab[8:16, 8:16][skin[8:16, 8:16]]
    out = np.zeros(TEX_SIZE, bool)
    for (x, y, w, h), (bx, by) in OVERLAYS:
        out[y : y + h, x : x + w] = skin[y : y + h, x : x + w] & ~skin[by : by + h, bx : bx + w]
    for y, x in np.argwhere(out):
        if len(face) and np.linalg.norm(face - lab[y, x], axis=1).min() <= OVERLAY_SKIN_DIST:
            out[y, x] = False
    return out


def dunmerify(a: np.ndarray, model: SkinModel, hair_model: HairModel, override: dict | None = None,
              by_hand: np.ndarray | None = None, name: str | None = None) -> np.ndarray:
    return dunmerify_with_masks(a, model, hair_model, override, by_hand, name)[0]


def dunmerify_with_masks(a: np.ndarray, model: SkinModel, hair_model: HairModel, override: dict | None = None,
                         by_hand: np.ndarray | None = None, name: str | None = None):
    """Returns (recoloured texture, skin mask, hair mask).

    by_hand: your pass-1 greying (see hand_greyed). When given it decides what is skin instead of
    the model; overrides and the dark-clothing rule still apply on top.
    name: the texture's name (e.g. alchemistmale1_w), to look up its citizen model so its props
    and clothing pieces are never greyed (see accessory_mask).
    """
    override = override or {}
    a, borrowed = borrow_areas(a, override.get("borrow_area", []))
    out = a.copy()
    forced_grey = colour_mask(a, override.get("grey", [])) | (area_mask(override.get("grey_area", [])) & (a[..., 3] > 0))
    forced_grey |= borrowed & (a[..., 3] > 0)
    forced_grey |= paint_mask(a, override.get("face_paint", []))
    forced_hair = colour_mask(a, override.get("hair_colours", [])) | (area_mask(override.get("hair_area", [])) & (a[..., 3] > 0))
    keep = colour_mask(a, override.get("keep", []))
    untouched = area_mask(override.get("keep_area", [])) | colour_mask(a, override.get("untouched", [])) | known_hats(a)
    model_skin = model.skin_mask(a)
    if by_hand is not None and face_missed(a, by_hand):
        by_hand = None  # the hand edit missed the face; let the model decide this skin
    found_skin = model_skin if by_hand is None else by_hand
    if by_hand is None:
        found_skin = found_skin | missed_face_colours(a, found_skin, model.prior)
    if override.get("dark_hair_rule", True):
        found_skin = ungrey_dark_hair(a, found_skin, model.prior)
    skin = (drop_dark_clothing(a, found_skin) | forced_grey) & ~keep & ~forced_hair & ~accessory_mask(name)
    hair = hair_model.hair_mask(a, skin) if override.get("hair", True) else np.zeros(TEX_SIZE, bool)
    found_hair = hair
    if by_hand is None:
        skin, hair = claim_skin(a, skin, hair, model.prior, fixed=forced_hair | keep | untouched)
    max_patch = HAIR_ZONE_MAX_PATCH if by_hand is None else HAIR_ZONE_MAX_PATCH_HAND
    skin, hair = reclaim_hair(a, skin, hair, model.prior, fixed=forced_grey | keep | untouched, max_patch=max_patch)
    hair = grow_hair(a, skin, hair, fixed=forced_grey | keep | untouched)
    hair = keep_hair_palette(a, hair) & in_hue_range(a, override.get("hair_hue")) | forced_hair
    skin &= ~untouched & ~overlay_on_clothes(a, skin) & ~black_not_skin(a, skin)
    hooded = hood(a)
    hair = hood_hair(a, hair, hooded) | forced_hair
    hair |= hair_pieces(a, hair, name) & ~untouched & ~keep
    if hooded is not None:
        skin &= ~hooded | forced_grey
    hair &= ~untouched & ~prop_mask(name)
    if "hair_only_area" in override:
        hair &= area_mask(override["hair_only_area"])
    hair_kind = tone_hair(out, hair, sample=found_hair & hair, treatment=override.get("hair_tone"))
    darken_mask(out, hair, override.get("hair_darker", 0))
    if "hair_tint" in override:
        # {colour, strength}: move the toned hair's colour that far towards the colour, keeping its lightness.
        tint_mask(out, hair, override["hair_tint"]["colour"], override["hair_tint"].get("strength", 0.3))
    ash_skin(a, out, skin, override.get("skin_darker", 0), override.get("soft_shadows", False),
             override.get("skin_undertone"), override.get("skin_contrast", 1.0),
             override.get("skin_redness", 0.0))
    # Blonde brows take the hair's colour: muting their own turns them grey-khaki. "brow_style":
    # "default" turns that off for a skin, "hair" turns it on.
    brow_style = override.get("brow_style", "hair" if hair_kind == "blonde" else "default")
    paint_brows(a, out, skin, untouched, hair, brow_style)
    face_paint(a, out, override.get("face_paint", []))
    paint_eyes(a, out, find_eye_rows(a))
    fill_pixels(out, override.get("fill", []))
    desaturate(a, out, override.get("desat", {}))
    ash_tint(a, out, override.get("ash", []))
    tint(a, out, override.get("tint", {}))
    tint_areas(out, override.get("tint_area", []))
    darken_areas(out, override.get("darken_area", []))
    if override.get("slim_arms", True):
        slim_arms(a, out, name)
    copy_areas(out, override.get("copy_area", []))
    torso_bottom = override.get("torso_bottom", "legs")
    if torso_bottom == "original":
        for x, y in TORSO_BOTTOM:
            out[y : y + 4, x : x + 8] = a[y : y + 4, x : x + 8]
    elif torso_bottom != "keep":
        torso_bottom_from_legs(out, torso_bottom if torso_bottom in ("left", "right") else None)
    return out, skin, hair


def desaturate(a: np.ndarray, out: np.ndarray, colours: dict[str, float]) -> None:
    """Scale the colourfulness (CIELAB chroma) of the finished pixels whose original colour is one of
    `colours` by its factor: 0 makes them grey, 0.5 halves their colour."""
    for colour, factor in colours.items():
        mask = colour_mask(a, [colour])
        if mask.any():
            lab = to_lab(out[..., :3][mask])
            lab[:, 1:] *= factor
            out[..., :3][mask] = np.round(from_lab(lab))


def ash_tint(a: np.ndarray, out: np.ndarray, colours: list[str]) -> None:
    """Give pixels of these original colours the skin's blue-grey undertone, keeping their own
    lightness: for skin-coloured trims that look odd next to grey skin but aren't skin."""
    mask = colour_mask(a, colours)
    if not mask.any():
        return
    lab = to_lab(a[..., :3][mask])
    hue = np.radians(SKIN_LIGHT[0])
    lab[:, 1], lab[:, 2] = SKIN_LIGHT[1] * np.cos(hue), SKIN_LIGHT[1] * np.sin(hue)
    out[..., :3][mask] = np.round(from_lab(lab))


def tint(a: np.ndarray, out: np.ndarray, tints: dict[str, list[str]]) -> None:
    """Recolour pixels of the listed original colours to the key colour, keeping their shading:
    each keeps its lightness offset from the group's median, around the key colour's lightness."""
    for target, colours in tints.items():
        mask = colour_mask(a, colours)
        if not mask.any():
            continue
        lab = to_lab(a[..., :3][mask])
        goal = to_lab(np.array([int(target.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)], float))
        lab[:, 0] = np.clip(goal[0] + lab[:, 0] - np.median(lab[:, 0]), 0, 100)
        lab[:, 1], lab[:, 2] = goal[1], goal[2]
        out[..., :3][mask] = np.round(from_lab(lab))


def tint_areas(out: np.ndarray, tints: list[dict]) -> None:
    """Move the colour of each {colour, area: [[x0, y0, x1, y1]], strength} box's visible pixels
    `strength` of the way to the colour, keeping their lightness (e.g. a slight purple on grey cloth)."""
    for t in tints:
        tint_mask(out, area_mask(t["area"]), t["colour"], t.get("strength", 0.3))


def tint_mask(out: np.ndarray, mask: np.ndarray, colour: str, strength: float) -> None:
    """Move the colour of the mask's visible pixels `strength` of the way to `colour`, keeping their lightness."""
    mask = mask & (out[..., 3] > 0)
    if not mask.any():
        return
    lab = to_lab(out[..., :3][mask])
    goal = to_lab(np.array([int(colour.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)], float))
    lab[:, 1:] += (goal[1:] - lab[:, 1:]) * strength
    out[..., :3][mask] = np.round(from_lab(lab))


def paint_mask(a: np.ndarray, paints: list[dict]) -> np.ndarray:
    """Visible pixels in any face_paint area (they are greyed as skin before the paint goes on)."""
    return area_mask([box for p in paints for box in p["area"]]) & (a[..., 3] > 0)


def face_paint(a: np.ndarray, out: np.ndarray, paints: list[dict]) -> None:
    """Lay face paint over the greyed skin: each pixel keeps its grey lightness (the shading) and
    moves its colour `strength` of the way to the paint colour, so the paint sits on the ash
    undertone instead of the original skin's."""
    for p in paints:
        mask = area_mask(p["area"]) & (a[..., 3] > 0)
        if not mask.any():
            continue
        lab = to_lab(out[..., :3][mask])
        goal = to_lab(np.array([int(p["colour"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)], float))
        s = p.get("strength", 0.5)
        lab[:, 1:] += (goal[1:] - lab[:, 1:]) * s
        out[..., :3][mask] = np.round(from_lab(lab))


def darken_mask(out: np.ndarray, mask: np.ndarray, amount: float) -> None:
    """Lower the lightness (CIELAB L*) of the masked pixels by `amount` (negative lightens)."""
    if not amount or not mask.any():
        return
    lab = to_lab(out[..., :3][mask])
    lab[:, 0] = np.clip(lab[:, 0] - amount, 0, 100)
    out[..., :3][mask] = np.round(from_lab(lab))


def darken_areas(out: np.ndarray, boxes: list[list[int]]) -> None:
    """Lower the lightness (CIELAB L*) of each [x0, y0, x1, y1, amount] box, e.g. to make brows stand out."""
    for x0, y0, x1, y1, amount in boxes:
        region = out[y0:y1 + 1, x0:x1 + 1]
        visible = region[..., 3] > 0
        lab = to_lab(region[..., :3][visible])
        lab[:, 0] = np.maximum(lab[:, 0] - amount, 0)
        region[..., :3][visible] = np.round(from_lab(lab))


def tint_outside(full: np.ndarray, tints: list[dict]) -> None:
    """Like tint_area, but on the whole texture after unframing, for parts outside the standard
    128x64 layout (e.g. a noble's dress in the lower half of a 128x128 texture)."""
    for t in tints:
        mask = np.zeros(full.shape[:2], bool)
        for x0, y0, x1, y1 in t["area"]:
            mask[y0 : y1 + 1, x0 : x1 + 1] = True
        mask &= full[..., 3] > 0
        if not mask.any():
            continue
        lab = to_lab(full[..., :3][mask])
        goal = to_lab(np.array([int(t["colour"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)], float))
        lab[:, 1:] += (goal[1:] - lab[:, 1:]) * t.get("strength", 0.3)
        full[..., :3][mask] = np.round(from_lab(lab))


def borrow_areas(a: np.ndarray, borrows: list[list]) -> tuple[np.ndarray, np.ndarray]:
    """Paste [skin, src_x, src_y, w, h, dst_x, dst_y] blocks from another skin's original over this
    one before recolouring, e.g. a bare hand's underside where this one only has a glove. Returns the
    new texture and the pasted pixels, which are greyed as skin."""
    pasted = np.zeros(TEX_SIZE, bool)
    if not borrows:
        return a, pasted
    a = a.copy()
    for skin, sx, sy, w, h, dx, dy in borrows:
        donor = frame(load_rgba(ORIGINALS / f"{skin}.png"))
        a[dy:dy + h, dx:dx + w] = donor[sy:sy + h, sx:sx + w]
        pasted[dy:dy + h, dx:dx + w] = True
    return a, pasted


def copy_areas(out: np.ndarray, copies: list[list[int]]) -> None:
    """Paint [src_x, src_y, w, h, dst_x, dst_y] blocks over other parts of the texture, e.g. a leg top
    from the pants texture. Every block is read before any is written. Runs before the torso's
    underside takes the leg tops' texture, so a copied leg top carries on up into the torso."""
    blocks = [out[sy:sy + h, sx:sx + w].copy() for sx, sy, w, h, _, _ in copies]
    for (_, _, w, h, dx, dy), block in zip(copies, blocks):
        out[dy:dy + h, dx:dx + w] = block


def fill_pixels(out: np.ndarray, pixels: list[list[int]]) -> None:
    """Paint over stray pixels (e.g. a dot left in the original) with the median of their visible
    neighbours that aren't being filled too. Transparent pixels (holes) become opaque."""
    filled = np.zeros(TEX_SIZE, bool)
    for x, y in pixels:
        filled[y, x] = True
    for x, y in pixels:
        around = [out[ny, nx, :3] for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
                  if 0 <= ny < TEX_SIZE[0] and 0 <= nx < TEX_SIZE[1] and out[ny, nx, 3] > 0 and not filled[ny, nx]]
        if around:
            out[y, x, :3] = np.median(around, 0).round()
            out[y, x, 3] = 255


# Standard layout UVs (x0, y0) of 4x4 / 8x4 faces seen from above or below: the torso's bottom
# (base, overlay) and each leg's top (base, overlay). Top and bottom faces share the same
# orientation (left column = the citizen's right side, top row = back), so they copy straight across.
TORSO_BOTTOM = ((28, 16), (28, 32))
RIGHT_LEG_TOP = ((4, 16), (4, 32))
LEFT_LEG_TOP = ((20, 48), (4, 48))


# Arm texture blocks (u, v): right arm base and outer layer, left arm base and outer layer.
ARM_BLOCKS = ((40, 16), (40, 32), (32, 48), (48, 48))


def slim_arms(a: np.ndarray, out: np.ndarray, name: str | None) -> None:
    """Re-lay arms painted for 4px-wide arms onto the 3px layout their model uses (the forester men,
    for one), so every face shows as painted instead of a column off: the hand's underside showing
    a strip of sleeve, and so on. A texture is painted 4-wide if its right arm's back face reaches
    the 16th column (a 3-wide layout leaves the last two columns empty). The faces that get
    narrower lose their third column (the back face its second, so it mirrors the front)."""
    if not name:
        return
    models = _citizen_models()
    m = re.match(r"^(.*?)(female|male)\d+", name)
    model = models["models"].get((models["types"].get(m.group(1)) or {}).get(m.group(2))) if m else None
    arms = [p for p in (model or {}).get("parts", []) if p["name"] in ("right_arm", "left_arm")]
    if not arms or any(p["cubes"][0][5] != 3 for p in arms):
        return
    if (a[20:32, 54:56, 3] > 0).mean() <= 0.5:
        return
    drop = lambda block, keep: np.delete(block, keep, axis=1)
    for u, v in ARM_BLOCKS:
        old = out[v:v + 16, u:u + 16].copy()
        new = np.zeros_like(old)
        new[0:4, 4:7] = drop(old[0:4, 4:8], 2)       # top
        new[0:4, 7:10] = drop(old[0:4, 8:12], 2)     # bottom (the hand's underside)
        new[4:16, 0:4] = old[4:16, 0:4]              # outer side
        new[4:16, 4:7] = drop(old[4:16, 4:8], 2)     # front
        new[4:16, 7:11] = old[4:16, 8:12]            # inner side
        new[4:16, 11:14] = drop(old[4:16, 12:16], 1)  # back
        out[v:v + 16, u:u + 16] = new


def torso_bottom_from_legs(out: np.ndarray, only: str | None = None) -> None:
    """Give the torso's underside the leg tops' texture, so it reads as the legs carrying on up.
    With `only` ("left" or "right"), that leg's top is used for both halves, mirrored for the other
    (for a leg top with stray colours on it).

    The overlay layer is only copied where the leg overlays have something on top, and is left
    see-through when it would be solid all over: a full outer-layer underside shows as an extra slab
    hanging below the torso. Skins whose leg tops are see-through (the unused deliveryman textures)
    keep their own underside."""
    if any((out[y : y + 4, x : x + 4, 3] == 0).any() for x, y in (RIGHT_LEG_TOP[0], LEFT_LEG_TOP[0])):
        return
    for layer, ((tx, ty), (rx, ry), (lx, ly)) in enumerate(zip(TORSO_BOTTOM, RIGHT_LEG_TOP, LEFT_LEG_TOP)):
        right, left = out[ry : ry + 4, rx : rx + 4].copy(), out[ly : ly + 4, lx : lx + 4].copy()
        if only == "left":
            right = left[:, ::-1]
        elif only == "right":
            left = right[:, ::-1]
        if layer and not (right[..., 3] > 0).any() and not (left[..., 3] > 0).any():
            continue
        if layer and (right[..., 3] > 0).all() and (left[..., 3] > 0).all():
            out[ty : ty + 4, tx : tx + 8] = 0
            continue
        out[ty : ty + 4, tx : tx + 4] = right
        out[ty : ty + 4, tx + 4 : tx + 8] = left


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
                img = to_img(frame(load_rgba(path))).resize((w, h), Image.Resampling.NEAREST)
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
    manual, a, out = frame(manual), frame(a), frame(out)
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
    ap.add_argument("--train-all", action="store_true",
                    help="with --examples, train on every hand-edited skin like --all does, so re-rendered "
                         "examples match a full run")
    args = ap.parse_args()

    if args.all:
        names = sorted(p.name for p in ORIGINALS.glob("*.png"))
        holdout: set[str] = set()
    else:
        names = [f"{n}.png" for n in args.examples or []]
        # Hold the examples' whole outfit group out of training so the demo is honest.
        holdout = set() if args.train_all else {n[:-6] for n in names}

    overrides = load_overrides()
    model = SkinModel(holdout)
    hair_model = HairModel(holdout)
    out_dir = new_output_dir(args.out)
    checked, flagged, used_hand = 0, [], 0
    for name in names:
        full = load_rgba(ORIGINALS / name)
        a = frame(full)
        override = override_for(overrides, name[:-4])
        use_hand = not args.no_hand and override.get("hand", True)
        by_hand = hand_greyed(name, full) if use_hand else None
        out, skin, _ = dunmerify_with_masks(a, model, hair_model, override, by_hand, name[:-4])
        result = unframe(full, out)
        tint_outside(result, override.get("outside_tint_area", []))
        save(result, out_dir / name)
        used_hand += by_hand is not None
        check = manual_check(name, full, result, skin)
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
