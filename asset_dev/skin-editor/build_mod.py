"""Package the add-on jar from the repo plus a folder of Dunmer skins.

Usage (from the repo root):
    python asset_dev/skin-editor/build_mod.py                      # newest asset_dev/5.auto_full*
    python asset_dev/skin-editor/build_mod.py --skins asset_dev/5.auto_full_v9

Writes dist/dunmercolonies-<version>.jar. Nothing in the repo is changed: the skins
folder is copied into the jar as the `dunmer` citizen style, replacing whatever is in
assets/minecolonies/textures/entity/citizen/dunmer/ inside the jar only.

Citizen icons (the little heads in the town hall GUI) are generated the same way
MineColonies' own datagen does it: face area scaled to 16x16 with a dim border.
"""

import argparse
import io
import re
import zipfile
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
ASSET_DEV = REPO / "asset_dev"
STYLE = "dunmer"
SKIN_DIR = f"assets/minecolonies/textures/entity/citizen/{STYLE}/"
ICON_DIR = f"assets/minecolonies/textures/entity_icon/citizen/{STYLE}/"
MOD_PARTS = ("META-INF", "pack.mcmeta", "data", "assets")
SKIP_NAMES = (".DS_Store",)


def newest_full_run():
    runs = [p for p in ASSET_DEV.glob("5.auto_full*") if p.is_dir()]
    if not runs:
        raise SystemExit("no asset_dev/5.auto_full* folder; run auto_dunmer.py --all first")
    version = lambda p: int(m.group(1)) if (m := re.search(r"_v(\d+)$", p.name)) else 1
    return max(runs, key=version)


def mod_version():
    text = (REPO / "META-INF/mods.toml").read_text()
    return re.search(r'^version="([^"]+)"', text, re.M).group(1)


def make_icon(skin):
    face = skin.convert("RGBA").crop((8, 8, 16, 16)).resize((16, 16), Image.NEAREST)
    icon = Image.new("RGBA", (16, 16), (0, 0, 0, 255))
    icon.alpha_composite(face)
    border = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    for i in range(16):
        for xy in ((0, i), (15, i), (i, 0), (i, 15)):
            border.putpixel(xy, (0, 0, 0, 128))
    icon.alpha_composite(border)
    out = io.BytesIO()
    icon.convert("RGB").save(out, "PNG")
    return out.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skins", type=Path, help="folder of finished skins (default: newest 5.auto_full*)")
    parser.add_argument("--out", type=Path, default=REPO / "dist")
    args = parser.parse_args()

    skins_dir = (args.skins or newest_full_run()).resolve()
    skins = sorted(p for p in skins_dir.glob("*.png") if not p.name.startswith("_"))
    if not skins:
        raise SystemExit(f"no skins in {skins_dir}")

    args.out.mkdir(parents=True, exist_ok=True)
    jar = args.out / f"dunmercolonies-{mod_version()}.jar"
    with zipfile.ZipFile(jar, "w", zipfile.ZIP_DEFLATED) as z:
        for part in MOD_PARTS:
            path = REPO / part
            files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
            for f in files:
                rel = f.relative_to(REPO).as_posix()
                if f.name in SKIP_NAMES or f.name.startswith("._") or rel.startswith(SKIN_DIR):
                    continue
                z.write(f, rel)
        for skin in skins:
            z.write(skin, SKIN_DIR + skin.name)
            with Image.open(skin) as im:
                z.writestr(ICON_DIR + skin.name, make_icon(im))

    print(f"{len(skins)} skins from {skins_dir.relative_to(REPO)} -> {jar.relative_to(REPO)}")


if __name__ == "__main__":
    main()
