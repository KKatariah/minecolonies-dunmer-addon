"""Package the add-on jar from the repo plus a folder of Dunmer skins.

Usage (from the repo root):
    python asset_dev/skin-editor/build_mod.py                      # newest 6.ears*, else 5.auto_full*
    python asset_dev/skin-editor/build_mod.py --skins asset_dev/skins_final
    python asset_dev/skin-editor/build_mod.py --target neoforge-1.21.1

Writes one jar per target to dist/:
    dunmercolonies-<version>-forge-1.20.1.jar      Forge 1.20.1 (also NeoForge 1.20.1)
    dunmercolonies-<version>-neoforge-1.21.1.jar   NeoForge 1.21.1
The repo's META-INF/mods.toml and pack.mcmeta are the Forge 1.20.1 ones; the NeoForge jar
gets META-INF/neoforge.mods.toml and pack.mcmeta made from them. MineColonies reads names
(data/*/citizennames) and skins (textures/entity/citizen/<style>) the same way on both.

Nothing in the repo is changed: the skins folder is copied into the jar as the `dunmer`
citizen style, replacing whatever is in assets/minecolonies/textures/entity/citizen/dunmer/
inside the jar only.

Citizen icons (the little heads in the town hall GUI) are generated the same way
MineColonies' own datagen does it: face area scaled to 16x16 with a dim border.
"""

import argparse
import io
import json
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
# Textures MineColonies still ships but no model uses: the deliveryman job wears the courier model,
# whose textures are named courier*.
UNUSED_SKINS = ("deliveryman",)
FORGE_TOML = "META-INF/mods.toml"
NEOFORGE_TOML = "META-INF/neoforge.mods.toml"
PACK_META = "pack.mcmeta"
TARGETS = ("forge-1.20.1", "neoforge-1.21.1")


def newest_run(pattern):
    runs = [p for p in ASSET_DEV.glob(pattern) if p.is_dir()]
    version = lambda p: int(m.group(1)) if (m := re.search(r"_v(\d+)$", p.name)) else 1
    return max(runs, key=version) if runs else None


def newest_full_run():
    run = newest_run("5.auto_full*")
    if not run:
        raise SystemExit("no asset_dev/5.auto_full* folder; run auto_dunmer.py --all first")
    return run


def newest_finished_run():
    """The newest skins with ears (6.ears*), else the newest full Dunmer run."""
    return newest_run("6.ears*") or newest_full_run()


def skin_files(folder):
    """The skins in a run folder, leaving out notes (_*.png) and textures no model uses."""
    return sorted(p for p in folder.glob("*.png") if not p.name.startswith("_") and not p.name.startswith(UNUSED_SKINS))


def mod_version():
    text = (REPO / "META-INF/mods.toml").read_text()
    return re.search(r'^version="([^"]+)"', text, re.M).group(1)


def neoforge_toml(text):
    """The Forge 1.20.1 mods.toml rewritten for NeoForge 1.21.1."""
    text = re.sub(r'^loaderVersion="[^"]*"', 'loaderVersion="[4,)"', text, flags=re.M)
    text = re.sub(r'^(\s*)mandatory=true', r'\1type="required"', text, flags=re.M)
    text = re.sub(r'^(\s*)mandatory=false', r'\1type="optional"', text, flags=re.M)
    text = re.sub(r'modId="forge"(\s*\n\s*type="required"\s*\n\s*)versionRange="[^"]*"',
                  r'modId="neoforge"\1versionRange="[21.1,)"', text)
    text = re.sub(r'(modId="minecraft"(?:\s*\n\s*\w+=.*)*?\s*\n\s*)versionRange="[^"]*"',
                  r'\1versionRange="[1.21.1,1.22)"', text)
    if 'modId="neoforge"' not in text or "[1.21.1," not in text:
        raise SystemExit(f"could not turn {FORGE_TOML} into a NeoForge 1.21.1 one; check its dependencies")
    return text


def neoforge_pack_meta(text):
    """pack.mcmeta for 1.21.1: resource pack format 34, data pack format 48."""
    meta = json.loads(text)
    meta["pack"]["pack_format"] = 34
    meta["pack"]["supported_formats"] = [34, 48]
    return json.dumps(meta, indent=2) + "\n"


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


def shown(path):
    """A path relative to the repo when it's inside it, for printing."""
    path = path.resolve()
    return path.relative_to(REPO) if path.is_relative_to(REPO) else path


def build_jar(target, skins, out):
    jar = out / f"dunmercolonies-{mod_version()}-{target}.jar"
    neoforge = target.startswith("neoforge")
    with zipfile.ZipFile(jar, "w", zipfile.ZIP_DEFLATED) as z:
        for part in MOD_PARTS:
            path = REPO / part
            files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
            for f in files:
                rel = f.relative_to(REPO).as_posix()
                if f.name in SKIP_NAMES or f.name.startswith("._") or rel.startswith(SKIN_DIR):
                    continue
                if neoforge and rel == FORGE_TOML:
                    z.writestr(NEOFORGE_TOML, neoforge_toml(f.read_text()))
                elif neoforge and rel == PACK_META:
                    z.writestr(PACK_META, neoforge_pack_meta(f.read_text()))
                else:
                    z.write(f, rel)
        for skin in skins:
            z.write(skin, SKIN_DIR + skin.name)
            with Image.open(skin) as im:
                z.writestr(ICON_DIR + skin.name, make_icon(im))
    return jar


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skins", type=Path, help="folder of finished skins (default: newest 6.ears*, else 5.auto_full*)")
    parser.add_argument("--target", choices=TARGETS + ("all",), default="all", help="which jar to build (default: all)")
    parser.add_argument("--out", type=Path, default=REPO / "dist")
    args = parser.parse_args()

    skins_dir = (args.skins or newest_finished_run()).resolve()
    skins = skin_files(skins_dir)
    if not skins:
        raise SystemExit(f"no skins in {skins_dir}")

    args.out.mkdir(parents=True, exist_ok=True)
    for target in TARGETS if args.target == "all" else (args.target,):
        jar = build_jar(target, skins, args.out)
        print(f"{len(skins)} skins from {shown(skins_dir)} -> {shown(jar)}")


if __name__ == "__main__":
    main()
