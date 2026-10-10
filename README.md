# MineColonies Dunmer Addon

This is a resource-only Forge addon jar (Minecraft 1.20.1, MineColonies) that provides:

- Name style: `dunmer`
- Citizen texture style: `dunmer`

## Build jar

```bash
python asset_dev/skin-editor/build_mod.py            # skins from the newest asset_dev/5.auto_full*
python asset_dev/skin-editor/build_mod.py --skins asset_dev/5.auto_full_v9
```

This writes `dist/dunmercolonies-<version>.jar` with the skins and their town hall icons.
Skins missing from the folder fall back to MineColonies' `default` style in game.

## Install

1. Put the jar into your Minecraft `mods` folder, next to MineColonies.
2. Start the game and load a world.
3. In the Town Hall, set the citizen style to `dunmer` and the name style to `dunmer`.
   MineColonies only enables these two pickers for its Patreon supporters.
