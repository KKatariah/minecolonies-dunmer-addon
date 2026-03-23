# MineColonies Dunmer Addon

This is a resource-only Forge addon jar that provides:

- Name style: `dunmer`
- Citizen texture style: `dunmer`

## Current scaffold

- `META-INF/mods.toml`
- `pack.mcmeta`
- `data/dunmercolonies/citizennames/dunmer.json`
- `assets/minecolonies/textures/entity/citizen/dunmer/*`

## Important

The `.png` files in this scaffold are placeholders and must be replaced with real PNG image data.

## Build jar (from this folder)

```bash
jar --create --file dunmercolonies-1.0.0.jar -C . .
```

If `jar` is unavailable, use zip and rename:

```bash
zip -r dunmercolonies-1.0.0.jar .
```

## Install

1. Put the jar into your Minecraft `mods` folder.
2. Start game and load world.
3. In Town Hall, set texture style to `dunmer`.
4. Set name style to `dunmer`.
