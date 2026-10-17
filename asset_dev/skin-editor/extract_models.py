"""Pull the citizen 3D models out of a MineColonies source checkout for the skin viewer.

Usage (from the repo root):
    python asset_dev/skin-editor/extract_models.py                     # ../minecolonies
    python asset_dev/skin-editor/extract_models.py --source path/to/minecolonies

Reads every Male*/Female*Model.java under core/client/model, follows the
createMesh() builder calls (parts, boxes, texture offsets, mirror, inflation,
poses) and the visibility switches in the constructor and setupAnim(), and writes
citizen_models.json next to this script. make_viewer.py renders skins with it.
Only needs re-running when MineColonies changes its models.
"""

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODEL_DIR = "src/main/java/com/minecolonies/core/client/model"
INIT_FILE = "src/main/java/com/minecolonies/apiimp/initializer/ModModelTypeInitializer.java"
IDS_FILE = "src/main/java/com/minecolonies/api/client/render/modeltype/ModModelTypes.java"

# HumanoidModel.createMesh(CubeDeformation.NONE, 0): the parts every citizen starts with.
HUMANOID = {
    "head": ([[0, 0, -4, -8, -4, 8, 8, 8, [0, 0, 0], False]], [0, 0, 0, 0, 0, 0]),
    "hat": ([[32, 0, -4, -8, -4, 8, 8, 8, [0.5] * 3, False]], [0, 0, 0, 0, 0, 0]),
    "body": ([[16, 16, -4, 0, -2, 8, 12, 4, [0, 0, 0], False]], [0, 0, 0, 0, 0, 0]),
    "right_arm": ([[40, 16, -3, -2, -2, 4, 12, 4, [0, 0, 0], False]], [-5, 2, 0, 0, 0, 0]),
    "left_arm": ([[40, 16, -1, -2, -2, 4, 12, 4, [0, 0, 0], True]], [5, 2, 0, 0, 0, 0]),
    "right_leg": ([[0, 16, -2, 0, -2, 4, 12, 4, [0, 0, 0], False]], [-1.9, 12, 0, 0, 0, 0]),
    "left_leg": ([[0, 16, -2, 0, -2, 4, 12, 4, [0, 0, 0], True]], [1.9, 12, 0, 0, 0, 0]),
}
# HumanoidModel renders these root parts; anything else at the root only shows if
# the model's renderToBuffer draws it itself.
RENDERED_ROOTS = list(HUMANOID)
FIELDS = {"head": "head", "hat": "hat", "body": "body", "rightArm": "right_arm",
          "leftArm": "left_arm", "rightLeg": "right_leg", "leftLeg": "left_leg"}


def split_args(s):
    """Split a Java argument list on top-level commas."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def call_args(s, start):
    """Given the index just after '(', return (args string, index after ')')."""
    depth = 1
    i = start
    while depth:
        depth += {"(": 1, ")": -1}.get(s[i], 0)
        i += 1
    return s[start:i - 1], i


def num(expr):
    e = expr.strip()
    e = re.sub(r"\(float\)\s*", "", e)
    e = re.sub(r"(?<=[\d.])[fFdD]\b", "", e)
    e = e.replace("Math.PI", "3.141592653589793")
    if not re.fullmatch(r"[\d.\s+\-*/()eE]+", e):
        raise ValueError(f"not a number: {expr!r}")
    return float(eval(e))


def deformation(expr, base=None):
    expr = expr.strip()
    if expr in ("CubeDeformation.NONE",):
        return [0.0, 0.0, 0.0]
    m = re.fullmatch(r"new CubeDeformation\((.*)\)", expr, re.S)
    if m:
        vals = [num(a) for a in split_args(m.group(1))]
        return vals * 3 if len(vals) == 1 else vals
    m = re.fullmatch(r"(.+)\.extend\((.*)\)", expr, re.S)
    if m:
        inner = deformation(m.group(1))
        vals = [num(a) for a in split_args(m.group(2))]
        vals = vals * 3 if len(vals) == 1 else vals
        return [a + b for a, b in zip(inner, vals)]
    raise ValueError(f"unknown deformation {expr!r}")


def cubes(expr):
    """Follow a CubeListBuilder chain."""
    out, u, v, mirror = [], 0, 0, False
    for m in re.finditer(r"\.(texOffs|mirror|addBox)\(", expr):
        args, _ = call_args(expr, m.end())
        a = split_args(args)
        if m.group(1) == "texOffs":
            u, v = int(num(a[0])), int(num(a[1]))
        elif m.group(1) == "mirror":
            mirror = (a[0] == "true") if a else True
        else:
            if a and a[0].startswith('"'):
                a = a[1:]
            box = [num(x) for x in a[:6]]
            rest = a[6:]
            grow, box_mirror, bu, bv = [0.0] * 3, mirror, u, v
            for r in rest:
                if r in ("true", "false"):
                    box_mirror = r == "true"
                elif "Deformation" in r or ".extend(" in r:
                    grow = deformation(r)
            ints = [x for x in rest if re.fullmatch(r"-?\d+", x)]
            if len(ints) == 2:  # addBox(name, x, y, z, w, h, d, [def], u, v)
                bu, bv = int(ints[0]), int(ints[1])
                u, v = bu, bv
            out.append([bu, bv, *box, grow, box_mirror])
    return out


def pose(expr):
    expr = expr.strip()
    if expr == "PartPose.ZERO":
        return [0.0] * 6
    m = re.fullmatch(r"PartPose\.(offset|offsetAndRotation|rotation)\((.*)\)", expr, re.S)
    if not m:
        raise ValueError(f"unknown pose {expr!r}")
    vals = [num(a) for a in split_args(m.group(2))]
    return {"offset": vals + [0.0] * 3, "offsetAndRotation": vals, "rotation": [0.0] * 3 + vals}[m.group(1)]


def method_body(src, name):
    m = re.search(r"\b" + name + r"\s*\([^)]*\)\s*\{", src)
    if not m:
        return ""
    depth, i = 1, m.end()
    while depth:
        depth += {"{": 1, "}": -1}.get(src[i], 0)
        i += 1
    return src[m.end():i - 1]


def statements(body):
    body = re.sub(r"//[^\n]*", "", body)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    return [s.strip() for s in body.split(";") if s.strip()]


def to_js(expr, locals_):
    """Turn a Java visibility/rotation condition into a JS expression over the
    viewer's switches: S.work (citizen at work), S.hat (wearing its hat), S.props
    (carrying job items)."""
    e = expr.strip()
    e = re.sub(r"isWorking\(\s*\w+\s*\)", "S.work", e)
    e = re.sub(r"displayHat\(\s*\w+\s*\)", "S.hat", e)
    e = re.sub(r"\w+\.getPose\(\)\s*!=\s*Pose\.SLEEPING", "true", e)
    e = re.sub(r"\w+\.getPose\(\)\s*==\s*Pose\.SLEEPING", "false", e)
    e = re.sub(r"!?\s*\w+\.getRenderMetadata\(\)\.contains\(\s*\w+\s*\)",
               lambda m: "!S.props" if m.group(0).startswith("!") else "S.props", e)
    e = re.sub(r"!?\s*\w+(\.\w+\([^()]*\))+", lambda m: "!S.props" if m.group(0).startswith("!") else "S.props", e)
    for name, js in locals_.items():
        e = re.sub(r"\b" + name + r"\b", f"({js})", e)
    e = re.sub(r"(?<=\d)[fF]\b", "", e)
    if not re.fullmatch(r"[\sS.workhatpropsruefl!&|()?:\d.\-]*", e):
        return None
    return e


def part_path(target, fields):
    """head.getChild("a").getChild("b") -> ["head", "a", "b"] (as part names)."""
    m = re.match(r"(?:this\.)?(\w+)((?:\.getChild\(\"\w+\"\))*)$", target.strip())
    if not m or m.group(1) not in fields:
        return None
    return fields[m.group(1)] + re.findall(r'getChild\("(\w+)"\)', m.group(2))


def parse_model(path):
    src = path.read_text()
    cls = path.stem
    parts = {name: {"parent": None, "cubes": c, "pose": p} for name, (c, p) in HUMANOID.items()}
    order = list(HUMANOID)
    vars_ = {}
    tex = [64, 32]
    for st in statements(method_body(src, "createMesh")):
        if "HumanoidModel.createMesh" in st:
            continue
        m = re.match(r"(?:PartDefinition\s+)?(\w+)\s*=\s*(\w+)\.getRoot\(\)$", st)
        if m:
            vars_[m.group(1)] = None
            continue
        m = re.match(r"(?:PartDefinition\s+)?(\w+)\s*=\s*(\w+)\.getChild\(\"(\w+)\"\)$", st)
        if m:
            vars_[m.group(1)] = m.group(3)
            continue
        m = re.match(r"(?:(?:PartDefinition\s+)?(\w+)\s*=\s*)?(\w+)\.addOrReplaceChild\(", st)
        if m:
            args, _ = call_args(st, m.end())
            name, cube_expr, pose_expr = split_args(args)
            name = name.strip('"')
            parent = vars_[m.group(2)] if m.group(2) in vars_ else None
            if name in parts and name in order:
                order.remove(name)
            parts[name] = {"parent": parent, "cubes": cubes(cube_expr), "pose": pose(pose_expr)}
            order.append(name)
            if m.group(1):
                vars_[m.group(1)] = name
            continue
        m = re.search(r"LayerDefinition\.create\(\s*\w+\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", st)
        if m:
            tex = [int(m.group(1)), int(m.group(2))]
            continue
        raise ValueError(f"{cls}: can't read statement: {st[:120]}")

    # A part replaced at the root is still a root part; children keep their place.
    for name in order:
        p = parts[name]
        if p["parent"] is not None and p["parent"] not in parts:
            raise ValueError(f"{cls}: {name} has unknown parent {p['parent']}")

    fields = {f: [p] for f, p in FIELDS.items()}
    hidden = []
    for st in statements(method_body(src, cls)):
        m = re.match(r"(?:this\.)?(\w+)\s*=\s*\w+\.getChild\(\"(\w+)\"\)$", st)
        if m:
            fields[m.group(1)] = [m.group(2)]
        m = re.match(r"(.+)\.visible\s*=\s*false$", st)
        if m and (pp := part_path(m.group(1), fields)):
            hidden.append(pp)

    rendered = list(RENDERED_ROOTS)
    for field in re.findall(r"(\w+)\.render\(", method_body(src, "renderToBuffer")):
        if field in fields and fields[field][0] not in rendered:
            rendered.append(fields[field][0])

    local_bools, rules, problems = {}, [], []
    for st in statements(method_body(src, "setupAnim")):
        m = re.match(r"(?:final\s+)?ModelPart\s+(\w+)\s*=\s*(.+)$", st)
        if m and (pp := part_path(m.group(2), fields)):
            fields[m.group(1)] = pp
            continue
        m = re.match(r"(?:final\s+)?boolean\s+(\w+)\s*=\s*(.+)$", st, re.S)
        if m:
            local_bools[m.group(1)] = to_js(m.group(2), local_bools) or "true"
            continue
        m = re.match(r"(.+?)\.visible\s*=\s*(.+)$", st, re.S)
        if m:
            pp = part_path(m.group(1), fields)
            js = to_js(m.group(2), local_bools)
            if pp is None or js is None:
                problems.append(st)
                continue
            rules.append([pp, js])

    lean = "0"
    ret = re.search(r"return\s+(.+?);", method_body(src, "getActualRotation"), re.S)
    if ret:
        lean = to_js(ret.group(1), local_bools) or "0"

    for p in problems:
        print(f"  {cls}: skipped {p[:100]}")
    return {
        "tex": tex,
        "parts": [{"name": n, **parts[n]} for n in order],
        "rendered": rendered,
        "hidden": hidden,
        "rules": rules,
        "lean": lean,
    }


def model_types(source):
    ids = dict(re.findall(r'(\w+_ID)\s*=\s*new ResourceLocation\([^,]+,\s*"(\w+)"\)', (source / IDS_FILE).read_text()))
    out = {}
    for m in re.finditer(r"new SimpleModelType\(ModModelTypes\.(\w+),\s*\d+,\s*new (\w+)(?:<>)?\(.*?\),\s*(?:new (\w+)\(|null)",
                         (source / INIT_FILE).read_text()):
        if m.group(1) in ids:
            out[ids[m.group(1)]] = {"male": m.group(2), "female": m.group(3) or m.group(2)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=REPO.parent / "minecolonies")
    parser.add_argument("--out", type=Path, default=HERE / "citizen_models.json")
    args = parser.parse_args()

    types = model_types(args.source)
    wanted = {c for t in types.values() for c in t.values() if c != "CitizenModel"}
    models = {}
    for cls in sorted(wanted):
        models[cls] = parse_model(args.source / MODEL_DIR / f"{cls}.java")
    args.out.write_text(json.dumps({"types": types, "models": models}, separators=(",", ":")))
    boxes = sum(len(p["cubes"]) for m in models.values() for p in m["parts"])
    print(f"{len(types)} model types, {len(models)} models, {boxes} boxes -> {args.out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
