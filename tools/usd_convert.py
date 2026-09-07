"""Canonical OpenUSD assets for a dress-on target (operator: mesh operators live in USD).

Runs in the `render` pixi environment (usd-core):

    pixi run -e render python tools/usd_convert.py --target work/t000

For every *.glb under the target: a sibling .usda with a UsdGeom.Mesh prim (points,
faceVertexCounts/Indices, and a vertex-interpolated primvars:displayColor when the
GLB carries COLOR_0). The two torso masks become UsdGeom.Cube prims with a
translate + scale xform from phenotype.json bounds. Stage up-axis Y, metres per
unit recorded from the identity's unit_scale so the unit-cube frame is invertible.
GLB and PLY stay only where VoxHammer's voxelizer needs them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, Vt


def mesh_to_usda(glb: Path, out: Path, meters_per_unit: float) -> dict:
    m = trimesh.load(glb, force="mesh", process=False)
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    root = UsdGeom.Xform.Define(stage, "/root")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/root/mesh")
    v = np.asarray(m.vertices, dtype=np.float32)
    f = np.asarray(m.faces, dtype=np.int32)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*m.bounds[0].tolist()), Gf.Vec3f(*m.bounds[1].tolist())]))
    rec = {"vertices": int(len(v)), "faces": int(len(f)), "display_color": False}
    if getattr(m.visual, "kind", None) == "vertex":
        c = (np.asarray(m.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0)
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray.FromNumpy(c))
        rec["display_color"] = True
    mesh.GetPrim().SetCustomDataByKey("source", glb.name)
    stage.GetRootLayer().Save()
    return rec


def cubes_to_usda(ph: dict, out: Path, meters_per_unit: float) -> dict:
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    root = UsdGeom.Xform.Define(stage, "/masks")
    stage.SetDefaultPrim(root.GetPrim())
    rec = {}
    for side in ("front", "back"):
        lo, hi = (np.asarray(ph[f"mask_{side}_bounds"][i], dtype=np.float64) for i in (0, 1))
        cube = UsdGeom.Cube.Define(stage, f"/masks/{side}")
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2).tolist()))
        xf.AddScaleOp().Set(Gf.Vec3f(*(hi - lo).tolist()))
        cube.GetPrim().SetCustomDataByKey("role", f"torso_{side}_edit_region")
        rec[side] = {"center": ((lo + hi) / 2).tolist(), "size": (hi - lo).tolist()}
    stage.GetRootLayer().Save()
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, required=True)
    a = ap.parse_args()
    ph = json.loads((a.target / "phenotype.json").read_text())
    mpu = 1.0 / float(ph.get("unit_scale", 1.0))  # one unit-cube unit = this many metres
    done = {}
    for glb in sorted(a.target.rglob("*.glb")):
        if glb.name.startswith("mask_"):
            continue
        out = glb.with_suffix(".usda")
        if out.exists():  # anny_body.py and the driver already write USD directly
            done[str(glb.relative_to(a.target))] = "present"
            continue
        done[str(glb.relative_to(a.target))] = mesh_to_usda(glb, out, mpu)
    if not (a.target / "mask_front.usda").exists():
        done["masks.usda"] = cubes_to_usda(ph, a.target / "masks.usda", mpu)
    (a.target / "usd_manifest.json").write_text(json.dumps({"meters_per_unit": mpu, "files": done}, indent=2))
    print(f"[usd] {len(done) - 1} meshes + masks.usda under {a.target} (metersPerUnit={mpu:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
