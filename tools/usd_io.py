"""OpenUSD read/write for the dress-on file contracts (operator: mesh operators live in USD).

Three prim kinds cover every artefact that crosses a tool boundary:
  Mesh   -- bodies and dressed outputs (points, triangle faces, optional vertex displayColor)
  Cube   -- torso edit regions (size 1 + translate/scale xform)
  Points -- voxel coordinate sets VoxHammer passes between its own steps

Every stage records upAxis and metersPerUnit, and a `frame` customData string, so a
reader never has to guess the convention (rule 6: conventions are data).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, Vt


def _stage(path: Path, up: str, meters_per_unit: float, frame: str, default_prim: str) -> Usd.Stage:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y if up.upper() == "Y" else UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, float(meters_per_unit))
    root = UsdGeom.Xform.Define(stage, f"/{default_prim}")
    root.GetPrim().SetCustomDataByKey("frame", frame)
    stage.SetDefaultPrim(root.GetPrim())
    return stage


def write_mesh(path: Path, v: np.ndarray, f: np.ndarray, rgb: np.ndarray | None = None, *,
               up: str = "Y", meters_per_unit: float = 1.0, frame: str = "unit-cube", source: str = "") -> None:
    stage = _stage(path, up, meters_per_unit, frame, "root")
    mesh = UsdGeom.Mesh.Define(stage, "/root/mesh")
    v = np.ascontiguousarray(v, dtype=np.float32)
    f = np.ascontiguousarray(f, dtype=np.int32)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lo, hi = v.min(0), v.max(0)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())]))
    if rgb is not None:
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(rgb, dtype=np.float32)))
    if source:
        mesh.GetPrim().SetCustomDataByKey("source", source)
    stage.GetRootLayer().Save()


def read_mesh(path: Path):
    """(points float64 (N,3), faces int (M,3), rgb float32 (N,3) or None, up 'Y'|'Z')"""
    stage = Usd.Stage.Open(str(path))
    prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    if len(prims) != 1:
        raise ValueError(f"{path}: expected one Mesh prim, found {len(prims)}")
    mesh = UsdGeom.Mesh(prims[0])
    v = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if not np.all(counts == 3):
        raise ValueError(f"{path}: non-triangle faces")
    f = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    rgb = None
    pv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("displayColor")
    if pv and pv.IsDefined() and pv.GetInterpolation() == UsdGeom.Tokens.vertex:
        rgb = np.asarray(pv.Get(), dtype=np.float32)
    up = "Y" if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y else "Z"
    return v, f, rgb, up


def write_cube(path: Path, lo: np.ndarray, hi: np.ndarray, *, name: str = "region", role: str = "",
               up: str = "Y", meters_per_unit: float = 1.0, frame: str = "unit-cube") -> None:
    stage = _stage(path, up, meters_per_unit, frame, "masks")
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    cube = UsdGeom.Cube.Define(stage, f"/masks/{name}")
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2).tolist()))
    xf.AddScaleOp().Set(Gf.Vec3f(*(hi - lo).tolist()))
    if role:
        cube.GetPrim().SetCustomDataByKey("role", role)
    stage.GetRootLayer().Save()


def read_cube(path: Path):
    """(lo, hi) world-space bounds of the single Cube prim in the stage, plus up axis."""
    stage = Usd.Stage.Open(str(path))
    prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Cube)]
    if len(prims) != 1:
        raise ValueError(f"{path}: expected one Cube prim, found {len(prims)}")
    cube = UsdGeom.Cube(prims[0])
    size = float(cube.GetSizeAttr().Get())
    m = np.asarray(UsdGeom.Xformable(cube).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64)
    half = size / 2
    corners = np.array([[x, y, z, 1.0] for x in (-half, half) for y in (-half, half) for z in (-half, half)])
    w = (corners @ m)[:, :3]  # USD Gf matrices are row-vector convention
    up = "Y" if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y else "Z"
    return w.min(0), w.max(0), up


def write_points(path: Path, pts: np.ndarray, *, up: str = "Z", frame: str = "voxhammer-zup-unit") -> None:
    stage = _stage(path, up, 1.0, frame, "root")
    p = UsdGeom.Points.Define(stage, "/root/points")
    p.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(pts, dtype=np.float32)))
    stage.GetRootLayer().Save()


def read_points(path: Path) -> np.ndarray:
    stage = Usd.Stage.Open(str(path))
    prims = [x for x in stage.Traverse() if x.IsA(UsdGeom.Points)]
    if len(prims) != 1:
        raise ValueError(f"{path}: expected one Points prim, found {len(prims)}")
    return np.asarray(UsdGeom.Points(prims[0]).GetPointsAttr().Get(), dtype=np.float64)


def yup_to_zup(v: np.ndarray) -> np.ndarray:
    v = np.atleast_2d(v)
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)


def zup_to_yup(v: np.ndarray) -> np.ndarray:
    v = np.atleast_2d(v)
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)
