"""Seeded Hammersley multi-view render in Mitsuba 3, emitting VoxHammer's Step-1 contract.

Runs in the `render` pixi environment:

    pixi run -e render python tools/render_hammersley.py --mesh body.glb --out work/t0/render --seed 1234
    pixi run -e render python tools/render_hammersley.py --mesh body.glb --out work/t0/aov --seed 1234 --views 64 --aov

Default layout (what voxhammer/extract_feature.py reads):
    NNN.png            512x512 RGBA, transparent background
    transforms.json    {"aabb","scale","offset","hammersley_seed","hammersley_offset","frames":[...]}
                       frames carry Blender-convention c2w (camera looks down local -Z, +Y up)
                       and camera_angle_x in radians, exactly as bpy_render.py:329-336 writes.
    mesh.ply           the mesh in Blender's z-up world, inside [-0.5,0.5]^3

--aov layout (what anny-render-corpus/score_render_pair.py reads):
    view_XXX.png       RGBA
    view_XXX.aov.npz   depth (H,W) float32, normal (H,W,3) float32
    view_XXX.json      camera record

The world is z-up because Blender's glTF importer converts y-up GLB to z-up
before bpy_render.py normalizes and orbits cameras about z; matching that frame
is what makes the two renderers interchangeable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

RADIUS = 2.0
FOV_DEG = 40.0
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    """Verbatim port of voxhammer/bpy_render.py:11-40. Returns [yaw, pitch]."""
    def radical_inverse(base, n):
        val, inv_base, inv_base_n = 0, 1.0 / base, 1.0 / base
        while n > 0:
            val += (n % base) * inv_base_n
            n //= base
            inv_base_n *= inv_base
        return val

    u = n / num_samples
    v = radical_inverse(PRIMES[0], n)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]


def camera_position(yaw, pitch, radius=RADIUS):
    return np.array([radius * np.cos(yaw) * np.cos(pitch),
                     radius * np.sin(yaw) * np.cos(pitch),
                     radius * np.sin(pitch)], dtype=np.float64)


def blender_c2w(pos: np.ndarray) -> np.ndarray:
    """TRACK_TO -Z / UP_Y at the origin: camera -Z points at the target, +Y toward world +Z."""
    fwd = -pos / np.linalg.norm(pos)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, true_up, -fwd, pos
    return m


def glb_yup_to_blender_zup(v: np.ndarray) -> np.ndarray:
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)


sys.path.insert(0, str(Path(__file__).resolve().parent))
import usd_io  # noqa: E402


def load_mesh_zup(path: Path):
    """(vertices z-up, faces, rgb float [0,1] or None) from .usda/.usdc, .glb, or .ply.

    USD is read with pxr straight into arrays -- no PLY conversion on the render path.
    USD and GLB are y-up (the canonical dress-on frame); PLY is taken as already z-up
    (it is what VoxHammer's voxelizer wrote).
    """
    suf = path.suffix.lower()
    rgb = None
    if suf in (".usda", ".usdc", ".usd"):
        v, f, rgb, up = usd_io.read_mesh(path)
        if up == "Y":
            v = glb_yup_to_blender_zup(v)
    else:
        m = trimesh.load(path, force="mesh", process=False)
        v, f = np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces)
        if getattr(m.visual, "kind", None) == "vertex":
            rgb = np.asarray(m.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0
        if suf in (".glb", ".gltf"):
            v = glb_yup_to_blender_zup(v)
    return v, f, rgb


def mitsuba_mesh(mi, v: np.ndarray, f: np.ndarray, rgb, reflectance_grey: float = 0.8):
    """In-memory Mitsuba mesh; vertex colours become the diffuse albedo when present."""
    bsdf = mi.load_dict({"type": "diffuse",
                         "reflectance": ({"type": "mesh_attribute", "name": "vertex_color"} if rgb is not None
                                         else {"type": "rgb", "value": [reflectance_grey] * 3})})
    props = mi.Properties()
    props["bsdf"] = bsdf
    mesh = mi.Mesh("body", vertex_count=len(v), face_count=len(f), props=props,
                   has_vertex_normals=False, has_vertex_texcoords=False)
    if rgb is not None:
        mesh.add_attribute("vertex_color", 3, mi.Float(np.ascontiguousarray(rgb, dtype=np.float32).ravel()))
    params = mi.traverse(mesh)
    params["vertex_positions"] = mi.Float(np.ascontiguousarray(v, dtype=np.float32).ravel())
    params["faces"] = mi.UInt32(np.ascontiguousarray(f, dtype=np.uint32).ravel())
    params.update()
    return mesh


def build_scene(mi, mesh_obj, resolution: int, aov: bool):
    # hide_emitters: the two area panels are real geometry (as in Blender) but must not
    # be seen by camera rays, or they opaque the alpha that extract_feature.py:46
    # premultiplies. Point lights were tried and rejected: they do not exist.
    integrator = {"type": "path", "max_depth": 4, "hide_emitters": True}
    if aov:
        integrator = {"type": "aov", "aovs": "depth:depth,nn:sh_normal", "inner": integrator}
    # Blender radiometry, bpy_render.py:112-134, converted to Mitsuba units:
    #   point 1000 W            -> radiant intensity 1000 / 4pi           = 79.6 W/sr
    #   area 10000 W, 100x100 m -> radiance 10000 / (pi * 10000 m^2)      = 0.318 W/sr/m^2
    #   area 1000 W, 0.25 m sq  -> radiance 1000 / (pi * 0.0625 m^2)     = 5093 W/sr/m^2
    #   world background 0.05 grey (Cycles default) -> constant emitter 0.05
    scene = {
        "type": "scene",
        "integrator": integrator,
        "mesh": mesh_obj,
    }
    if aov:
        # Unlit: AOVs need no light, and any emitter hit would be sample-averaged into
        # edge pixels' depth. With geometry only, depth = alpha * true depth exactly,
        # and score_candidates.py divides it back out.
        return mi.load_dict({**scene, "sensor": _sensor(mi, resolution)})
    scene.update({
        "key": {"type": "point", "position": [4.0, 1.0, 6.0],
                "intensity": {"type": "rgb", "value": [1000.0 / (4 * np.pi)] * 3}},
        "top": {"type": "rectangle",
                "to_world": mi.ScalarTransform4f().translate([0, 0, 10]).rotate([1, 0, 0], 180).scale([50, 50, 1]),
                "emitter": {"type": "area", "radiance": {"type": "rgb", "value": [10000.0 / (np.pi * 1e4)] * 3}}},
        "bottom": {"type": "rectangle",
                   "to_world": mi.ScalarTransform4f().translate([0, 0, -10]).scale([0.125, 0.125, 1]),
                   "emitter": {"type": "area", "radiance": {"type": "rgb", "value": [1000.0 / (np.pi * 0.0625)] * 3}}},
        "world": {"type": "constant", "radiance": {"type": "rgb", "value": [0.05] * 3}},
        "sensor": _sensor(mi, resolution),
    })
    return mi.load_dict(scene)


def _sensor(mi, resolution: int):
    return {
        "type": "perspective", "fov": FOV_DEG, "fov_axis": "x", "near_clip": 0.01, "far_clip": 100.0,
        "to_world": mi.ScalarTransform4f().look_at(origin=[0, 0, RADIUS], target=[0, 0, 0], up=[0, 1, 0]),
        "film": {"type": "hdrfilm", "width": resolution, "height": resolution,
                 "pixel_format": "rgba", "rfilter": {"type": "box"}},
        "sampler": {"type": "independent", "sample_count": 64},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--views", type=int, default=150)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--aov", action="store_true", help="emit the score_render_pair.py layout")
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--keep-frame", action="store_true",
                    help="do not re-normalize (pass >= 2: input is already in the canonical frame)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import mitsuba as mi
    for variant in ("cuda_ad_rgb", "llvm_ad_rgb", "scalar_rgb"):
        try:
            mi.set_variant(variant)
            break
        except Exception:
            continue
    print(f"[render] mitsuba {mi.__version__} variant={mi.variant()}")

    v, f, rgb = load_mesh_zup(a.mesh)
    if a.aov:
        rgb = None
    lo, hi = v.min(0), v.max(0)
    if a.keep_frame:
        # Pass >= 2: the input is a previous pass's output, already in the canonical
        # frame the masks were built in. Re-normalizing drifts the frame by a few mm
        # per pass; keep it, and only refuse if it has left the unit cube.
        scale, offset = 1.0, np.zeros(3)
        if (hi - lo).max() > 1.0 + 1e-3 or np.abs(lo).max() > 0.5 + 1e-3 or np.abs(hi).max() > 0.5 + 1e-3:
            print(f"[render] --keep-frame but mesh outside unit cube: {lo} {hi}"); return 2
    else:
        # Same normalization as bpy_render.normalize_scene (scale then recentre), so
        # mesh.ply and voxels.ply match Blender's even when the input is a hair off unit.
        scale = 1.0 / float((hi - lo).max())
        v = v * scale
        lo2, hi2 = v.min(0), v.max(0)
        offset = -(lo2 + hi2) / 2
        v = v + offset
    # mesh.usda is the side-output VoxHammer's voxelizer reads (the driver patches
    # extract_feature.voxelize_mesh to take it); the render takes the arrays in memory.
    # Written z-up with upAxis=Z so the file states its own convention.
    usd_io.write_mesh(a.out / "mesh.usda", v, f, None, up="Z", frame="voxhammer-zup-unit", source=a.mesh.name)
    print(f"[render] {'kept frame' if a.keep_frame else 'normalized'} scale={scale:.5f} offset={np.round(offset, 4).tolist()}"
          f"  source={a.mesh.suffix}{'  vertex colours' if rgb is not None else ''}")

    # Legacy global RNG on purpose: bpy_render.py:304 draws `np.random.rand()` twice, so a
    # Blender run preceded by np.random.seed(seed) gets these exact offsets and the two
    # renderers share one camera set. default_rng would not reproduce it.
    np.random.seed(a.seed)
    offset = (float(np.random.rand()), float(np.random.rand()))
    scene = build_scene(mi, mitsuba_mesh(mi, v, f, rgb), a.resolution, a.aov)
    params = mi.traverse(scene)
    sensor_key = next(k for k in params.keys() if k.endswith("to_world") and "sensor" in k.lower() or k == "sensor.to_world")

    frames = []
    for i in range(a.views):
        yaw, pitch = sphere_hammersley_sequence(i, a.views, offset)
        pos = camera_position(yaw, pitch)
        params[sensor_key] = mi.ScalarTransform4f().look_at(origin=pos.tolist(), target=[0, 0, 0], up=[0, 0, 1])
        params.update()
        img = mi.render(scene, spp=a.spp, seed=i)
        arr = np.asarray(img)
        rgba = np.clip(arr[..., :4], 0, 1)
        rgba[..., :3] = rgba[..., :3] ** (1 / 2.2)
        png = (rgba * 255).astype(np.uint8)
        from PIL import Image
        c2w = blender_c2w(pos)
        if a.aov:
            stem = f"view_{i:03d}"
            Image.fromarray(png, "RGBA").save(a.out / f"{stem}.png")
            depth = arr[..., 4].astype(np.float32)
            normal = arr[..., 5:8].astype(np.float32)
            np.savez_compressed(a.out / f"{stem}.aov.npz", depth=depth, normal=normal)
            (a.out / f"{stem}.json").write_text(json.dumps({"index": i, "yaw": yaw, "pitch": pitch,
                                                              "position": pos.tolist(), "fov_rad": np.deg2rad(FOV_DEG),
                                                              "transform_matrix": c2w.tolist()}))
        else:
            Image.fromarray(png, "RGBA").save(a.out / f"{i:03d}.png")
        frames.append({"file_path": f"{i:03d}.png", "camera_angle_x": float(np.deg2rad(FOV_DEG)),
                       "transform_matrix": c2w.tolist()})
        if i % 25 == 0 or i == a.views - 1:
            print(f"[render] view {i + 1}/{a.views}  alpha coverage {float((png[..., 3] > 0).mean()):.1%}")

    if not a.aov:
        (a.out / "transforms.json").write_text(json.dumps({
            "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], "scale": 1.0, "offset": [0.0, 0.0, 0.0],
            "hammersley_seed": a.seed, "hammersley_offset": list(offset), "frames": frames}, indent=2))
    print(f"[render] wrote {a.views} views to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
