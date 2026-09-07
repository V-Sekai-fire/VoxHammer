"""Driver for the VoxHammer edit path with an explicit foreground provider.

Upstream calls rembg to recover a foreground alpha. This environment replaces
rembg (see vendor-shims/rembg-shim), so the alpha has to be supplied. The
shim refuses to invent one, which is why this driver exists rather than a
default hidden inside the shim.

Provider precedence:
  1. an alpha channel already on the image  -- exact, no estimation
  2. background-luma thresholding           -- for renders on a flat backdrop
  3. (future) rf-detr segmentation head     -- for real photographs

Real garment photos need (3): 6-datasource/rf-detr-segmentation-data ships
CC BY 4.0 clothing instance masks across 47 categories, and BRIA RMBG is
blocklisted.
"""
import os, sys
# OpenUSD first: its Tf/Usd DLLs must load before torch/open3d bring their own runtime
# copies, or `import pxr` fails with "DLL load failed while importing _tf".
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
import usd_io  # noqa: E402

# Blender is off the path, and bpy cannot even coexist with pxr in one process (Blender
# bundles its own USD/TBB DLLs). voxhammer/delete_region_voxel.py imports bpy at module
# top but only touches it inside glb_to_ply, which is replaced below, so a stub satisfies
# the import and nothing from Blender is ever loaded.
import types as _types  # noqa: E402
if "bpy" not in sys.modules:
    _bpy = _types.ModuleType("bpy")
    for _n in ("ops", "context", "data", "types"):
        setattr(_bpy, _n, _types.SimpleNamespace())
    _bpy.__stub__ = "blender-off-the-path"
    sys.modules["bpy"] = _bpy
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import rembg  # noqa: E402


# Foreground-provider precedence, most trustworthy first:
#   1. sidecar alpha PNG at <image>.alpha.png       -- BiRefNet_HR-matting,
#                                                     produced offline in the
#                                                     `matting` pixi env by
#                                                     tools/mask_offline.py
#   2. genuine alpha channel on an RGBA input       -- exact, no estimation
#   3. backdrop-luma thresholding                   -- flat-backdrop fallback,
#                                                     good enough for renders
#
# The shim also has a real segmenter path for photographs (RF-DETR head, with
# CC BY 4.0 clothing masks in 6-datasource/rf-detr-segmentation-data). That
# routes through option 1 as well: precompute in `matting`, feed the PNG.
def foreground_alpha(img: Image.Image) -> np.ndarray:
    """HxW uint8 alpha, taken from the highest-trust source available."""
    if hasattr(img, "filename") and img.filename:
        from pathlib import Path
        sidecar = Path(img.filename).with_suffix(Path(img.filename).suffix + ".alpha.png")
        if sidecar.exists():
            a = np.array(Image.open(sidecar).convert("L").resize(img.size, Image.BILINEAR))
            print(f"[matting] alpha from sidecar {sidecar.name} coverage={(a>127).mean():.1%}")
            return a.astype(np.uint8)

    a = np.array(img)
    if img.mode == "RGBA" and a.shape[2] == 4 and (a[:, :, 3] < 255).any():
        return a[:, :, 3].astype(np.uint8)

    luma = np.array(img.convert("L"))
    border = np.concatenate([luma[0, :], luma[-1, :], luma[:, 0], luma[:, -1]])
    bg = int(np.median(border))
    return (np.abs(luma.astype(int) - bg) > 12).astype(np.uint8) * 255


rembg.set_mask_provider(foreground_alpha)
print(f"[driver] foreground provider registered: {foreground_alpha.__name__}")

import inference  # noqa: E402  (must follow provider registration)
import torch  # noqa: E402
import trimesh  # noqa: E402
from trellis.utils import postprocessing_utils  # noqa: E402

ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)  # postprocessing_utils.py:457
SH_C0 = 0.28209479177387814
_EXPORT_STATE = {"gaussian_path": None, "face_count": None}


def gaussian_vertex_color_glb(app_rep, mesh, **_):
    """Replacement for postprocessing_utils.to_glb without nvdiffrast.

    The Gaussian splat is the decoded appearance field; each mesh vertex takes the
    inverse-distance mean colour of its 8 nearest Gaussians (chunked GPU k-NN).
    Emits COLOR_0 vertex colours, which is what pixal3d.cpp's mesh_export does by
    default. postprocess_mesh (simplify + fill_holes) is skipped: fill_holes renders.
    """
    v = mesh.vertices.detach().float().cuda()
    f = mesh.faces.detach().cpu().numpy()
    xyz = app_rep.get_xyz.detach().float().cuda()
    rgb = (app_rep._features_dc.detach().float().cuda()[:, 0, :] * SH_C0 + 0.5).clamp(0, 1)
    cols = torch.empty((v.shape[0], 3), device="cuda")
    k = min(8, xyz.shape[0])
    for s in range(0, v.shape[0], 4096):
        d = torch.cdist(v[s:s + 4096], xyz)
        dk, ik = d.topk(k, dim=1, largest=False)
        w = 1.0 / (dk + 1e-6)
        cols[s:s + 4096] = (rgb[ik] * w[..., None]).sum(1) / w.sum(1, keepdim=True)
    rgb01 = cols.cpu().numpy()
    vc = np.concatenate([(rgb01 * 255).astype(np.uint8), np.full((v.shape[0], 1), 255, np.uint8)], 1)
    vy = v.cpu().numpy() @ ZUP_TO_YUP
    out = trimesh.Trimesh(vy, f, vertex_colors=vc, process=False)
    _EXPORT_STATE["face_count"] = int(len(f))
    _EXPORT_STATE["last_mesh"] = (vy, f, rgb01)
    if _EXPORT_STATE["gaussian_path"]:
        app_rep.save_ply(_EXPORT_STATE["gaussian_path"])
    print(f"[export] vertex-coloured mesh {len(v)} verts / {len(f)} faces from {xyz.shape[0]} gaussians")
    return out


postprocessing_utils.to_glb = gaussian_vertex_color_glb

# Blender is off the path (operator, 2026-09-07). The two places VoxHammer reaches for
# bpy are the Step-1 render (replaced by tools/render_hammersley.py, whose output is
# passed as --render_dir) and glb_to_ply for the mask, replaced here: Blender's glTF
# importer turns y-up GLB into z-up, then exports PLY, which trimesh does in two lines.
import voxhammer.delete_region_voxel as _drv  # noqa: E402


def glb_to_ply_trimesh(input_glb_path, input_ply_path):
    if not os.path.exists(input_glb_path):
        raise FileNotFoundError(f"Input GLB file not found: {input_glb_path}")
    m = trimesh.load(input_glb_path, force="mesh", process=False)
    v = np.asarray(m.vertices)
    zup = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
    trimesh.Trimesh(zup, np.asarray(m.faces), process=False).export(input_ply_path)
    print(f"[mask] {os.path.basename(input_glb_path)} -> {os.path.basename(input_ply_path)} via trimesh (z-up)")


_drv.glb_to_ply = glb_to_ply_trimesh

# PLY -> OpenUSD at every boundary (operator, 2026-09-07). Step 1's skip test and the
# voxelizer read mesh.usda; the mask arrives as a Cube prim; VoxHammer's own voxel
# point sets are written and read as UsdGeom.Points. Upstream files untouched.
import voxhammer.extract_feature as _ef  # noqa: E402
import utils3d  # noqa: E402


def run_3d_rendering_usd(input_model_path, render_dir, **_):
    if not (os.path.exists(os.path.join(render_dir, "transforms.json")) and os.path.exists(os.path.join(render_dir, "mesh.usda"))):
        sys.exit(f"[driver] {render_dir} lacks transforms.json + mesh.usda; there is no Blender fallback")
    print(f"[step1] using Mitsuba render in {render_dir}")
    return {"rendered": True, "num_views": 150, "output_dir": render_dir,
            "transforms_file": os.path.join(render_dir, "transforms.json"), "mesh_file": os.path.join(render_dir, "mesh.usda")}


def voxelize_mesh_usd(output_dir):
    import open3d as o3d
    v, f, _, _ = usd_io.read_mesh(os.path.join(output_dir, "mesh.usda"))
    v = np.clip(v, -0.5 + 1e-6, 0.5 - 1e-6)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f.astype(np.int32)))
    grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1 / 64,
                                                                          min_bound=(-0.5,) * 3, max_bound=(0.5,) * 3)
    idx = np.array([vx.grid_index for vx in grid.get_voxels()])
    assert np.all(idx >= 0) and np.all(idx < 64), "voxel index out of bounds"
    pts = (idx + 0.5) / 64 - 0.5
    out = os.path.join(output_dir, "voxels.usda")
    usd_io.write_points(out, pts)
    print(f"[voxelize] {len(pts)} voxels -> {out}")
    return out


def _usd_path(p):
    return os.path.splitext(p)[0] + ".usda"


def read_ply_usd(path):
    return (usd_io.read_points(_usd_path(path)),)


def write_ply_usd(path, vertices, *a, **k):
    usd_io.write_points(_usd_path(path), np.asarray(vertices))


inference.run_3d_rendering = run_3d_rendering_usd
_ef.voxelize_mesh = voxelize_mesh_usd

# extract_features (extract_feature.py:60-126) checks that mesh.ply exists and then
# never reads it; everything it does read goes through the shims above. Re-bind the
# reference function with that one filename swapped, and refuse if the swap is not
# exactly one occurrence -- upstream drift would otherwise pass silently.
import inspect as _inspect  # noqa: E402
_src = _inspect.getsource(_ef.extract_features)
# Its loader thread swallows exceptions and the main thread then blocks on the queue
# forever (measured: 30 min at 0 % CPU on a stale voxels.ply). Forward the exception.
_swaps = {
    '"mesh.ply"': '"mesh.usda"',
    '"voxels.ply"': '"voxels.usda"',
    'print(f"Error loading data: {e}")': 'print(f"Error loading data: {e}"); load_queue.put(e)',
    'data, positions = load_queue.get()': 'data, positions = _driver_unwrap(load_queue.get())',
}
for _old, _new in _swaps.items():
    assert _src.count(_old) == 1, f"extract_features no longer contains {_old} exactly once"
    _src = _src.replace(_old, _new)


def _driver_unwrap(got):
    if isinstance(got, BaseException):
        raise RuntimeError("feature loader failed") from got
    return got


_ns = dict(_ef.__dict__, _driver_unwrap=_driver_unwrap)
exec(compile(_src, _ef.__file__ + "#usd", "exec"), _ns)
_ef.extract_features = _ns["extract_features"]
inference.extract_features = _ef.extract_features

# inference.run_3d_editing checks that voxels.ply / voxels_delete.ply exist before the
# edit; the edit itself reads them through the read_ply shim. Same rebind, two names.
_src = _inspect.getsource(inference.run_3d_editing)
for _old in ('"voxels.ply"', '"voxels_delete.ply"'):
    assert _src.count(_old) >= 1, f"run_3d_editing no longer names {_old}"
    _src = _src.replace(_old, _old.replace(".ply", ".usda"))
_ns = dict(inference.__dict__)
exec(compile(_src, inference.__file__ + "#usd", "exec"), _ns)
inference.run_3d_editing = _ns["run_3d_editing"]
utils3d.io.read_ply = read_ply_usd
utils3d.io.write_ply = write_ply_usd

# Mask path: the edit region arrives as a UsdGeom.Cube (y-up canonical frame); the
# preset 64^3 grid and the deleted-voxel set are UsdGeom.Points. open3d's point-cloud
# I/O is wrapped so util_voxel_filtering.py's body runs unchanged on .usda paths.
import open3d as _o3d  # noqa: E402
import voxhammer.util_voxel_filtering as _uvf  # noqa: E402

_o3d_read_pcd, _o3d_write_pcd = _o3d.io.read_point_cloud, _o3d.io.write_point_cloud
PRESET_PLY = "assets/preset/preset_grid64.ply"
PRESET_USDA = "assets/preset/preset_grid64.usda"


def o3d_read_point_cloud_usd(path, *a, **k):
    if str(path).lower().endswith(".usda"):
        return _o3d.geometry.PointCloud(_o3d.utility.Vector3dVector(usd_io.read_points(path)))
    return _o3d_read_pcd(path, *a, **k)


def o3d_write_point_cloud_usd(path, pcd, *a, **k):
    if str(path).lower().endswith(".usda"):
        usd_io.write_points(path, np.asarray(pcd.points)); return True
    return _o3d_write_pcd(path, pcd, *a, **k)


def load_trimesh_usd(model_path):
    if str(model_path).lower().endswith(".usda"):
        lo, hi, up = usd_io.read_cube(model_path)
        box = trimesh.creation.box(extents=hi - lo, transform=trimesh.transformations.translation_matrix((lo + hi) / 2))
        v = usd_io.yup_to_zup(np.asarray(box.vertices)) if up == "Y" else np.asarray(box.vertices)
        return trimesh.Trimesh(v, np.asarray(box.faces), process=False)
    return _uvf.load_trimesh.__wrapped__(model_path)  # pragma: no cover


def process_delete_usd(input_mask_path, render_dir, filter_method="volume", voxel_size=1 / 64):
    if not os.path.exists(PRESET_USDA):
        pts = np.asarray(_o3d_read_pcd(PRESET_PLY).points)
        usd_io.write_points(PRESET_USDA, pts)
        print(f"[mask] converted {PRESET_PLY} -> {PRESET_USDA} ({len(pts)} points)")
    out = os.path.join(render_dir, "voxels_delete.usda")
    _uvf.process_voxels_with_improved_filtering(PRESET_USDA, input_mask_path, out,
                                                method=filter_method, voxel_size=voxel_size, inside=True)
    print(f"[mask] {os.path.basename(input_mask_path)} -> {out}")


_o3d.io.read_point_cloud = o3d_read_point_cloud_usd
_o3d.io.write_point_cloud = o3d_write_point_cloud_usd
_orig_load_trimesh = _uvf.load_trimesh
load_trimesh_usd.__wrapped__ = _orig_load_trimesh
_uvf.load_trimesh = lambda p: load_trimesh_usd(p) if str(p).lower().endswith(".usda") else _orig_load_trimesh(p)
_drv.process_delete_ply = process_delete_usd
if hasattr(inference, "process_delete_ply"):  # `from ... import process_delete_ply` binds a copy there
    inference.process_delete_ply = process_delete_usd

# Source reconstruction: VoxHammer's output is a re-decode of the latent, so geometry
# outside the mask can only match the SOURCE'S OWN DECODE, not the original mesh. That
# decode is the floor every control is measured against (CLAUDE.md rule 4), and it is
# what rank5 ("undressed") should be. Wrap feats_to_slat to export it once per run.
import voxhammer.edit_pipeline as _ep  # noqa: E402
_orig_feats_to_slat = _ep.feats_to_slat


def feats_to_slat_with_recon(pipeline, path):
    slat = _orig_feats_to_slat(pipeline, path)
    if _EXPORT_STATE.get("source_recon_path"):
        assets = pipeline.decode_slat(slat, ["gaussian", "mesh"])
        keep = _EXPORT_STATE["gaussian_path"]
        _EXPORT_STATE["gaussian_path"] = os.path.splitext(_EXPORT_STATE["source_recon_path"])[0] + "_gaussian.ply"
        gaussian_vertex_color_glb(assets["gaussian"][0], assets["mesh"][0]).export(_EXPORT_STATE["source_recon_path"])
        _EXPORT_STATE["gaussian_path"] = keep
        print(f"[export] source reconstruction -> {_EXPORT_STATE['source_recon_path']}")
    return slat


_ep.feats_to_slat = feats_to_slat_with_recon

if __name__ == "__main__":
    # inference.py has TWO entry points with different argument names:
    # main() (line ~230) takes --output_path, while the __main__ block
    # (line ~295) takes --output_dir and is the one that actually works.
    # Mirror the __main__ block rather than calling main().
    import argparse
    from trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_model", default="assets/example/model.glb")
    ap.add_argument("--mask_glb", default="assets/example/mask.glb")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--render_dir", default=None)
    ap.add_argument("--image_dir", default="assets/example/images")
    ap.add_argument("--is_text", type=bool, default=False)
    ap.add_argument("--source_prompt", default="")
    ap.add_argument("--target_prompt", default="")
    ap.add_argument("--seed", type=int, default=0, help="seeds numpy (Hammersley offset) and torch (samplers)")
    ap.add_argument("--output_name", default="output.glb")
    ap.add_argument("--export-source-recon", action="store_true",
                    help="also export the unedited source decode (<output>_source_recon.glb): the floor for controls")
    ap.add_argument("--delete-only", action="store_true",
                    help="only (re)write <render_dir>/voxels_delete.usda for --mask_glb; no models, no GPU")
    args = ap.parse_args()

    if args.delete_only:
        process_delete_usd(args.mask_glb, os.path.abspath(args.render_dir))
        sys.exit(0)

    import json, time
    t0 = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()

    name = "microsoft/TRELLIS-text-large" if args.is_text else "microsoft/TRELLIS-image-large"
    pipeline = (TrellisTextTo3DPipeline if args.is_text else TrellisImageTo3DPipeline).from_pretrained(name)
    pipeline.cuda()
    print(f"[driver] pipeline models: {sorted(pipeline.models.keys())}")
    t_load = time.time() - t0

    # Absolute render_dir: Blender resolves a RELATIVE render.filepath against
    # the blend-file dir (unset in bpy-as-a-module, so it falls back to the
    # drive root) while Python's open() uses the cwd -- the same path string
    # then writes PNGs and transforms.json to two different places.
    if not args.render_dir:
        sys.exit("[driver] --render_dir is required: render with tools/render_hammersley.py (Mitsuba). "
                 "There is no Blender fallback.")
    render_dir = os.path.abspath(args.render_dir)
    for needed in ("transforms.json", "mesh.usda"):
        if not os.path.exists(os.path.join(render_dir, needed)):
            sys.exit(f"[driver] {needed} missing in {render_dir}; nothing here renders with Blender.")
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.abspath(os.path.join(args.output_dir, args.output_name))
    _EXPORT_STATE["gaussian_path"] = os.path.splitext(output_path)[0] + "_gaussian.ply"
    if args.export_source_recon:
        _EXPORT_STATE["source_recon_path"] = os.path.splitext(output_path)[0] + "_source_recon.glb"

    t1 = time.time()
    inference.run_complete_pipeline(
        pipeline=pipeline,
        input_model_path=args.input_model,
        mask_glb_path=args.mask_glb,
        render_dir=render_dir,
        output_path=output_path,
        image_dir=args.image_dir,
        is_text=args.is_text,
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
    )
    timing = {"seed": args.seed, "model": name, "wall_load_s": round(t_load, 1),
              "wall_pipeline_s": round(time.time() - t1, 1), "wall_total_s": round(time.time() - t0, 1),
              "peak_vram_mib": int(torch.cuda.max_memory_allocated() / 2 ** 20),
              "output_face_count": _EXPORT_STATE["face_count"], "render_dir": render_dir,
              "input_model": args.input_model, "mask_glb": args.mask_glb, "image_dir": args.image_dir}
    # Canonical output is USD beside the GLB (Mesh prim, vertex displayColor, y-up unit frame).
    if _EXPORT_STATE.get("last_mesh"):
        vy, f, rgb01 = _EXPORT_STATE["last_mesh"]
        usd_io.write_mesh(os.path.splitext(output_path)[0] + ".usda", vy, f, rgb01, up="Y",
                          frame="unit-cube-yup-forward+z", source=f"voxhammer:{name}:seed{args.seed}")
    with open(os.path.splitext(output_path)[0] + ".timing.json", "w") as fh:
        json.dump(timing, fh, indent=2)
    print(f"[driver] done in {timing['wall_total_s']} s, peak VRAM {timing['peak_vram_mib']} MiB -> {output_path}")
