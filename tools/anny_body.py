"""Identity phenotypes -> body.glb (unit cube, y-up) + two half-torso mask GLBs.

Runs in the `anny` pixi environment:

    pixi run -e anny python tools/anny_body.py --identity work/identities/female-p50.json --out work/t000

VoxHammer's importer flattens to static geometry and deletes materials, so the
export is bare triangles. ANNY is z-up / forward -y; VoxHammer's shipped example
is y-up, so vertices are rotated (x,y,z)->(x,z,-y), after which forward is +z.
Both the body and the masks are pre-normalized into [-0.5,0.5]^3 in that frame,
so VoxHammer's own normalize_scene is ~identity and the masks stay aligned.
"""
from __future__ import annotations
import argparse, json, re, sys, warnings
from pathlib import Path

import numpy as np
import torch
import trimesh

warnings.filterwarnings("ignore", category=DeprecationWarning)

TORSO_RE = re.compile(r"spine|chest|breast|torso|clavicle", re.I)
PAD = 0.08


def build_model():
    import anny
    return anny.Anny(rig="anny", topology="anny", phenotypes="all",
                     local_changes="default", skinning_method="lbs").to(dtype=torch.float64)


def rest_vertices(model, pheno: dict[str, float]) -> np.ndarray:
    pose = torch.eye(4, dtype=torch.float64)[None, None].repeat(1, model.bone_count, 1, 1)
    kw = {k: torch.tensor([float(v)], dtype=torch.float64) for k, v in pheno.items()}
    with torch.no_grad():
        return model(pose_parameters=pose, phenotype_kwargs=kw)["vertices"][0].cpu().numpy()


def to_yup(v: np.ndarray) -> np.ndarray:
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)


def dominant_bone(model) -> np.ndarray:
    w = model.vertex_bone_weights.detach().cpu().numpy()
    idx = model.vertex_bone_indices.detach().cpu().numpy()
    return idx[np.arange(len(idx)), w.argmax(1)]


def box_glb(lo: np.ndarray, hi: np.ndarray, path: Path) -> list[list[float]]:
    b = trimesh.creation.box(extents=hi - lo, transform=trimesh.transformations.translation_matrix((lo + hi) / 2))
    trimesh.Scene(b).export(path)
    return b.bounds.tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity", type=Path, required=True, help="JSON from identity_appendix_e.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--torso-regex", default=TORSO_RE.pattern)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ident = json.loads(a.identity.read_text())

    model = build_model()
    labels = list(model.bone_labels)
    v = to_yup(rest_vertices(model, ident["phenotype"]))
    faces = model.faces.cpu().numpy()
    if faces.shape[1] == 4:
        faces = np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]], 0)

    lo, hi = v.min(0), v.max(0)
    center, scale = (lo + hi) / 2, 1.0 / float((hi - lo).max())
    vn = (v - center) * scale
    body = trimesh.Trimesh(vn, faces, process=False)
    trimesh.Scene(body).export(a.out / "body.glb")
    # Canonical asset is USD; the GLB is a convenience copy for viewers.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import usd_io
    usd_io.write_mesh(a.out / "body.usda", vn, faces, up="Y", meters_per_unit=1.0 / scale,
                      frame="unit-cube-yup-forward+z", source="anny:" + ident.get("identity_source", ""))

    rx = re.compile(a.torso_regex, re.I)
    dom = dominant_bone(model)
    torso_bones = [i for i, n in enumerate(labels) if rx.search(n)]
    sel = np.isin(dom, torso_bones)
    if sel.sum() < 100:
        print(f"[anny_body] torso regex matched {sel.sum()} vertices; bone labels are:\n  " + "\n  ".join(labels))
        return 2
    tlo, thi = vn[sel].min(0), vn[sel].max(0)
    ext = thi - tlo; tlo, thi = tlo - PAD * ext, thi + PAD * ext
    zmid = float(vn[sel][:, 2].mean())                 # coronal plane; forward is +z after to_yup
    front = box_glb(np.array([tlo[0], tlo[1], zmid]), thi, a.out / "mask_front.glb")
    back = box_glb(tlo, np.array([thi[0], thi[1], zmid]), a.out / "mask_back.glb")
    for side, b in (("front", front), ("back", back)):
        usd_io.write_cube(a.out / f"mask_{side}.usda", b[0], b[1], name=side, role=f"torso_{side}_edit_region",
                          up="Y", meters_per_unit=1.0 / scale, frame="unit-cube-yup-forward+z")

    meta = dict(identity=ident, vertex_count=int(len(vn)), face_count=int(len(faces)),
                anny_bounds_m=[lo.tolist(), hi.tolist()], unit_scale=scale, unit_center=center.tolist(),
                up_axis=[0, 1, 0], forward_axis=[0, 0, 1],
                torso_bones=[labels[i] for i in torso_bones], torso_vertex_count=int(sel.sum()),
                mask_front_bounds=front, mask_back_bounds=back, coronal_z=zmid)
    (a.out / "phenotype.json").write_text(json.dumps(meta, indent=2))
    print(f"[anny_body] body.glb {len(vn)} verts / {len(faces)} faces, bounds {np.round(body.bounds, 3).tolist()}")
    print(f"[anny_body] torso {sel.sum()} verts via {len(torso_bones)} bones; front z>{zmid:.3f}, back z<{zmid:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
