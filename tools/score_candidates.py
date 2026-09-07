"""Geometric MaskScore for dress-on candidates, split outside/inside the torso masks.

Runs in the `render` pixi environment:

    pixi run -e render python tools/score_candidates.py --reference work/t0/aov_rank5 \
        --candidate work/t0/aov_rank1 --phenotype work/t0/phenotype.json --out work/t0/scores/rank1.json

Metric per view is anny-render-corpus/score_render_pair.py:3-7 (depth L1, normal L1,
normal dot on the reference alpha), computed twice: on reference pixels OUTSIDE the
projected torso boxes (VoxHammer's "rest exact" guarantee as a number) and INSIDE
(how much the edit changed). Cameras come from each view's JSON, so the torso boxes
are projected exactly as pick_view.py does and nothing is re-rendered.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_view import box_corners, glb_yup_to_blender_zup, project  # noqa: E402


def load_aov(stem: Path):
    z = np.load(stem.with_suffix(".aov.npz"))
    depth = z["depth"].astype(np.float32)
    normal = z["normal"].astype(np.float32)
    alpha = np.asarray(Image.open(stem.with_suffix(".png")).convert("RGBA")).astype(np.float32)[:, :, 3] / 255.0
    # The unlit AOV pass averages surface hits with escaped samples, so a pixel's depth
    # is alpha * depth. Undo that where the pixel is counted (alpha > 0.5).
    sel = alpha > 0.5
    depth[sel] /= alpha[sel]
    normal[sel] /= alpha[sel, None]
    return depth, normal, alpha


def torso_mask(view: dict, ph: dict, w: int, h: int) -> np.ndarray:
    c2w = np.asarray(view["transform_matrix"])
    m = np.zeros((h, w), np.uint8)
    for side in ("front", "back"):
        corners = glb_yup_to_blender_zup(box_corners(ph[f"mask_{side}_bounds"]))
        uv = project(corners, c2w, view["fov_rad"], w, h)
        hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2).astype(np.int32)
        cv2.fillConvexPoly(m, hull, 255)
    return m > 0


def metrics(rd, rn, cd, cn, sel):
    n = int(sel.sum())
    if n == 0:
        return {"n": 0, "depth_l1": 0.0, "normal_l1": 0.0, "normal_dot": 1.0}
    rn_m = rn[sel] / (np.linalg.norm(rn[sel], axis=-1, keepdims=True) + 1e-9)
    cn_m = cn[sel] / (np.linalg.norm(cn[sel], axis=-1, keepdims=True) + 1e-9)
    return {"n": n,
            "depth_l1": float(np.abs(rd[sel] - cd[sel]).mean()),
            "normal_l1": float(np.abs(rn[sel] - cn[sel]).mean()),
            "normal_dot": float(np.clip((rn_m * cn_m).sum(-1), -1, 1).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True, help="AOV dir of the undressed input")
    ap.add_argument("--candidate", type=Path, required=True, help="AOV dir of the candidate")
    ap.add_argument("--phenotype", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ph = json.loads(a.phenotype.read_text())

    ref_views = sorted(p for p in a.reference.glob("view_*.json"))
    if not ref_views:
        print(f"no view_*.json in {a.reference}"); return 2
    rows = []
    for rj in ref_views:
        cj = a.candidate / rj.name
        if not cj.exists():
            print(f"missing candidate view {cj}"); return 2
        view = json.loads(rj.read_text())
        rd, rn, ra = load_aov(rj.with_suffix(""))
        cd, cn, _ = load_aov(cj.with_suffix(""))
        h, w = rd.shape
        covered = ra > 0.5
        tm = torso_mask(view, ph, w, h)
        row = {"view_index": view["index"]}
        for tag, sel in (("outside", covered & ~tm), ("inside", covered & tm), ("all", covered)):
            for k, v in metrics(rd, rn, cd, cn, sel).items():
                row[f"{k}_{tag}"] = v
        rows.append(row)

    def col(k):
        return np.array([r[k] for r in rows], dtype=np.float64)

    summary = {
        "n_views": len(rows),
        "max_depth_l1_all": float(col("depth_l1_all").max()),
        "max_depth_l1_outside": float(col("depth_l1_outside").max()),
        "mean_depth_l1_outside": float(col("depth_l1_outside").mean()),
        "mean_depth_l1_inside": float(col("depth_l1_inside").mean()),
        "mean_normal_dot_outside": float(col("normal_dot_outside").mean()),
        "mean_normal_dot_inside": float(col("normal_dot_inside").mean()),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"reference": a.reference.name, "candidate": a.candidate.name,
                                 "summary": summary, "views": rows}, indent=2))
    print(f"[score] {a.candidate.name} vs {a.reference.name}: {len(rows)} views  "
          f"outside depth_l1 max={summary['max_depth_l1_outside']:.5f} mean={summary['mean_depth_l1_outside']:.5f}  "
          f"inside depth_l1 mean={summary['mean_depth_l1_inside']:.5f}  "
          f"normal_dot out={summary['mean_normal_dot_outside']:.4f} in={summary['mean_normal_dot_inside']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
