"""Pick the front-most / rear-most Hammersley view and project a mask box into it.

Runs in the `render` pixi environment:

    pixi run -e render python tools/pick_view.py --render work/t0/render --phenotype work/t0/phenotype.json \
        --side front --out work/t0/images_front

Writes 2d_render.png (copy of the chosen frame), 2d_mask.png (the half-torso box
projected through that frame's camera, filled, clipped to the body alpha) and
view.json. Cameras come from transforms.json, so they match the render by
construction; nothing is re-rendered.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def glb_yup_to_blender_zup(v: np.ndarray) -> np.ndarray:
    v = np.atleast_2d(v)
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)


def box_corners(bounds) -> np.ndarray:
    lo, hi = np.asarray(bounds[0]), np.asarray(bounds[1])
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def project(points_w: np.ndarray, c2w: np.ndarray, fov_x: float, w: int, h: int) -> np.ndarray:
    w2c = np.linalg.inv(c2w)
    p = (w2c @ np.c_[points_w, np.ones(len(points_w))].T).T[:, :3]
    depth = -p[:, 2]                                   # Blender camera looks down -Z
    t = np.tan(fov_x / 2)
    x = p[:, 0] / depth / t
    y = p[:, 1] / depth / (t * h / w)
    u = (x + 1) / 2 * w
    v = (1 - y) / 2 * h
    return np.stack([u, v], 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", type=Path, required=True, help="dir with transforms.json + NNN.png")
    ap.add_argument("--phenotype", type=Path, required=True, help="phenotype.json from anny_body.py")
    ap.add_argument("--side", choices=("front", "back"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    tf = json.loads((a.render / "transforms.json").read_text())
    ph = json.loads(a.phenotype.read_text())
    forward = glb_yup_to_blender_zup(np.asarray(ph["forward_axis"], dtype=float))[0]
    forward /= np.linalg.norm(forward)
    sign = 1.0 if a.side == "front" else -1.0

    best, best_dot = None, -2.0
    for fr in tf["frames"]:
        c2w = np.asarray(fr["transform_matrix"])
        pos = c2w[:3, 3]
        d = sign * float(pos @ forward) / np.linalg.norm(pos)
        if d > best_dot:
            best, best_dot = fr, d
    c2w = np.asarray(best["transform_matrix"])
    src = a.render / best["file_path"]
    img = Image.open(src).convert("RGBA")
    w, h = img.size
    alpha = np.asarray(img)[..., 3]

    corners = glb_yup_to_blender_zup(box_corners(ph[f"mask_{a.side}_bounds"]))
    uv = project(corners, c2w, best["camera_angle_x"], w, h)
    hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    mask[alpha < 128] = 0

    shutil.copy(src, a.out / "2d_render.png")
    Image.fromarray(mask, "L").save(a.out / "2d_mask.png")
    rec = {"side": a.side, "frame": best["file_path"], "index": int(best["file_path"][:3]),
           "camera_position": c2w[:3, 3].tolist(), "forward_dot": best_dot,
           "mask_pixels": int((mask > 0).sum()), "mask_bbox": [int(v) for v in cv2.boundingRect(hull)],
           "body_pixels": int((alpha > 127).sum())}
    (a.out / "view.json").write_text(json.dumps(rec, indent=2))
    print(f"[pick_view] {a.side}: frame {best['file_path']} dot={best_dot:.3f} "
          f"mask {rec['mask_pixels']} px of body {rec['body_pixels']} px, bbox {rec['mask_bbox']}")
    if rec["mask_pixels"] == 0:
        print("[pick_view] empty mask -- box projected outside the body silhouette")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
