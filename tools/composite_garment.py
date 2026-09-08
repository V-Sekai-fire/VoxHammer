"""2D try-on: paste the matted garment over the body render inside the mask bbox.

Runs in the `matting` pixi environment:

    pixi run -e matting python tools/composite_garment.py --images work/t0/images_front \
        --garment work/g0/garment-0-front.jpg --alpha work/g0/garment-0-front.alpha.png

Reads 2d_render.png + 2d_mask.png from --images, writes 2d_edit.png there. The
garment's alpha bbox is fitted uniformly into the mask bbox and centred; the
output keeps RGBA with alpha = max(body alpha, pasted garment alpha) so a
garment wider than the body (a skirt) can grow the silhouette. Pass-1 scope is
silhouette + palette, which is what an alpha paste carries.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--garment", type=Path, required=True)
    ap.add_argument("--alpha", type=Path, required=True, help="BiRefNet alpha PNG for the garment")
    ap.add_argument("--fill", type=float, default=1.0, help="fraction of the mask bbox the garment fills")
    ap.add_argument("--alpha-threshold", type=int, default=128)
    ap.add_argument("--rotation", type=int, default=90, help="CCW degrees; station photos are neckline-right")
    a = ap.parse_args()

    render = Image.open(a.images / "2d_render.png").convert("RGBA")
    mask = np.asarray(Image.open(a.images / "2d_mask.png").convert("L"))
    g_rgb = Image.open(a.garment).convert("RGB")
    g_a = Image.open(a.alpha).convert("L").resize(g_rgb.size, Image.BILINEAR)
    # Binarize the matte: BiRefNet returns soft alpha on sheer fabric, and the photo
    # station's grid shows through it. Pass-1 scope is silhouette + palette, so the
    # garment is treated as opaque; a small closing removes grid-line pinholes.
    import cv2
    ga = (np.asarray(g_a) >= a.alpha_threshold).astype(np.uint8) * 255
    ga = cv2.morphologyEx(ga, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    g_a = Image.fromarray(ga, "L")
    ys, xs = np.nonzero(ga > 127)
    if len(xs) == 0:
        print("[composite] garment alpha is empty"); return 2
    gx0, gx1, gy0, gy1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop_rgb = g_rgb.crop((gx0, gy0, gx1, gy1))
    crop_a = g_a.crop((gx0, gy0, gx1, gy1))
    # The photo station lays every garment flat with the neckline to the RIGHT (checked on
    # garments 0, 1, 2 -- a folded-sleeve sweater gave a square crop, so an aspect-ratio
    # heuristic is wrong). Rotate 90 deg CCW unconditionally; --rotation overrides.
    rotation = a.rotation
    if rotation:
        crop_rgb = crop_rgb.rotate(rotation, expand=True)
        crop_a = crop_a.rotate(rotation, expand=True)

    mys, mxs = np.nonzero(mask > 127)
    if len(mxs) == 0:
        print("[composite] 2d_mask.png is empty"); return 2
    mx0, mx1, my0, my1 = mxs.min(), mxs.max() + 1, mys.min(), mys.max() + 1
    bh = (my1 - my0) * a.fill
    # Fit by torso HEIGHT; width may overflow the body (a dress is wider than a torso).
    s = bh / crop_rgb.height
    nw, nh = max(1, int(crop_rgb.width * s)), max(1, int(crop_rgb.height * s))
    crop_rgb = crop_rgb.resize((nw, nh), Image.LANCZOS)
    crop_a = crop_a.resize((nw, nh), Image.BILINEAR)
    # Align the garment's alpha CENTROID (not its bbox centre) with the torso centre:
    # an asymmetric garment -- a sheer panel on one side -- otherwise hangs off-axis.
    ca = np.asarray(crop_a) > 127
    cys, cxs = np.nonzero(ca)
    gcx = float(cxs.mean()) if len(cxs) else nw / 2
    ox, oy = int((mx0 + mx1) / 2 - gcx), int((my0 + my1) / 2 - nh / 2)

    layer = Image.new("RGBA", render.size, (0, 0, 0, 0))
    layer.paste(crop_rgb, (ox, oy), mask=crop_a)
    out = Image.alpha_composite(render, layer)
    out_a = np.maximum(np.asarray(render)[..., 3], np.asarray(layer)[..., 3])
    out.putalpha(Image.fromarray(out_a, "L"))
    out.save(a.images / "2d_edit.png")

    pasted = np.asarray(layer)[..., 3] > 127
    # Placement gate: the garment must sit on the torso's VERTICAL span and be centred
    # on it. Width is free -- a dress is wider than a torso and is meant to overflow.
    rows = np.zeros_like(mask, dtype=bool)
    rows[my0:my1, :] = True
    inside_rows = float((pasted & rows).sum() / max(1, pasted.sum()))
    torso = mask > 127
    # The gate that matters: does the garment cover the torso? Centre alignment is by
    # construction now, so it is reported, not gated.
    torso_coverage = float((pasted & torso).sum() / max(1, torso.sum()))
    inside_mask = float((pasted & torso).sum() / max(1, pasted.sum()))
    rec = {"composite_method": "alpha-paste-fit-mask-height-centroid", "garment": a.garment.name,
           "rotation_deg": rotation, "alpha_threshold": a.alpha_threshold,
           "scale": s, "placed_at": [ox, oy], "size": [nw, nh], "pasted_pixels": int(pasted.sum()),
           "fraction_in_torso_rows": inside_rows, "torso_coverage": torso_coverage,
           "fraction_inside_body_mask": inside_mask}
    (a.images / "composite.json").write_text(json.dumps(rec, indent=2))
    print(f"[composite] {a.garment.name} rot={rotation} -> 2d_edit.png  {nw}x{nh} at ({ox},{oy})  "
          f"{inside_rows:.1%} in torso rows, covers {torso_coverage:.1%} of torso, "
          f"{inside_mask:.1%} of garment on body")
    # Placement is the gate. Coverage is garment-dependent (a vest covers less than a
    # dress) and is recorded on the row for the judge, not gated here.
    return 0 if inside_rows >= 0.9 else 3


if __name__ == "__main__":
    sys.exit(main())
