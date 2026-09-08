"""Negative control for rest-exact: regenerate the body from the same composite with
TRELLIS alone, nothing preserved.

Runs in the `default` (edit) pixi environment on the 4090:

    pixi run -e default env ATTN_BACKEND=xformers SPARSE_ATTN_BACKEND=xformers SPCONV_ALGO=native \
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \
        python tools/wholesale_control.py --image <target>/candidates/rank1/images_front/2d_edit.png \
        --out <target>/controls/wholesale/dressed_front.usda --seed 1000

VoxHammer itself cannot run with an empty preserved set (its SLat inversion reshapes
zero elements), so the "delete everything" edit is stood in for by the generator the
editor is built on, conditioned on the same 2d_edit.png. tools/voxel_controls.py reads
the result as the containment a wholesale regeneration achieves; the gate's tolerance
must sit between that and the decode floor.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import usd_io  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True, help="RGBA composite (alpha is the foreground)")
    ap.add_argument("--out", type=Path, required=True, help=".usda to write (z-up VoxHammer frame)")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--model", default="microsoft/TRELLIS-image-large")
    a = ap.parse_args()

    from PIL import Image
    import torch
    from trellis.pipelines import TrellisImageTo3DPipeline

    t0 = time.time()
    pipe = TrellisImageTo3DPipeline.from_pretrained(a.model)
    pipe.cuda()
    img = Image.open(a.image).convert("RGBA")
    if np.all(np.asarray(img)[:, :, 3] == 255):
        print("REFUSED: the composite has no alpha; TRELLIS would fall back to rembg"); return 2
    out = pipe.run(img, seed=a.seed, formats=["mesh"])
    mesh = out["mesh"][0]
    v = mesh.vertices.detach().cpu().numpy().astype(np.float32)
    f = mesh.faces.detach().cpu().numpy().astype(np.int32)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    usd_io.write_mesh(a.out, v, f, None, up="Z", frame="unit-cube-zup",
                      source=f"trellis-image-large:wholesale-regeneration:seed{a.seed}")
    print(f"[wholesale] {len(v)} verts {len(f)} faces in {time.time() - t0:.0f} s, "
          f"peak VRAM {torch.cuda.max_memory_allocated() // 2**20} MiB -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
