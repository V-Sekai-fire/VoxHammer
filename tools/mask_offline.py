"""Foreground alpha from BiRefNet_HR-matting.

Run in the `matting` pixi environment (torch 2.5.1+cu124, no ONNX):

    pixi run -e matting python tools/mask_offline.py <in.png> <alpha.png>
    pixi run -e matting python tools/mask_offline.py --batch <in-glob> <out-dir>

Writes an HxW grayscale PNG that the edit env's rembg-shim provider reads as
its highest-trust foreground source.
"""
from __future__ import annotations
import argparse, glob, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

MODEL_ID = "ZhengPeng7/BiRefNet_HR-matting"


def load_model(device: str) -> torch.nn.Module:
    model = AutoModelForImageSegmentation.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(device).eval()
    if device.startswith("cuda"):
        model.half()  # README says all numbers were reported in FP16
    return model


def matte(model, img: Image.Image, device: str, size: int = 2048) -> np.ndarray:
    tx = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tx(img.convert("RGB")).unsqueeze(0).to(device)
    if device.startswith("cuda"): x = x.half()
    with torch.no_grad():
        pred = model(x)[-1].sigmoid().float().cpu()[0, 0]
    alpha = (pred.numpy() * 255).clip(0, 255).astype(np.uint8)
    return np.array(Image.fromarray(alpha).resize(img.size, Image.BILINEAR))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="single image, or a glob for --batch")
    ap.add_argument("output", type=Path, help="single .png, or a directory for --batch")
    ap.add_argument("--batch", action="store_true", help="input is a glob, output is a dir")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--size", type=int, default=2048)
    args = ap.parse_args()

    print(f"[matting] device={args.device} model={MODEL_ID} size={args.size}", flush=True)
    model = load_model(args.device)

    inputs = sorted(map(Path, glob.glob(str(args.input)))) if args.batch else [args.input]
    if not inputs:
        print(f"no inputs matched {args.input}", file=sys.stderr); return 2
    if args.batch:
        args.output.mkdir(parents=True, exist_ok=True)

    for src in inputs:
        img = Image.open(src)
        t0 = time.time()
        alpha = matte(model, img, args.device, size=args.size)
        dt = time.time() - t0
        dst = (args.output / f"{src.stem}.alpha.png") if args.batch else args.output
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(alpha, mode="L").save(dst)
        coverage = float((alpha > 127).mean())
        print(f"[matting] {src.name} -> {dst.name}  {img.size}  coverage={coverage:.1%}  {dt*1000:.0f}ms",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
