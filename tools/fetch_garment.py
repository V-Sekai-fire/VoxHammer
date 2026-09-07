"""Fetch one garment's three views + metadata from chibifire/zenodo-second-hand-fashion-v3.

Runs in the `matting` pixi environment:

    pixi run -e matting python tools/fetch_garment.py --garment-id 0 --out outputs/dress-on/inputs

Streams the dataset (no full download) and writes garment-<id>-{front,back,brand}.jpg
plus garment-<id>.json. Refuses garments missing any of the three views.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATASET = "chibifire/zenodo-second-hand-fashion-v3"
META = ("garment_id", "brand", "category", "type", "cut", "size", "pattern", "colors", "material", "condition", "is_gold", "station")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--garment-id", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    ds = load_dataset(DATASET, split="train", streaming=True)
    for r in ds:
        if int(r["garment_id"]) != a.garment_id:
            continue
        if any(r[f"image_{v}"] is None for v in ("front", "back", "brand")):
            print(f"garment {a.garment_id} lacks a view; refusing"); return 2
        for v in ("front", "back", "brand"):
            r[f"image_{v}"].convert("RGB").save(a.out / f"garment-{a.garment_id}-{v}.jpg", quality=95)
        (a.out / f"garment-{a.garment_id}.json").write_text(json.dumps({k: r.get(k) for k in META}, indent=2, default=str))
        print(f"[fetch] garment {a.garment_id}: {r.get('category')} / {r.get('type')} / {r.get('colors')} -> {a.out}")
        return 0
    print(f"garment {a.garment_id} not found in {DATASET}"); return 2


if __name__ == "__main__":
    sys.exit(main())
