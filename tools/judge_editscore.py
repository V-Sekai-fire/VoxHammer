"""EditScore judge over dress-on candidates. Frozen-judge doctrine: the tuple
(base, adapter, precision, num_pass, prompt template) is fixed for the run and written
on every row; nothing about the judge changes while it scores.

Runs in the `judge` pixi environment:

    pixi run -e judge python tools/judge_editscore.py --targets "work/batch1/*__g*" [--num-pass 1]

For each target and candidate (rank1, rank3, rank5): evaluate([input front render,
candidate front render], instruction) where the instruction names the garment from
the dataset's own metadata (type, colours, category). A refusal is recorded as
refused=true with overall=-1, never as a score. Loader details follow
6-datasource/anny-render-corpus/score_edits.py (512^2 cap fits 8 GB; 1024 does not).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

from PIL import Image

BASE = "Qwen/Qwen3-VL-8B-Instruct"
ADAPTER = "EditScore/EditScore-Qwen3-VL-8B-Instruct"
PRECISION = "bf16"
SCORE_RANGE = 25
MAX_PIXELS = 262144
PROMPT = ("Dress this body in a {colors} {type} ({category}) like the garment in the source photo. "
          "Keep the body, pose and everything outside the torso unchanged.")
RANKS = ("rank1", "rank3", "rank5")


def cap(im: Image.Image) -> Image.Image:
    if im.width * im.height <= MAX_PIXELS:
        return im
    s = (MAX_PIXELS / (im.width * im.height)) ** 0.5
    return im.resize((int(im.width * s), int(im.height * s)), Image.BICUBIC)


def on_white(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB")


def instruction_for(target: Path) -> str:
    g = json.loads(next(target.glob("garment/garment-*.json")).read_text())
    colors = g.get("colors") or []
    colors = " and ".join(str(c).lower() for c in colors) if isinstance(colors, list) else str(colors).lower()
    return PROMPT.format(colors=colors or "plain", type=str(g.get("type", "garment")).lower(),
                         category=str(g.get("category", "")).lower())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="glob of target dirs")
    ap.add_argument("--num-pass", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    targets = sorted(Path(p) for p in glob.glob(a.targets) if Path(p).is_dir())
    if not targets:
        print("no targets"); return 2

    import torch
    from editscore import EditScore
    t0 = time.time()
    scorer = EditScore(backbone="qwen3vl", model_name_or_path=BASE, lora_path=ADAPTER,
                       score_range=SCORE_RANGE, num_pass=a.num_pass)
    print(f"[judge] loaded {BASE} + {ADAPTER} in {time.time() - t0:.0f}s, "
          f"weights {torch.cuda.memory_allocated() / 2 ** 30:.2f} GiB")
    prompt_sha = hashlib.sha256(PROMPT.encode()).hexdigest()[:16]

    for t in targets:
        instruction = instruction_for(t)
        src_path = t / "images_front" / "2d_render.png"
        if not src_path.exists():  # batch layout keeps the input front view under rank1's first pass
            src_path = next(t.glob("candidates/rank1/images_front/2d_render.png"))
        src = cap(on_white(src_path))
        for r in RANKS:
            out = t / "judge" / f"{r}.json"
            if out.exists() and not a.force:
                continue
            cand_png = t / "candidates" / r / "final_front" / "2d_render.png"
            if not cand_png.exists():
                print(f"[judge] {t.name}/{r}: no final_front render, skipping (counted)"); continue
            torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            try:
                res = scorer.evaluate([src, cap(on_white(cand_png))], instruction)
                overall = res.get("overall")
                refused = overall is None or not isinstance(overall, (int, float))
                raw = json.dumps(res)[:400]
            except Exception as exc:
                overall, refused, raw = None, True, f"{type(exc).__name__}: {exc}"
            rec = {"base": BASE, "adapter": ADAPTER, "precision": PRECISION, "num_pass": a.num_pass,
                   "score_range": SCORE_RANGE, "max_pixels": MAX_PIXELS, "prompt_sha": prompt_sha,
                   "instruction": instruction, "overall": None if refused else float(overall),
                   "refused": bool(refused), "seconds": round(time.time() - t1, 1),
                   "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), "raw": raw}
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rec, indent=2))
            print(f"[judge] {t.name:28s} {r}: overall={rec['overall']} refused={refused} {rec['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
