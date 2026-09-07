"""Dress-on batch orchestrator: identities x garments -> MaskScore rows. Stdlib only.

    python tools/dress_on_batch.py --out work/batch1 --garments 0 1 2 --identities all

Each step runs in its own pixi environment and hands files to the next
(one-pixi-env-per-usage). Per (identity, garment) target:

    anny env    identity_appendix_e.py (once)  anny_body.py
    matting env mask_offline.py (once per garment)
    render env  render_hammersley.py  pick_view.py
    matting env composite_garment.py
    default env run_edit_test.py  (pass 1 front, pass 2 back; rank1 own garment,
                                   rank3 wrong garment; rank5 = source decode)
    render env  render_hammersley.py --aov (x3)  score_candidates.py (x3)  usd_convert.py
    matting env write_dress_on_rows.py

A crash at target k leaves k-1 rows written. Every subprocess failure stops the
batch with the step named; nothing is skipped silently.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# HF_HUB_OFFLINE: every weight this pipeline loads is already in the cache; an online
# check per model load is a network wait on every step and, unauthenticated, a
# rate-limit stall. Nothing here needs the Hub at run time.
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
EDIT_ENV = {"ATTN_BACKEND": "xformers", "SPARSE_ATTN_BACKEND": "xformers", "SPCONV_ALGO": "native",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1", **OFFLINE}
GARMENT_INPUTS = REPO / "outputs" / "dress-on" / "inputs"
GARMENT_MATTED = REPO / "outputs" / "dress-on" / "matted"


def done(*outputs: Path) -> bool:
    """Resume support: a step whose outputs all exist is not re-run."""
    return all(p.exists() for p in outputs)


def run(env: str, *args: str, extra_env: dict | None = None, log: Path | None = None, outputs: tuple = ()) -> None:
    if outputs and done(*outputs):
        print(f"  [{env:7s}] {Path(str(args[0])).name:26s}   (exists, skipped)")
        return
    cmd = ["pixi", "run", "-e", env]
    # Every GPU step on the 4090 (PCI index 1). The 3090 is the operator's display/VR
    # card; Mitsuba's CUDA variant would otherwise take device 0.
    gpu = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1"}
    extra_env = {**OFFLINE, **gpu, **(extra_env or {})} if env in ("default", "matting", "judge", "render") else extra_env
    if extra_env:
        cmd += ["env"] + [f"{k}={v}" for k, v in extra_env.items()]
    cmd += ["python", *map(str, args)]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    tail = [l for l in proc.stdout.splitlines() if l.startswith("[") or l.startswith("ok ") or "REFUSED" in l][-3:]
    print(f"  [{env:7s}] {Path(str(args[0])).name:26s} {time.time() - t0:6.1f}s  {' | '.join(tail)[-160:]}")
    if proc.returncode != 0:
        print(proc.stdout[-2000:]); print(proc.stderr[-2000:])
        raise SystemExit(f"step failed ({env} {args[0]}), exit {proc.returncode}")


def garment_files(gid: int) -> dict:
    return {side: GARMENT_INPUTS / f"garment-{gid}-{side}.jpg" for side in ("front", "back", "brand")}


def ensure_garment(gid: int, work: Path) -> None:
    if not all(p.exists() for p in garment_files(gid).values()):
        run("matting", HERE / "fetch_garment.py", "--garment-id", str(gid), "--out", GARMENT_INPUTS)
    if not (GARMENT_MATTED / f"garment-{gid}-front.alpha.png").exists():
        run("matting", HERE / "mask_offline.py", "--batch", str(GARMENT_INPUTS / f"garment-{gid}-*.jpg"), GARMENT_MATTED)


def edit(target: Path, cand: str, body_glb: Path, render_dir: Path, gid: int, seed: int, recon: bool) -> Path:
    cdir = target / "candidates" / cand
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "garment_used.json").write_text(json.dumps({"garment_id": gid}))
    current, current_render = body_glb, render_dir
    for k, side in enumerate(("front", "back")):
        images = cdir / f"images_{side}"
        run("render", HERE / "pick_view.py", "--render", current_render, "--phenotype", target / "phenotype.json",
            "--side", side, "--out", images)
        run("matting", HERE / "composite_garment.py", "--images", images,
            "--garment", GARMENT_INPUTS / f"garment-{gid}-{side}.jpg",
            "--alpha", GARMENT_MATTED / f"garment-{gid}-{side}.alpha.png")
        out_dir = cdir / f"pass{k + 1}"
        args = [REPO / "run_edit_test.py", "--input_model", current, "--mask_glb", target / f"mask_{side}.usda",
                "--render_dir", current_render, "--image_dir", images, "--output_dir", out_dir,
                "--output_name", f"dressed_{side}.glb", "--seed", str(seed + k)]
        if recon and k == 0:
            args.append("--export-source-recon")
        run("default", *args, extra_env=EDIT_ENV, log=out_dir / "run.log",
            outputs=(out_dir / f"dressed_{side}.glb", out_dir / f"dressed_{side}_gaussian.ply"))
        current = out_dir / f"dressed_{side}.glb"
        if side == "front":
            current_render = cdir / "render_pass1"
            run("render", HERE / "render_hammersley.py", "--mesh", current, "--out", current_render,
                "--seed", str(seed), "--views", "150", "--keep-frame",
                outputs=(current_render / "transforms.json", current_render / "149.png", current_render / "mesh.usda"))
    shutil.copy2(current, cdir / "candidate.glb")
    if current.with_suffix(".usda").exists():
        shutil.copy2(current.with_suffix(".usda"), cdir / "candidate.usda")
    shutil.copy2(current.with_name(current.stem + "_gaussian.ply"), cdir / "candidate_gaussian.ply")
    shutil.copy2(current.with_name(current.stem + ".timing.json"), cdir / "timing.json")
    if recon:
        r5 = target / "candidates" / "rank5"
        r5.mkdir(parents=True, exist_ok=True)
        src = cdir / "pass1" / "dressed_front_source_recon.glb"
        shutil.copy2(src, r5 / "candidate.glb")
        shutil.copy2(src.with_name("dressed_front_source_recon_gaussian.ply"), r5 / "candidate_gaussian.ply")
        (r5 / "timing.json").write_text(json.dumps({"seed": seed, "wall_total_s": 0.0, "peak_vram_mib": 0, "output_face_count": 0}))
    return cdir / "candidate.glb"


def score_all(target: Path, seed: int) -> None:
    for cand in ("rank5", "rank1", "rank3"):
        cpath = target / "candidates" / cand
        mesh = cpath / "candidate.usda" if (cpath / "candidate.usda").exists() else cpath / "candidate.glb"
        run("render", HERE / "render_hammersley.py", "--mesh", mesh,
            "--out", target / "aov" / cand, "--seed", str(seed), "--views", "64", "--aov", "--spp", "32", "--keep-frame",
            outputs=(target / "aov" / cand / "view_063.json",))
    run("render", HERE / "render_hammersley.py", "--mesh", target / "body.usda", "--out", target / "aov" / "input",
        "--seed", str(seed), "--views", "64", "--aov", "--spp", "32", "--keep-frame")
    for cand in ("rank5", "rank1", "rank3"):
        run("render", HERE / "score_candidates.py", "--reference", target / "aov" / "input",
            "--candidate", target / "aov" / cand, "--phenotype", target / "phenotype.json",
            "--out", target / "scores" / f"{cand}.json")
        # Lit front/back views of the finished candidate: what the judge looks at.
        cpath = target / "candidates" / cand
        mesh = cpath / "candidate.usda" if (cpath / "candidate.usda").exists() else cpath / "candidate.glb"
        lit = cpath / "render_final"
        run("render", HERE / "render_hammersley.py", "--mesh", mesh, "--out", lit, "--seed", str(seed),
            "--views", "150", "--keep-frame", outputs=(lit / "transforms.json", lit / "149.png"))
        for side in ("front", "back"):
            run("render", HERE / "pick_view.py", "--render", lit, "--phenotype", target / "phenotype.json",
                "--side", side, "--out", cpath / f"final_{side}", outputs=(cpath / f"final_{side}" / "2d_render.png",))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--garments", type=int, nargs="+", required=True, help="garment_ids; rank3 uses the next one")
    ap.add_argument("--identities", nargs="+", default=["all"])
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--stage", type=Path, default=None)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--diagonal", action="store_true",
                    help="row i = identity i x garment i (distinct identities AND garments in a small batch)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    stage = a.stage or (a.out / "stage")

    idents = a.out / "identities"
    if not idents.exists():
        run("anny", HERE / "identity_appendix_e.py", "--out", idents, "--negative-control")
    ident_files = sorted(idents.glob("*.json"))
    if a.identities != ["all"]:
        ident_files = [p for p in ident_files if p.stem in a.identities]
    for gid in a.garments:
        ensure_garment(gid, a.out)

    if len(a.garments) < 2:
        sys.exit("need >= 2 garments: rank3 is the same body edited with a DIFFERENT garment")
    pairs = ([(ident, gi) for ident in ident_files for gi in range(len(a.garments))] if not a.diagonal
             else [(ident, i % len(a.garments)) for i, ident in enumerate(ident_files)])
    n, t_batch = 0, time.time()
    for ident, gi in pairs:
        gid = a.garments[gi]
        if True:
            if a.max_rows and n >= a.max_rows:
                break
            wrong = a.garments[(gi + 1) % len(a.garments)]
            target = a.out / f"{ident.stem}__g{gid}"
            if (target / "DONE").exists():
                print(f"= {target.name}: done, skipping"); n += 1; continue
            seed = a.seed + n * 10
            print(f"= {target.name}  (rank3 garment {wrong}, seed {seed})")
            t0 = time.time()
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ident, target / "identity.json")
            gdir = target / "garment"; gdir.mkdir(exist_ok=True)
            for p in list(GARMENT_INPUTS.glob(f"garment-{gid}*")) + list(GARMENT_MATTED.glob(f"garment-{gid}-*.alpha.png")):
                shutil.copy2(p, gdir / p.name)
            run("anny", HERE / "anny_body.py", "--identity", target / "identity.json", "--out", target,
                outputs=(target / "body.glb", target / "mask_front.usda", target / "phenotype.json"))
            ph = json.loads((target / "phenotype.json").read_text()); ph["hammersley_seed"] = seed
            (target / "phenotype.json").write_text(json.dumps(ph, indent=2))
            run("render", HERE / "render_hammersley.py", "--mesh", target / "body.usda", "--out", target / "render",
                "--seed", str(seed), "--views", "150",
                outputs=(target / "render" / "transforms.json", target / "render" / "149.png", target / "render" / "mesh.usda"))
            edit(target, "rank3", target / "body.glb", target / "render", wrong, seed + 100, recon=True)
            edit(target, "rank1", target / "body.glb", target / "render", gid, seed, recon=False)
            score_all(target, seed)
            run("render", HERE / "usd_convert.py", "--target", target)
            # Regeneration control (TRELLIS alone on rank1's front composite, nothing
            # preserved) and the 3D rest-exact gate; both negative controls sit inside.
            wholesale = target / "controls" / "wholesale" / "dressed_front.usda"
            run("default", HERE / "wholesale_control.py", "--image", target / "candidates" / "rank1" / "images_front" / "2d_edit.png",
                "--out", wholesale, "--seed", str(seed), extra_env=EDIT_ENV, outputs=(wholesale,))
            run("render", HERE / "voxel_controls.py", "--target", target)
            run("matting", HERE / "write_dress_on_rows.py", "--target", target, "--stage", stage)
            (target / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            n += 1
            print(f"  row {n} written in {time.time() - t0:.0f}s")
    print(f"batch: {n} rows in {time.time() - t_batch:.0f}s -> {stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
