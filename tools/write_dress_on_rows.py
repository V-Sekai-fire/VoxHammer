"""Emit one dress-on target as MaskScore-shaped ETNF parquets, controls first.

Runs in the `matting` pixi environment (pyarrow):

    pixi run -e matting python tools/write_dress_on_rows.py --target work/t000 --stage stage/

Reads a target directory laid out by dress_on_batch.py:

    identity.json  phenotype.json  body.glb  mask_front.glb  mask_back.glb
    garment/garment-<id>-{front,back,brand}.jpg + .alpha.png + garment-<id>.json
    candidates/<rank>/candidate.glb, candidate_gaussian.ply, timing.json,
                      images_front/{2d_render,2d_edit,2d_mask,view,composite}, images_back/...
    scores/<rank>.json          (tools/score_candidates.py, always present)
    judge/<rank>.json           (optional: EditScore record)

and appends to four parquets under <stage>/data/ following
anny-render-corpus/maskscore_rung_1_mesh.py: root, candidates, scores (geometric,
per view), judge (per candidate, only when a judge ran). Assets are copied under
<stage>/assets/<key>/ and referenced by RELATIVE path; no nullable column anywhere.

Controls are asserted before anything is written; a failure refuses the whole
target (rule 2: a gate that cannot fail is decoration).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

STUB = ("dress_on", "garment_edit", "instruction_following", "input_mesh", "anny_glb")
RANKS = {"rank1": 1, "rank3": 3, "rank5": 5}
VOXEL = 1.0 / 64
ABSOLUTE_RE = re.compile(r"(?:[A-Za-z]:[\\/]|/c/Users/|/Users/)", re.IGNORECASE)


def die(msg: str) -> None:
    raise SystemExit(f"REFUSED: {msg}")


def load(p: Path):
    return json.loads(p.read_text())


def assert_controls(scores: dict[str, dict], vox: dict) -> dict:
    """rank5 is the source's own decode (the floor); rank1/rank3 are read against it.

    Rest-exact is measured in 3D on the preserved voxel set (tools/voxel_controls.py):
    VoxHammer frees any structure cell that is not wholly preserved, so geometry may
    legitimately appear outside the box in 2D; the 2D outside-box numbers are reported
    per view, not gated. The voxel tool has already refused when its own controls fail;
    this re-checks its verdict so a stale or hand-edited file cannot pass.
    """
    s5 = scores["rank5"]["summary"]
    out = {"floor_depth_l1_outside": s5["mean_depth_l1_outside"], "floor_depth_l1_inside": s5["mean_depth_l1_inside"],
           "floor_voxel_containment": float(vox["floor_containment"]),
           "voxel_control_shifted": float(vox["control_shifted_containment"]),
           "voxel_control_regeneration": float(vox["control_wholesale_containment"]),
           "voxel_gate_threshold": float(vox["gate_threshold"])}
    if vox.get("failures"):
        die("rest-exact (voxel): " + "; ".join(vox["failures"]))
    thr, floor = vox["gate_threshold"], vox["floor_containment"]
    if not (vox["control_wholesale_containment"] < thr and vox["control_shifted_containment"] < thr < floor):
        die("rest-exact (voxel): a negative control does not fall below the gate threshold; gate is decoration")
    for r in ("rank1", "rank3"):
        c = vox["candidates"][r]["far_containment"]
        if c < thr:
            die(f"rest-exact (voxel): {r} far containment {c:.4f} < threshold {thr:.4f} (floor {floor:.4f})")
    if not scores["rank1"]["summary"]["mean_depth_l1_inside"] > s5["mean_depth_l1_inside"]:
        die("negative (geometry): rank1 inside-mask depth_l1 is not strictly above the floor -- nothing changed")
    n = {r: scores[r]["summary"]["n_views"] for r in RANKS}
    if len(set(n.values())) != 1:
        die(f"ragged scores satellite: view counts differ {n}")
    return out


def assert_judge(judge: dict[str, dict]) -> str:
    """The judge's own negative control. A failure invalidates the judge on this
    target, not the row: its rows are omitted and the verdict is recorded, so
    "not judged" and "judge could not order the candidates" stay distinguishable."""
    if not judge:
        return "absent"
    if set(judge) != set(RANKS):
        die(f"judge rows must cover all candidates or none; got {sorted(judge)}")
    if any(judge[r].get("refused") for r in RANKS):
        return "refused"  # a refusal is recorded, not ranked
    o = {r: judge[r]["overall"] for r in RANKS}
    if not (o["rank1"] > o["rank3"] > o["rank5"]):
        print(f"JUDGE REJECTED on this target: expected rank1 > rank3 > rank5, got {o}; judge rows omitted")
        return "failed-order"
    return "passed"


# Payload lives in the parquet, not beside it (operator 2026-09-08): an asset next to
# a parquet is one ignore rule away from being dropped, which is how the FBD corpora lost
# their diagrams. Images take the viewer's {bytes, path} shape so a reader can look at
# them; meshes and point clouds are bytes with the staged path kept alongside as provenance.
IMAGE_COLS = ("garment_front", "garment_back", "garment_brand", "matted_front", "matted_back",
              "2d_render_front", "2d_edit_front", "2d_mask_front",
              "2d_render_back", "2d_edit_back", "2d_mask_back")
BLOB_COLS = ("input_asset", "mask_front", "mask_back", "candidate_asset", "candidate_gaussian")


def embed(rec: dict, stage: Path) -> dict:
    """Replace staged references with their bytes; an absent asset is empty, never null."""
    out: dict = {}
    for k, v in rec.items():
        if k in IMAGE_COLS:
            out[k] = {"bytes": (stage / v).read_bytes() if v else b"", "path": v}
        elif k in BLOB_COLS:
            out[k] = (stage / v).read_bytes() if v else b""
            out[f"{k}_path"] = v
        else:
            out[k] = v
    return out


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_asset(src: Path, stage: Path, key_dir: str, sub: str = "") -> str:
    # Canonical mesh assets are OpenUSD (CLAUDE.md archive rule; operator 2026-09-07).
    # usd_convert.py writes a .usda beside every .glb; stage that one when it exists.
    if src.suffix.lower() == ".glb" and src.with_suffix(".usda").is_file():
        src = src.with_suffix(".usda")
    if not src.is_file():
        die(f"missing asset {src}")
    dst = stage / "assets" / key_dir / sub / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # Two sources landing on one destination is what made three candidates share
        # one mesh; refuse it rather than let the second copy be dropped in silence.
        if sha256_file(dst) != sha256_file(src):
            die(f"two different sources stage onto {dst.relative_to(stage).as_posix()}: {src}")
    else:
        shutil.copy2(src, dst)
    rel = dst.relative_to(stage).as_posix()
    if ABSOLUTE_RE.search(rel):
        die(f"absolute path leaked into staged reference: {rel}")
    return rel


def append(table: pa.Table, path: Path, level: int = 10) -> None:
    if path.exists():
        old = pq.read_table(path)
        table = pa.concat_tables([old, table.cast(old.schema)])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", compression_level=level, row_group_size=4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--stage", type=Path, required=True)
    a = ap.parse_args()
    t, stage = a.target, a.stage

    ident, ph = load(t / "identity.json"), load(t / "phenotype.json")
    gmeta = next(t.glob("garment/garment-*.json"))
    garment = load(gmeta)
    gid = int(garment["garment_id"])
    ikey = ident["identity_source"].replace("appendix-e:", "").replace(":", "-")
    key = f"dress_on/{ikey}/{gid}"
    key_dir = key.replace("/", "_")

    scores = {r: load(t / "scores" / f"{r}.json") for r in RANKS}
    if not (t / "scores" / "voxel_controls.json").is_file():
        die("scores/voxel_controls.json missing: run tools/voxel_controls.py first (no gate, no row)")
    vox = load(t / "scores" / "voxel_controls.json")
    judge = {r: load(t / "judge" / f"{r}.json") for r in RANKS if (t / "judge" / f"{r}.json").exists()}
    floor = assert_controls(scores, vox)
    judge_control = assert_judge(judge)
    if judge_control == "failed-order":
        judge = {}

    for r in ("rank1", "rank3"):
        for side in ("front", "back"):
            comp = load(t / "candidates" / r / f"images_{side}" / "composite.json")
            if "brand" in comp["garment"]:
                die(f"{r}/{side} conditioned on the brand close-up (pass-1 no-logos rule)")

    S = lambda p, sub="": stage_asset(p, stage, key_dir, sub)  # noqa: E731
    g = t / "garment"
    root = {
        "key": key, "task_type": STUB[1], "dimension": STUB[2], "input_column": STUB[3],
        "input_asset": S(t / "body.glb"), "input_asset_kind": STUB[4],
        "identity_source": ident["identity_source"], "anny_sha": ident["anny_sha"],
        "target_stature_m": float(ident["target_stature_m"]), "achieved_stature_m": float(ident["achieved_stature_m"]),
        "target_mass_kg": float(ident["target_mass_kg"]), "achieved_mass_kg": float(ident["achieved_mass_kg"]),
        "mass_fitted": bool(ident["mass_fitted"]), "unmapped_measurements": list(ident["unmapped_measurements"]),
        **{f"pheno_{k}": float(v) for k, v in ident["phenotype"].items()},
        "garment_id": gid, "garment_source": "chibifire/zenodo-second-hand-fashion-v3",
        "garment_category": str(garment.get("category", "")), "garment_type": str(garment.get("type", "")),
        "garment_front": S(g / f"garment-{gid}-front.jpg"), "garment_back": S(g / f"garment-{gid}-back.jpg"),
        "garment_brand": S(g / f"garment-{gid}-brand.jpg"),
        "matted_front": S(g / f"garment-{gid}-front.alpha.png"), "matted_back": S(g / f"garment-{gid}-back.alpha.png"),
        "mask_front": S(t / "mask_front.usda"), "mask_back": S(t / "mask_back.usda"),
        "hammersley_seed": int(ph.get("hammersley_seed", 1)),
        "floor_depth_l1_outside": float(floor["floor_depth_l1_outside"]),
        "floor_depth_l1_inside": float(floor["floor_depth_l1_inside"]),
        "rest_exact_basis": "voxel-containment-of-preserved-set-vs-decode-floor",
        "judge_control": judge_control,
        "floor_voxel_containment": float(floor["floor_voxel_containment"]),
        "voxel_control_shifted_containment": float(floor["voxel_control_shifted"]),
        "voxel_control_regeneration_containment": float(floor["voxel_control_regeneration"]),
        "voxel_gate_threshold": float(floor["voxel_gate_threshold"]),
        "body_provenance": f"constructed:anny:{ident['anny_sha']}",
        "garment_provenance": f"real:zenodo-second-hand-fashion-v3:{gid}",
        "renderer": "mitsuba3-seeded-hammersley",
    }

    cands, score_rows, judge_rows = [], [], []
    for r, rank in RANKS.items():
        cdir = t / "candidates" / r
        timing = load(cdir / "timing.json") if (cdir / "timing.json").exists() else {}
        rec = {
            "row_key": key, "candidate": r, "rank": rank,
            "candidate_asset": S(cdir / "candidate.glb", r),
            "candidate_gaussian": S(cdir / "candidate_gaussian.ply", r) if (cdir / "candidate_gaussian.ply").exists() else S(cdir / "candidate.glb", r),
            "candidate_provenance": ("constructed-decode:voxhammer-source-recon" if r == "rank5"
                                     else f"generated:voxhammer:trellis-image-large:seed{timing.get('seed', -1)}"),
            "garment_id_used": int(load(cdir / "garment_used.json")["garment_id"]) if (cdir / "garment_used.json").exists() else -1,
            "wall_edit_s": float(timing.get("wall_total_s", 0.0)),
            "peak_vram_mib": int(timing.get("peak_vram_mib", 0)),
            "face_count": int(timing.get("output_face_count", 0)),
            "voxel_containment": float(vox["candidates"][r]["containment"]),
            "voxel_far_containment": float(vox["candidates"][r]["far_containment"]),
            "voxel_far_within_1": float(vox["candidates"][r]["far_within_1"]),
            "voxel_new_outside": int(vox["candidates"][r]["new_outside"]),
            "voxel_new_outside_frac": float(vox["candidates"][r]["new_outside_frac"]),
        }
        for side in ("front", "back"):
            idir = cdir / f"images_{side}"
            for name in ("2d_render", "2d_edit", "2d_mask"):
                p = idir / f"{name}.png"
                rec[f"{name}_{side}"] = S(p, f"{r}/{side}") if p.exists() else ""
        cands.append(rec)
        for v in scores[r]["views"]:
            score_rows.append({"row_key": key, "candidate": r, **{k: (float(x) if isinstance(x, float) else int(x)) for k, x in v.items()}})
        if r in judge:
            j = judge[r]
            judge_rows.append({"row_key": key, "candidate": r,
                               "judge_base": str(j["base"]), "judge_adapter": str(j["adapter"]),
                               "judge_precision": str(j["precision"]), "judge_num_pass": int(j["num_pass"]),
                               "prompt_sha": str(j["prompt_sha"]), "instruction": str(j["instruction"]),
                               "overall": float(j["overall"] if j["overall"] is not None else -1.0),
                               "refused": bool(j["refused"])})

    data = stage / "data"
    # Payload tables carry already-compressed images, so level 3 buys the same size for
    # a fraction of the time; the two small tables keep 10.
    append(pa.Table.from_pylist([embed(root, stage)]), data / "dress_on.parquet", level=3)
    append(pa.Table.from_pylist([embed(c, stage) for c in cands]), data / "dress_on_candidates.parquet", level=3)
    append(pa.Table.from_pylist(score_rows), data / "dress_on_scores.parquet")
    if judge_rows:
        append(pa.Table.from_pylist(judge_rows), data / "dress_on_judge.parquet")

    for name in ("dress_on", "dress_on_candidates", "dress_on_scores", "dress_on_judge"):
        p = data / f"{name}.parquet"
        if p.exists():
            tb = pq.read_table(p)
            nulls = {c: tb.column(c).null_count for c in tb.column_names if tb.column(c).null_count}
            if nulls:
                die(f"{name}.parquet has nulls: {nulls}")
    print(f"ok {key}: root 1 row, candidates {len(cands)}, scores {len(score_rows)}, judge {len(judge_rows)} -> {data}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
