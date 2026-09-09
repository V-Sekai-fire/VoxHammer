"""Publish the dress-on stage: joined HF view + satellites + assets, locally or to the Hub.

Runs in the `matting` pixi environment:

    pixi run -e matting python tools/publish_dress_on.py --stage stage/ --local ~/Desktop/hf
    pixi run -e matting python tools/publish_dress_on.py --stage stage/ --hub chibifire/anny-dress-on-stage-train

--local writes the exact folder that would be uploaded (operator: "pretend DESKTOP is
hf for now"). --hub uploads with HfApi.upload_folder (resumable, incremental: no
delete_patterns) using the token stored in bao at agents/<CN> field hf_token.

The joined view follows anny-render-corpus/maskscore_rung_1_hf_publish.py: the root row
with candidates nested, each candidate with its scores and judge rows nested. An
unjudged candidate carries judge = [] -- a value, not a null. README has no
`configs:` block; the auto-parquet indexer picks up data/<config>/.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

FORBIDDEN = ("val2017", "coco-ood", "rf-detr-keypoint-data", "nf4", "int4", "int8")
ABS_MARKERS = ("C:\\", "C:/", "/c/Users", "/Users/")


def build_wide(root: pd.DataFrame, cands: pd.DataFrame, scores: pd.DataFrame, judge: pd.DataFrame | None):
    def per_view(df):
        return df.drop(columns=["row_key", "candidate"]).to_dict("records")
    rows = []
    for _, r in root.iterrows():
        rec = r.to_dict()
        cl = []
        for _, c in cands[cands.row_key == r.key].sort_values("rank").iterrows():
            crec = c.drop(labels=["row_key"]).to_dict()
            crec["scores"] = per_view(scores[(scores.row_key == r.key) & (scores.candidate == c.candidate)])
            crec["judge"] = (per_view(judge[(judge.row_key == r.key) & (judge.candidate == c.candidate)])
                             if judge is not None else [])
            cl.append(crec)
        rec["candidates"] = cl
        rows.append(rec)
    return pd.DataFrame(rows)


def scan_parquet_for_absolute(path: Path) -> list[str]:
    """Payload lives in the parquet now, so the scan reads into it rather than past it.

    String columns are read directly; embedded OpenUSD is text and can carry an asset
    reference, so those blobs are decoded too. Images and point clouds are opaque and
    a byte match inside them would be a coincidence, not a leak.
    """
    hits, tb = [], pq.read_table(path)
    for name, ty in zip(tb.column_names, tb.schema.types):
        if pa.types.is_string(ty) or pa.types.is_large_string(ty):
            for v in tb.column(name).to_pylist():
                if v and any(m in v for m in ABS_MARKERS):
                    hits.append(f"{path.name}:{name}={v[:60]}")
                    break
        elif pa.types.is_binary(ty) or pa.types.is_large_binary(ty):
            paths = tb.column(f"{name}_path").to_pylist() if f"{name}_path" in tb.column_names else None
            for i, blob in enumerate(tb.column(name).to_pylist()):
                rel = paths[i] if paths else ""
                if not blob or not rel.endswith(".usda"):
                    continue
                if any(m in blob.decode("utf-8", "ignore") for m in ABS_MARKERS):
                    hits.append(f"{path.name}:{name} embedded USD names a local filesystem ({rel})")
                    break
    return hits


def refuse_if_absolute(root: Path) -> None:
    hits = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".parquet":
            hits += scan_parquet_for_absolute(p)
            continue
        if p.suffix.lower() in (".png", ".jpg", ".ply", ".glb", ".usdc"):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if any(m in text for m in ABS_MARKERS):
            hits.append(str(p.relative_to(root)))
    if hits:
        sys.exit(f"REFUSED: {len(hits)} staged reference(s) name a local filesystem: {hits[:5]}")


def refuse_if_forbidden(root: Path) -> None:
    bad = [str(p) for p in root.rglob("*") if any(m in str(p).lower() for m in FORBIDDEN)]
    if bad:
        sys.exit(f"REFUSED: blocked marker in staged path: {bad[:5]}")


def hf_token_from_bao() -> str:
    env = dict(os.environ, BAO_ADDR="https://100.124.200.34:8200", BAO_TLS_SERVER_NAME="weftspun-bao.internal",
               BAO_CACERT=os.path.expanduser("~/.magi/ca-bundle.pem"),
               BAO_CLIENT_CERT=os.path.expanduser("~/.magi/v3-leaf-int.pem"),
               BAO_CLIENT_KEY=os.path.expanduser("~/.magi/bao-client-v3.key"))
    bao = os.path.expanduser("~/bin/bao.exe")
    # the login prints a note about not storing the token above the token itself
    tok = subprocess.run([bao, "login", "-method=cert", "-no-store", "-field=token"], env=env,
                         capture_output=True, text=True, timeout=60).stdout.strip().splitlines()[-1].strip()
    env["BAO_TOKEN"] = tok
    hf = subprocess.run([bao, "kv", "get", "-field=hf_token", "agents/magi-16739d.agents.weftspun"], env=env,
                        capture_output=True, text=True, timeout=60).stdout.strip()
    if not hf.startswith("hf_"):
        sys.exit("FAIL: no hf_token in bao agents row")
    return hf


# The viewer renders a column as a picture only when the card declares it, so the
# declaration is generated from the schema rather than typed out: a column added to
# the writer cannot then go undeclared. Images are struct<bytes, path>, which is the
# shape `datasets` decodes.
IMAGE_COLS = ("garment_front", "garment_back", "garment_brand", "matted_front", "matted_back",
              "2d_render_front", "2d_edit_front", "2d_mask_front",
              "2d_render_back", "2d_edit_back", "2d_mask_back")
PA_TO_HF = {"string": "string", "large_string": "string", "bool": "bool",
            "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
            "uint8": "uint8", "uint16": "uint16", "uint32": "uint32", "uint64": "uint64",
            "float": "float32", "double": "float64", "halffloat": "float16",
            "binary": "binary", "large_binary": "binary"}


def hf_feature(field) -> dict:
    t = field.type
    if field.name in IMAGE_COLS and pa.types.is_struct(t):
        return {"name": field.name, "dtype": "image"}
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        vt = t.value_type
        if pa.types.is_struct(vt):
            return {"name": field.name, "list": [hf_feature(vt.field(i)) for i in range(vt.num_fields)]}
        return {"name": field.name, "sequence": PA_TO_HF[str(vt)]}
    if pa.types.is_struct(t):
        return {"name": field.name, "struct": [hf_feature(t.field(i)) for i in range(t.num_fields)]}
    key = str(t)
    if key not in PA_TO_HF:
        sys.exit(f"REFUSED: no HF feature mapping for column {field.name} of type {key}")
    return {"name": field.name, "dtype": PA_TO_HF[key]}


def render_features(feats: list[dict], indent: int) -> str:
    pad, out = " " * indent, []
    for f in feats:
        out.append(f"{pad}- name: {f['name']}\n")
        for key in ("dtype", "sequence"):
            if key in f:
                out.append(f"{pad}  {key}: {f[key]}\n")
        for key in ("list", "struct"):
            if key in f:
                out.append(f"{pad}  {key}:\n" + render_features(f[key], indent + 2))
    return "".join(out)


def dataset_info(schemas: dict) -> str:
    out = ["dataset_info:\n"]
    for name, sch in schemas.items():
        out.append(f"- config_name: {name}\n  features:\n")
        out.append(render_features([hf_feature(sch.field(i)) for i in range(len(sch))], 2))
        out.append("  splits:\n  - name: train\n")
    return "".join(out)


def readme(n_rows: int, n_cands: int, n_scores: int, n_judge: int, schemas: dict) -> str:
    # Without this block the viewer folds every data/<name>/ directory into one
    # "default" config (measured 2026-09-07: 600 rows, 48 union features). Each table
    # is its own config; the joined view is the default one.
    names = ["dress_on", "dress_on_root", "dress_on_candidates", "dress_on_scores"] + (["dress_on_judge"] if n_judge else [])
    configs = "".join(
        f"- config_name: {n}\n" + ("  default: true\n" if n == "dress_on" else "")
        + f"  data_files:\n  - split: train\n    path: data/{n}/*.parquet\n" for n in names)
    return f"""---
license: cc-by-4.0
task_categories:
- image-to-3d
tags:
- dress-on
- anny
- voxhammer
- maskscore
configs:
{configs}{dataset_info(schemas)}---

# anny-dress-on-stage-train

Dress-on edits of ANNY parametric bodies with second-hand garment photos, edited by
VoxHammer (training-free 3D latent editing on TRELLIS-image-large), rendered by
Mitsuba 3 over a seeded Hammersley camera sequence, and scored with the MaskScore
geometric metric against the source's own decode.

MaskScore-shaped ETNF: `dress_on` (root), `dress_on_candidates` (rank1 own garment,
rank3 wrong garment, rank5 source decode = the floor), `dress_on_scores` (per view,
outside/inside the torso masks), `dress_on_judge` (only when a judge ran). The
`dress_on` config under `data/dress_on/` is the joined view for browsing.

Provenance per column: body `constructed:anny:<sha>`, garment
`real:zenodo-second-hand-fashion-v3:<id>`, dressed mesh
`generated:voxhammer:trellis-image-large:seed<n>`. Identities are the appendix-E
anthropometry rows (stature, mass) solved onto ANNY phenotypes; every unmapped
measurement is listed per row. Pass-1 scope is silhouette + palette; the brand
close-up is stored and never used for conditioning.

Rest-exact is measured in 3D (`rest_exact_basis`): per candidate,
`voxel_far_containment` is the fraction of preserved source voxels farther than
two voxels from the edit region that the candidate's surface still occupies at the
64^3 grid, read against rank5 (`floor_voxel_containment`). A row was written only
when rank1 and rank3 sat above `voxel_gate_threshold`, the midpoint between the
floor and a TRELLIS regeneration of the same composite
(`voxel_control_regeneration_containment`), and both negative controls (that
regeneration; the source shifted three voxels) fell below it. `voxel_far_within_1`
is the same fraction with a one-voxel tolerance, reported only. VoxHammer frees any structure cell not wholly preserved, so geometry
outside the box is permitted and reported as `voxel_new_outside`. The per-view 2D
depth/normal numbers outside and inside the projected torso boxes are reported,
not gated.

## Corrections, 2026-09-08

Two defects were fixed and every row re-staged. A copy pulled before this date
carries both.

**Three candidates shared one mesh.** Assets were staged under a per-row
directory keyed on the source file name, so `rank1`, `rank3` and `rank5` all
wrote `candidate.usda` and `candidate_gaussian.ply` to one destination and the
later copies were silently dropped; `2d_render_front` and `2d_render_back`
collapsed onto one file the same way. Measured before: 12 distinct meshes for
36 candidate rows, and front equal to back on every row. After: 36 of 36
distinct by content, front differing from back on all 24 that carry both. Two
different sources reaching one destination is now refused rather than aliased.

**The edited candidates declared the wrong scale.** `rank1` and `rank3` carried
`metersPerUnit = 1` while the body, `rank5` and the masks carried the subject's
stature, so a consumer honouring USD units read the two edited candidates
between 30 and 49 per cent too small. Geometry was never affected: all meshes
are unit-height and only the declared scale was wrong. Every mesh now declares
its subject's stature and agrees with the row's `achieved_stature_m`.

**Assets now live in the parquet.** Images are `struct<bytes, path>` and are
declared in `dataset_info`, so the viewer renders them; meshes, masks and point
clouds are bytes with the staged path kept beside them as provenance. There is
no `assets/` tree: a payload next to a parquet is one ignore rule away from
being dropped.

Rows: {n_rows} · candidates: {n_cands} · score rows: {n_scores} · judge rows: {n_judge}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--local", type=Path, help="write the publishable folder here (pretend it is HF)")
    ap.add_argument("--hub", help="repo id to upload to, e.g. chibifire/anny-dress-on-stage-train")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    if not (a.local or a.hub):
        sys.exit("give --local <dir> and/or --hub <repo>")

    d = a.stage / "data"
    root = pq.read_table(d / "dress_on.parquet").to_pandas()
    cands = pq.read_table(d / "dress_on_candidates.parquet").to_pandas()
    scores = pq.read_table(d / "dress_on_scores.parquet").to_pandas()
    judge = pq.read_table(d / "dress_on_judge.parquet").to_pandas() if (d / "dress_on_judge.parquet").exists() else None
    wide = build_wide(root, cands, scores, judge)

    out = a.stage / "publish"
    if out.exists():
        shutil.rmtree(out)
    schemas = {}
    tables = [("dress_on", wide), ("dress_on_root", root), ("dress_on_candidates", cands),
              ("dress_on_scores", scores)] + ([("dress_on_judge", judge)] if judge is not None else [])
    for name, tbl in tables:
        p = out / "data" / name / "train-00000-of-00001.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(tbl, preserve_index=False)
        schemas[name] = table.schema
        pq.write_table(table, p, compression="zstd", compression_level=3, row_group_size=4)
    # Payload rides in the parquet (operator 2026-08-08); an assets/ tree beside it is
    # the shape that lost the FBD corpora their diagrams, so nothing is copied here.
    (out / "README.md").write_text(
        readme(len(root), len(cands), len(scores), 0 if judge is None else len(judge), schemas), encoding="utf-8")
    refuse_if_absolute(out)
    refuse_if_forbidden(out)
    mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"[publish] staged {len(root)} rows, {len(cands)} candidates, {len(scores)} score rows, {mb:.0f} MB -> {out}")

    if a.local:
        dst = a.local / "chibifire" / "anny-dress-on-stage-train"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(out, dst)
        print(f"[publish] local (pretend-HF) copy -> {dst}")
    if a.hub:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token_from_bao())
        api.create_repo(a.hub, repo_type="dataset", private=a.private, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=a.hub, repo_type="dataset",
                          commit_message=f"dress-on stage: {len(root)} rows")
        files = api.list_repo_files(a.hub, repo_type="dataset")
        print(f"[publish] hub {a.hub}: {len(files)} files, data/dress_on present: {'data/dress_on/train-00000-of-00001.parquet' in files}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
