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


def refuse_if_absolute(root: Path) -> None:
    hits = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() in (".png", ".jpg", ".ply", ".glb", ".parquet", ".usda", ".usdc"):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if any(m in text for m in ABS_MARKERS):
            hits.append(str(p.relative_to(root)))
    if hits:
        sys.exit(f"REFUSED: {len(hits)} staged file(s) name a local filesystem: {hits[:5]}")


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
    tok = subprocess.run([bao, "login", "-method=cert", "-no-store", "-field=token"], env=env,
                         capture_output=True, text=True, timeout=60).stdout.strip()
    env["BAO_TOKEN"] = tok
    hf = subprocess.run([bao, "kv", "get", "-field=hf_token", "agents/magi-16739d.agents.weftspun"], env=env,
                        capture_output=True, text=True, timeout=60).stdout.strip()
    if not hf.startswith("hf_"):
        sys.exit("FAIL: no hf_token in bao agents row")
    return hf


def readme(n_rows: int, n_cands: int, n_scores: int, n_judge: int) -> str:
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
{configs}---

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
    for name, tbl in (("dress_on", wide), ("dress_on_root", root), ("dress_on_candidates", cands),
                      ("dress_on_scores", scores)):
        p = out / "data" / name / "train-00000-of-00001.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(tbl, preserve_index=False), p, compression="zstd", compression_level=10)
    if judge is not None:
        p = out / "data" / "dress_on_judge" / "train-00000-of-00001.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(judge, preserve_index=False), p, compression="zstd", compression_level=10)
    shutil.copytree(a.stage / "assets", out / "assets")
    (out / "README.md").write_text(readme(len(root), len(cands), len(scores), 0 if judge is None else len(judge)), encoding="utf-8")
    refuse_if_absolute(out)
    refuse_if_forbidden(out)
    n_assets = sum(1 for _ in (out / "assets").rglob("*") if _.is_file())
    print(f"[publish] staged {len(root)} rows, {len(cands)} candidates, {len(scores)} score rows, {n_assets} assets -> {out}")

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
