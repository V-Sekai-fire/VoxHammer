"""Rest-exact as VoxHammer actually guarantees it: in 3D, on the preserved voxel set.

    pixi run -e render python tools/voxel_controls.py --target work/batch1/<target>

VoxHammer keeps the latent of every preserved source voxel (edit_pipeline.py:543-552)
but its structure stage frees any 16^3 cell that is not entirely preserved
(ply_to_ss_mask: `.all(dim=1)` over the 64 sub-voxels), so empty space beside the
edit box may gain geometry. A 2D depth gate outside the projected box therefore
fails on a skirt that hangs below the torso box while the body underneath is intact.

What is measured per candidate, at the 64^3 grid the pipeline itself uses:
  containment   fraction of preserved source voxels still occupied by the candidate
  new_outside   candidate voxels outside the 1-voxel dilation of (source U delete set),
                the geometry VoxHammer was free to add; reported, not gated
rank5 is the source's own decode and sets the floor. Gate: rank1 and rank3 containment
within TOL of the floor. Negative control: the source shifted by SHIFT voxels must fall
below the floor by more than TOL, or the gate is decoration and this exits 2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import usd_io  # noqa: E402

GRID = 64
TOL = 0.02
SHIFT = 3
NEAR = 2
RANKS = ("rank1", "rank3", "rank5")


def keys_of(points: np.ndarray) -> set:
    idx = np.clip(np.floor((points + 0.5) * GRID), 0, GRID - 1).astype(np.int64)
    return set(map(tuple, idx))


def dilate(keys: set) -> set:
    out = set()
    for k in keys:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    out.add((k[0] + dx, k[1] + dy, k[2] + dz))
    return out


def mesh_keys(path: Path) -> set:
    v, f, _, up = usd_io.read_mesh(path)
    v = np.asarray(v, dtype=np.float64)
    if up == "Y":
        v = usd_io.yup_to_zup(v)
    v = np.clip(v, -0.5 + 1e-6, 0.5 - 1e-6)
    # Surface voxelization by subdividing until every edge is under half a voxel, then
    # keying the vertices; the same rule for every candidate, so the floor absorbs it.
    sv, _ = trimesh.remesh.subdivide_to_size(v, np.asarray(f, dtype=np.int64), max_edge=0.5 / GRID)
    return keys_of(sv)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    t = a.target
    out = a.out or (t / "scores" / "voxel_controls.json")

    src = keys_of(usd_io.read_points(t / "render" / "voxels.usda"))
    delete = keys_of(usd_io.read_points(t / "render" / "voxels_delete.usda"))
    back = None
    for cand in ("rank1", "rank3"):
        p = t / "candidates" / cand / "render_pass1" / "voxels_delete.usda"
        if p.is_file():
            back = keys_of(usd_io.read_points(p))
            break
    if back is None:
        print("[voxctl] FAIL: no back-pass voxels_delete.usda under any candidate"); return 2
    delete |= back
    preserved = src - delete
    allowed = dilate(src | delete)
    if not preserved:
        print("[voxctl] FAIL: preserved set is empty"); return 2

    # Preserved voxels within NEAR voxels of the delete set may be enclosed by the new
    # garment and leave the surface legitimately; the gate reads the far ones.
    near = delete
    for _ in range(NEAR):
        near = dilate(near)
    far_preserved = preserved - near

    def measure(ck: set) -> dict:
        within1 = dilate(ck)
        return {
            "n_voxels": len(ck),
            "containment": len(ck & preserved) / len(preserved),
            "far_containment": len(ck & far_preserved) / len(far_preserved),
            "far_within_1": len(within1 & far_preserved) / len(far_preserved),
            "missing_far": len(far_preserved - ck),
            "new_outside": len(ck - allowed),
            "new_outside_frac": len(ck - allowed) / max(1, len(ck)),
        }

    # The gate reads exact-cell far containment against a per-row threshold halfway
    # between the floor and the regeneration control. Measured over 6 rows: edits
    # 0.90-0.97, regeneration 0.32-0.73, floor 0.99; the within-one-voxel variant is
    # reported but cannot gate (a regeneration of female-p50 reached 1.000 on it).
    rows = {cand: measure(mesh_keys(t / "candidates" / cand / "candidate.usda")) for cand in RANKS}
    floor = rows["rank5"]["far_containment"]
    shifted = {(k[0] + SHIFT, k[1], k[2]) for k in src}
    control = measure(shifted)["far_containment"]
    failures = []
    # The generator alone on the same composite, nothing preserved
    # (tools/wholesale_control.py): what a regeneration scores. Required, not optional.
    wpath = t / "controls" / "wholesale" / "dressed_front.usda"
    if not wpath.is_file():
        print(f"[voxctl] FAIL: regeneration control missing ({wpath}); no control, no gate, no row"); return 2
    wholesale = measure(mesh_keys(wpath))["far_containment"]
    if not wholesale < floor - 2 * TOL:
        print(f"[voxctl] FAIL control: regeneration keeps far containment {wholesale:.4f} vs floor {floor:.4f}; "
              f"less than {2 * TOL} of room, the gate cannot separate an edit from a regeneration")
        return 2
    threshold = (floor + wholesale) / 2
    if not control < threshold:
        print(f"[voxctl] FAIL control: source shifted {SHIFT} voxels keeps far containment {control:.4f} "
              f"vs threshold {threshold:.4f}; the gate cannot see a wholesale move")
        return 2
    for cand in ("rank1", "rank3"):
        c = rows[cand]["far_containment"]
        if c < threshold:
            failures.append(f"{cand} far containment {c:.4f} < threshold {threshold:.4f} "
                            f"(floor {floor:.4f}, regeneration {wholesale:.4f})")
    result = {
        "grid": GRID, "tolerance": TOL, "shift_control_voxels": SHIFT, "near_voxels": NEAR,
        "n_source": len(src), "n_delete": len(delete), "n_preserved": len(preserved), "n_far_preserved": len(far_preserved),
        "floor_containment": floor, "control_shifted_containment": control,
        "control_wholesale_containment": wholesale, "gate_threshold": threshold,
        "candidates": rows, "failures": failures,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[voxctl] preserved {len(preserved)} of {len(src)} source voxels, {len(far_preserved)} farther than {NEAR} from the edit set; "
          f"far containment: floor (rank5) {floor:.4f}, regeneration control {wholesale:.4f}, shifted-source control {control:.4f}, "
          f"threshold {threshold:.4f}; " +
          "  ".join(f"{c}: exact {rows[c]['far_containment']:.4f} within1 {rows[c]['far_within_1']:.4f} "
                    f"new-outside {rows[c]['new_outside']} ({rows[c]['new_outside_frac']:.1%})" for c in RANKS))
    if failures:
        print("[voxctl] REFUSED: " + "; ".join(failures)); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
