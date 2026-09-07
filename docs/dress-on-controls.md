# Dress-on controls: what "rest exact" measures

Records the measurement that changed the rest-exact gate on 2026-09-07, and the
numbers the gate is read against. Tools: `tools/score_candidates.py` (2D, per
view), `tools/voxel_controls.py` (3D, per candidate), `tools/wholesale_control.py`
(the regeneration control), `tools/write_dress_on_rows.py` (asserts, then writes).

## Retracted: 2D depth outside the projected torso box

The first gate required every candidate's mean depth L1 on reference pixels
outside the projected torso boxes to stay within one voxel (1/64) of the rank5
floor on all 64 views. On the first batch row (female-p1, garment 0) it refused:

| candidate | max over views of depth_l1_outside |
|---|---|
| rank5 (source decode, floor) | 0.0227 |
| rank1 (own garment) | 0.1456 |
| rank3 (wrong garment) | 0.2902 |

The refusal was correct about the number and wrong about what the number
means. VoxHammer's structure stage (`voxhammer/edit_pipeline.py`,
`ply_to_ss_mask`) freezes a 16^3 latent cell only when all 64 of its
sub-voxels are preserved; every cell touching empty space is free. The
guarantee is therefore "preserved occupied voxels keep their latent", not
"empty space stays empty": a dress may grow wider than the torso box and hang
below it, and from a low camera that new geometry occludes the legs. The 2D
numbers stay in `dress_on_scores.parquet` per view; they no longer gate.

## Current gate: exact far containment against a per-row midpoint

At the pipeline's own 64^3 grid: source voxels are `render/voxels.usda`, the
delete set is the union of the front and back box sets, preserved = source
minus delete, far = preserved farther than 2 voxels from the delete set.
Each candidate mesh (`candidates/<rank>/candidate.usda`) is surface-voxelized
by subdividing until every edge is under half a voxel and keying the vertices.
Two numbers per set: exact-cell containment, and the same with a one-voxel
tolerance.

| target | floor (rank5) | rank1 | rank3 | regeneration control | shifted control |
|---|---|---|---|---|---|
| female-p1, g0 | 0.991 / 1.000 | 0.953 / 1.000 | 0.917 / 0.994 | 0.408 / 0.751 | 0.226 / 0.644 |
| female-p1, g1 | 0.991 / 1.000 | 0.961 / 1.000 | 0.919 / 1.000 | 0.401 / 0.698 | 0.226 / 0.644 |
| female-p1, g2 | 0.990 / 1.000 | 0.972 / 1.000 | 0.900 / 0.983 | 0.395 / 0.766 | 0.226 / 0.644 |
| female-p50, g0 | 0.993 / 1.000 | 0.898 / 0.998 | 0.931 / 0.993 | **0.726 / 1.000** | 0.227 / 0.623 |
| female-p50, g1 | 0.993 / 1.000 | 0.966 / 1.000 | 0.902 / 1.000 | 0.413 / 0.698 | 0.227 / 0.623 |
| male-p50, g2 | 0.992 / 1.000 | 0.962 / 1.000 | 0.968 / 1.000 | 0.320 / 0.693 | 0.228 / 0.633 |

Cells are exact / within one voxel; 1500 to 1758 far voxels per target.

**Retracted the same day:** a first version gated on the within-one-voxel
column with floor minus 0.02, because on the first row exact-cell losses next
to the box looked like surface jitter that dilation recovers. The regeneration
control on female-p50 with garment 0 then scored 1.000 on that column: a
regeneration of this body from its own render reproduces the far surface to
voxel precision, so that column cannot tell an edit from a regeneration. The
tool refused the row (exit 2) rather than pass it, which is what the control is
for. The exact column separates every edit (0.898 and up) from every
regeneration (0.726 and down) on all six rows.

The gate now: rank1 and rank3 exact far containment must be at least the
midpoint between the row's floor and its regeneration control; the
regeneration must sit at least 0.04 below the floor and the shifted source
below the midpoint, or the tool exits 2 and no row is written. The threshold
is recorded per row (`voxel_gate_threshold`). A planted value below the
threshold in `scores/voxel_controls.json` is refused by the writer.

New geometry outside the one-voxel dilation of source and delete sets is
reported per candidate (`voxel_new_outside`, 7.2 % of rank1's voxels here) and
not gated: it is what VoxHammer permits.

VoxHammer itself cannot serve as its own regeneration control: with the whole
body as the delete region the preserved set is empty and its SLat inversion
raises on a zero-element reshape. The control is the generator alone on the
same conditioning image (`tools/wholesale_control.py`, 45 s on the 4090).
That run is TRELLIS-image-large used as a generator, which the trellis2
blocklist scope names; here its output is a measurement instrument that
calibrates a gate, is stored under `controls/`, and never enters a corpus or a
row. If that reading is rejected, the shifted-source control alone still
bounds the gate (0.64 against a floor of 1.00), with less margin.
