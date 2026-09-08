# `chibifire/starforged-std-3001-appendix-e` — schema as read 2026-09-07

Config `human`, split `train`, 120 rows. Columns:
`universe, section, category, subcategory, metric, value, percentile, sex, population, source, notes`.
Long form: one (section, subcategory, metric, percentile, sex) per row; `value` is a float.
`population` is `worldwide_civilian_2024` on every row.

| section | category | rows | percentile | sex | what it carries |
|---|---|---|---|---|---|
| E.2 | anthropometry | 36 | p1 / p50 / p99 | female, male | 9 dimensions in cm (IPD in mm): stature, sitting_height, eye_height_seated, thigh_clearance, shoulder_breadth, hip_breadth_seated, hand_length, head_circumference, interpupillary_distance |
| E.3 | range_of_motion | 16 | — | any | 8 joints × angle_deg_min/max (AAOS Norkin-White) |
| E.4 | body_surface_area | 2 | p1 female, p99 male | | area_m2 (Du Bois) |
| E.5 | body_mass | 10 | p1 female, p99 male | | whole_body mass_kg (2 rows) + 8 segment fractions_of_whole |
| E.6 | body_volume | 2 | p1 female, p99 male | | volume_L = mass / 1.0 g/cm³ (derived from E.5) |
| E.7 | strength | 8 | min | any | force_N / torque per action |
| E.8 | species | 2 | — | any | homo_sapiens fraction 1.0 |
| E.9 | sex | 6 | — | any | male/female/intersex fractions + counts |
| E.10 | age | 12 | — | any | 6 age bands, fraction + count |
| E.11 | region | 12 | — | any | 6 regions, fraction + count |
| E.12 | handedness | 4 | — | any | right/left fraction + count |
| E.13 | body_model_identity | 10 | — | any | the workspace's own spec: SOMA 77 joints / 78 params, silhouette_iou_target 0.92, corrective_blendshape_L1 budget 0.15, hammersley_views 512, twist_bones 8, face_facs_units 52, motionbricks_styles 15, voice_customvoice_slots 9 |

## The percentile coverage is asymmetric — this bounds the identity grid

E.2 stature rows, verbatim:

| sex | p1 | p50 | p99 |
|---|---|---|---|
| female | 140.0 cm | 158.5 cm | — |
| male | — | 171.5 cm | 196.0 cm |

E.5 whole-body mass rows: female p1 = 38 kg, male p99 = 120 kg. **No p50 mass.**

So the honest identity set is **4**, not the 6 (2 × 3) a full grid would give:

| identity | stature | mass | residuals fitted |
|---|---|---|---|
| female p1 | 140.0 cm | 38 kg | height + mass |
| female p50 | 158.5 cm | — | height only; `mass` in `unmapped_measurements` |
| male p50 | 171.5 cm | — | height only; `mass` in `unmapped_measurements` |
| male p99 | 196.0 cm | 120 kg | height + mass |

Every other E.2 dimension (sitting height, breadths, hand length, head
circumference, IPD) has no ANNY phenotype knob and is recorded per row as
unmapped. Age: E.2/E.5 are not age-stratified, so every identity takes the
adult band (E.10 `adult_25_44`).

## Conflicts to raise, not resolve here

- E.13 `hammersley_views = 512` vs MaskScore's 64 scoring views and VoxHammer's
  150 feature views. Likely the full identity-corpus render count; recorded.
- E.13 `twist_bones = 8` vs `4-entities/godot-soma-twist` shipping answer of
  zero twist bones (wrist ramp). A genuine doctrine conflict between the spec
  table and the measured result.
- E.6 `source` string carries a mojibake byte (`g/cm�`) — an encoding defect
  in the dataset card, not a value error.
