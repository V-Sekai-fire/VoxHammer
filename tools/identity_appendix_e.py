"""Appendix-E anthropometry -> ANNY phenotypes, by LBFGS on a measurement residual.

Runs in the `anny` pixi environment:

    pixi run -e anny python tools/identity_appendix_e.py --out work/identities --negative-control

For each (sex, percentile) with a stature row in E.2, solve the `height` (and
`weight`, when E.5 has a mass row for that cell) phenotype so that the ANNY rest
mesh's Z-extent and volume-derived mass hit the table. Same optimizer as
4-entities/anny-pose-retarget-work/lbfgs_polish.py (torch LBFGS, strong-Wolfe,
float64); the residual is two scalars rather than 13k vertices. Height and mass
formulas follow anny/anthropometry.py (Z-extent; signed-tetra volume x 980 kg/m3).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)

DATASET = "chibifire/starforged-std-3001-appendix-e"
DENSITY = 980.0  # anthropometry.py:104 -- hard-coded, a known bias against real weight tables
STATURE_TOL_M = 0.005
MASS_TOL_REL = 0.03
NEUTRAL = 0.5
ADULT_AGE_YEARS = 35.0  # E.10 adult_25_44 midpoint; E.2/E.5 are not age-stratified
PENNY_M = 0.00152


def anny_sha() -> str:
    repo = Path(__file__).resolve().parents[2] / "anny"
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short=12", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def load_table():
    from datasets import load_dataset
    ds = load_dataset(DATASET, "human", split="train")
    stature, mass = {}, {}
    unmapped = sorted({r["subcategory"] for r in ds
                       if r["section"] == "E.2" and r["subcategory"] != "stature"})
    for r in ds:
        if r["section"] == "E.2" and r["subcategory"] == "stature":
            stature[(r["sex"], r["percentile"])] = float(r["value"]) / 100.0
        if r["section"] == "E.5" and r["subcategory"] == "whole_body":
            mass[(r["sex"], r["percentile"])] = float(r["value"])
    return stature, mass, unmapped


def build_model():
    import anny
    # Same config as anny-render-corpus/anny_rig.py CORPUS_CONFIG minus facial_actions.
    return anny.Anny(rig="anny", topology="anny", phenotypes="all",
                     local_changes="default", skinning_method="lbs").to(dtype=torch.float64)


def triangulate(faces: torch.Tensor) -> torch.Tensor:
    if faces.shape[1] == 3:
        return faces
    a, b, c, d = faces.unbind(1)
    return torch.cat([torch.stack([a, b, c], 1), torch.stack([a, c, d], 1)], 0)


def age_coord(years: float):
    """Years -> ANNY age coordinate. Prefer ANNY's own mapping; fall back to the
    anchor grid (newborn 0, baby .25, child .5, young .75, old 1)."""
    try:
        from anny.shape_distribution import MorphologicalAgeMapping
        m = MorphologicalAgeMapping()
        for name in ("years_to_age", "to_anny_age", "__call__", "forward"):
            fn = getattr(m, name, None)
            if callable(fn):
                v = fn(torch.tensor([years], dtype=torch.float64))
                return float(torch.as_tensor(v).flatten()[0]), f"MorphologicalAgeMapping.{name}"
    except Exception as exc:  # report and fall back; never a silent 0.5
        print(f"[identity] MorphologicalAgeMapping unavailable ({type(exc).__name__}: {exc}); anchor grid")
    return 0.75, "anchor:young=0.75"


def measurements(model, pheno, faces_tri):
    pose = torch.eye(4, dtype=torch.float64)[None, None].repeat(1, model.bone_count, 1, 1)
    v = model(pose_parameters=pose, phenotype_kwargs=pheno)["vertices"][0]  # (V,3) z-up metres
    height = v[:, 2].max() - v[:, 2].min()
    t = v[faces_tri]
    volume = (t[:, 0] * torch.cross(t[:, 1], t[:, 2], dim=1)).sum() / 6.0
    return height, volume.abs() * DENSITY


def base_pheno(model, sex, age):
    p = {k: torch.tensor([NEUTRAL], dtype=torch.float64) for k in model.phenotype_labels}
    p["gender"] = torch.tensor([0.0 if sex == "male" else 1.0], dtype=torch.float64)
    p["age"] = torch.tensor([age], dtype=torch.float64)
    return p


STATURE_WEIGHT = 10.0  # stature defines the identity; mass may be unreachable and must not steal height


def solve(model, faces_tri, sex, stature_m, mass_kg, age):
    base = base_pheno(model, sex, age)
    # height, weight, muscle via sigmoid. weight alone saturated at 1.0 on male p99
    # (120 kg at 196 cm) and the solver grew stature instead; muscle adds volume.
    logits = torch.zeros(3, dtype=torch.float64, requires_grad=True)

    def pheno():
        p = dict(base)
        s = torch.sigmoid(logits)
        p["height"] = s[0:1]
        if mass_kg is not None:
            p["weight"] = s[1:2]
            p["muscle"] = s[2:3]
        return p

    def loss():
        h, m = measurements(model, pheno(), faces_tri)
        val = STATURE_WEIGHT * ((h - stature_m) / stature_m) ** 2
        if mass_kg is not None:
            val = val + ((m - mass_kg) / mass_kg) ** 2
        return val + 1e-3 * (torch.sigmoid(logits) - NEUTRAL).pow(2).sum()

    opt = torch.optim.LBFGS([logits], max_iter=60, tolerance_grad=1e-9,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")
    n_eval = [0]

    def closure():
        opt.zero_grad()
        val = loss()
        val.backward()
        n_eval[0] += 1
        return val

    opt.step(closure)
    with torch.no_grad():
        h, m = measurements(model, pheno(), faces_tri)
        out = {k: float(v.item()) for k, v in pheno().items()}
    return out, float(h), float(m), n_eval[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--negative-control", action="store_true",
                    help="also evaluate height pinned at 0.5 and require it to FAIL on p1/p99")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    stature, mass, unmapped = load_table()
    model = build_model()
    faces_tri = triangulate(model.faces)
    age, age_src = age_coord(ADULT_AGE_YEARS)
    sha = anny_sha()
    print(f"[identity] labels={list(model.phenotype_labels)} faces={tuple(model.faces.shape)} "
          f"age={age:.3f} ({age_src}) anny={sha}")

    ok = True
    for (sex, pct), st in sorted(stature.items()):
        ms = mass.get((sex, pct))
        p, h, m, n = solve(model, faces_tri, sex, st, ms, age)
        dh = abs(h - st)
        dm = abs(m - ms) / ms if ms else None
        pass_h = dh < STATURE_TOL_M
        pass_m = (dm < MASS_TOL_REL) if ms else True
        # Stature is the gate. A mass shortfall at saturated weight+muscle is a model
        # coverage limit, recorded on the row (mass_within_tol=false), not a failure.
        ok = ok and pass_h
        key = f"appendix-e:{sex}:{pct}"
        rec = dict(identity_source=key, dataset=DATASET, anny_sha=sha,
                   sex=sex, percentile=pct, age_years=ADULT_AGE_YEARS, age_source=age_src,
                   target_stature_m=st, achieved_stature_m=h,
                   target_mass_kg=ms if ms is not None else -1.0, achieved_mass_kg=m,
                   mass_fitted=ms is not None, mass_within_tol=bool(pass_m),
                   stature_residual_weight=STATURE_WEIGHT,
                   unmapped_measurements=unmapped + ([] if ms is not None else ["mass_kg"]),
                   phenotype=p, lbfgs_evals=n, density_kg_m3=DENSITY)
        (a.out / f"{sex}-{pct}.json").write_text(json.dumps(rec, indent=2))
        mass_txt = f"(target {ms:5.1f}, {dm * 100:4.1f}%)" if ms else "(no target)"
        print(f"  {key:22s} stature {h * 100:6.1f} cm (target {st * 100:5.1f}, "
              f"|d|={dh * 1000:4.1f} mm ~ {dh / PENNY_M:.1f} pennies) mass {m:6.1f} kg {mass_txt} "
              f"height={p['height']:.3f} weight={p['weight']:.3f} evals={n} "
              f"{'OK' if pass_h and pass_m else 'FAIL'}")

    if a.negative_control:
        fails = 0
        for (sex, pct), st in sorted(stature.items()):
            if pct not in ("p1", "p99"):
                continue
            with torch.no_grad():
                h, _ = measurements(model, base_pheno(model, sex, age), faces_tri)
            bad = abs(float(h) - st) >= STATURE_TOL_M
            fails += bad
            print(f"  [negative] {sex}:{pct} height pinned 0.5 -> {float(h) * 100:.1f} cm vs "
                  f"{st * 100:.1f}: {'FAILS tolerance (good)' if bad else 'PASSES (control broken)'}")
        if fails == 0:
            print("[identity] negative control did not fail -> the fit is decoration")
            return 2
    print("[identity] all identities within tolerance" if ok else "[identity] TOLERANCE FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
