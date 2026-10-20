"""One-step native MPAS agreement, with a deliberately bounded claim.

Against the strict GNU native the bound is byte identity (0 differing words,
``assess(..., identity=True)``).  Against an optimised native: six binary32
ULP for nonzero thermodynamic fields. The six Intel/CUDA
measurements on 2026-10-11 reached five ULP in theta_m, four in density
and three in Exner pressure. One additional ULP is regression headroom.
This is not the old nine-ULP endpoint fixture. Near-zero fields need an
absolute bound; their raw ULP distance is still reported, never hidden.
"""
from __future__ import annotations

import numpy as np

ULP_LIMIT = 6
LIMITS = {
    "rho": ("ulp", ULP_LIMIT), "rho_theta": ("ulp", ULP_LIMIT),
    "theta_m": ("ulp", ULP_LIMIT), "exner": ("ulp", ULP_LIMIT),
    "scalars": ("ulp", 0),
    "rho_u": ("abs", 8e-5), "rho_w": ("abs", 8e-6),
    "density_perturbation": ("abs", 1e-7),
    "rho_theta_perturbation": ("abs", 2.5e-5),
    "pressure_perturbation": ("abs", 0.025),
    "normal_velocity": ("abs", 2e-4), "vertical_velocity": ("abs", 3e-5),
}


def ulp_distance(a, b):
    def ordered(x):
        bits = np.ascontiguousarray(x, dtype=np.float32).view(np.int32).astype(np.int64)
        return np.where(bits < 0, -(bits & 0x7fffffff), bits)
    return np.abs(ordered(a) - ordered(b))


def field_report(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.dtype != np.float32:
        return {"identical": False, "finite": False,
                "why": f"expected matching binary32 arrays: {actual.shape}/{actual.dtype}, {expected.shape}/{expected.dtype}"}
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    different = np.ascontiguousarray(actual).view(np.uint32) != np.ascontiguousarray(expected).view(np.uint32)
    return {"identical": not bool(different.any()), "finite": finite,
            "words": int(actual.size), "different_words": int(different.sum()),
            "max_ulp": int(ulp_distance(actual, expected).max(initial=0)),
            "max_abs": float(np.abs(actual.astype(np.float64)-expected).max(initial=0)) if finite else None}


#: Native builds the dycore claim is stated against: GNU Fortran single
#: precision with no FP contraction or vectorised reassociation.  Against
#: these the claim is byte identity (dycore-match, 2026-10-11: 0 of 66,612
#: differing values after one step on x1.2562), so the bound is 0 differing
#: words in every field.  Against an optimised native the per-field ULP and
#: absolute bounds below apply, because two native builds differ by as much.
STRICT_IDENTITY_RECIPES = ("gnu-o0", "gnu-strict")


def assess(fields, *, identity=False):
    """Require every field, finite matching shapes, and every stated bound.

    ``identity`` (a strict GNU native): every field must have 0 differing
    words, whatever its ULP or absolute distance.
    """
    if identity:
        return _assess_identity(fields)
    rows = {}
    for field, (metric, limit) in LIMITS.items():
        value = fields.get(field, {})
        measured = value.get("max_" + metric)
        # Older archived reports omit metrics when all words agree.
        if value.get("identical") and measured is None:
            measured = 0
        passed = (value.get("finite", "why" not in value) and value.get("words",0)>0 and measured is not None
                  and np.isfinite(measured) and measured <= limit)
        rows[field] = {**value, "metric": metric, "limit": limit, "passed": bool(passed)}
    return {"passed": all(r["passed"] for r in rows.values()), "fields": rows}


def _assess_identity(fields):
    rows = {}
    for field in LIMITS:
        value = fields.get(field, {})
        words = value.get("different_words")
        # Older archived reports omit counts when all words agree.
        if value.get("identical") and words is None:
            words = 0
        passed = (value.get("finite", "why" not in value) and value.get("words", 0) > 0
                  and words is not None and int(words) == 0)
        rows[field] = {**value, "metric": "words", "limit": 0, "passed": bool(passed)}
    return {"passed": all(r["passed"] for r in rows.values()), "fields": rows, "identity": True}


def print_table(result, label):
    print(label)
    print(f'{"field":27} {"differing/values":18} {"max ULP":12} {"max absolute":13} {"tolerance":14} result')
    for name, row in result["fields"].items():
        count = f'{row.get("different_words",0)}/{row.get("words","?")}'
        gap = row.get("max_abs")
        gap_text = "invalid" if gap is None else f"{gap:.6g}"
        print(f'{name:27} {count:18} {row.get("max_ulp",0):12} '
              f'{gap_text:13} {str(row["limit"])+" "+row["metric"]:14} '
              f'{"PASS" if row["passed"] else "FAIL"}')
    print("PASS" if result["passed"] else "FAIL")
