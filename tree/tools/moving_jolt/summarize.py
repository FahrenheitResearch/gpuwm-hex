import hashlib, json
from pathlib import Path
import numpy as np

names = [
    "cpu-three-moves",
    "cpu-regenerated-three-moves",
    "gpu-three-moves",
    "gpu-regenerated-three-moves",
    "gpu-lts-three-moves",
    "gpu-lts-rates-only",
    "gpu-refined-three-moves",
    "gpu-refined-lts-three-moves",
]
summary = {}
errors = {key: 0.0 for key in ["mass", "theta_mass", "tracer"]}
for name in names:
    x = json.loads(Path("evidence", name + ".json").read_text())
    summary[name] = x["metrics"]
    for arm in ("balanced",):
        if arm not in x["results"]:
            continue
        for b in x["results"][arm]["budgets"]:
            b = b["budget"]
            for key, alias in [
                ("mass_total_kg", "mass"),
                ("theta_mass_total_kg_kelvin", "theta_mass"),
            ]:
                errors[alias] = max(
                    errors[alias], abs(b[key]["absolute_error"] / b[key]["before"])
                )
            for val in b["tracer_totals_kg"].values():
                errors["tracer"] = max(
                    errors["tracer"], abs(val["absolute_error"] / val["before"])
                )
fixed = {}
for lts in (False, True):
    current = json.loads(
        Path(
            "evidence", "gpu-final-" + ("lts-" if lts else "") + "fixed.json"
        ).read_text()
    )["results"]["fixed"]
    for base in ("original", "accepted"):
        old = json.loads(
            Path(
                "evidence", "gpu-" + base + "-" + ("lts-" if lts else "") + "fixed.json"
            ).read_text()
        )["results"]["fixed"]
        key = base + ("-lts" if lts else "")
        fixed[key] = dict(
            state_identical=current["final_hashes"] == old["final_hashes"],
            series_identical=current["rows"] == old["rows"],
        )
        if "saved_hashes" in old:
            fixed[key]["saved_identical"] = (
                current["saved_hashes"] == old["saved_hashes"]
            )
        assert all(fixed[key].values()), fixed[key]
Path("evidence/summary.json").write_text(
    json.dumps(
        dict(metrics=summary, worst_move_conservation=errors, fixed_identity=fixed),
        indent=2,
    )
    + "\n"
)
print(json.dumps(dict(worst_move_conservation=errors, fixed_identity=fixed), indent=2))
for name in ["gpu-refined-three-moves", "gpu-refined-lts-three-moves"]:
    print(name)
    for row in summary[name]:
        print(row)
