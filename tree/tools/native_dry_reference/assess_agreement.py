"""Turn a one-step measurement into a per-field tolerance table and exit status."""
import json
from pathlib import Path
import sys
from hexcore.dycore_agreement import STRICT_IDENTITY_RECIPES, assess, print_table


def main():
    path = Path(sys.argv[1])
    raw = json.loads(path.read_text())
    results = {}
    for backend in ("cpu_one_step_from_each_native_output", "cuda_continuous_from_native_time_zero"):
        if backend not in raw:
            continue
        steps = raw[backend]["steps"]
        if len(steps) != 1 or steps[0]["step"] != 1:
            raise ValueError("agreement gate covers exactly step one")
        # <recipe>.<schedule>.<backend>.raw.json: a strict GNU native is held
        # to byte identity, an optimised one to the per-field bounds.
        identity = path.name.split(".")[0] in STRICT_IDENTITY_RECIPES
        result = assess(steps[0]["fields"], identity=identity)
        results[backend] = result
        print_table(result, path.name + " " + backend)
    payload = {"schema": "gpuwm-hex.native-dry-agreement/v1", "source": path.name,
               "config": raw["config"], "hex_commit": raw["hex_commit"],
               "passed": bool(results) and all(r["passed"] for r in results.values()), "results": results}
    path.with_suffix(".agreement.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
