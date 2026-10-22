"""Full-window weekly ERA5 and RMM verification, reduced by rw_s2s."""
from __future__ import annotations

from pathlib import Path

from .s2s_door import write_json
from .s2s_mjo import run_backend


def verify_weeks(value, state, args, env):
    output = Path(value["paths"]["work"]) / "s2s-scores"
    if not args.s2s_observations:
        result = {"status": "unavailable", "reason": "--s2s-observations is required for weekly ERA5 "
                  "ACC/RMSE and matched RMM bivariate correlation", "coverage_days": 0}
        write_json(output / "receipt.json", result)
        return result
    work = Path(value["paths"]["work"])
    plan = {"observation_manifest": str(Path(args.s2s_observations).absolute()),
            "products": str(work / "s2s-products"), "mjo": str(work / "s2s-mjo" / "receipt.json"),
            "output": str(output), "days": args.days, "start": value["start_time"],
            "members": [member["run_id"] for member in value["members"]],
            "research_data_license": args.s2s_reference_license}
    return run_backend("score", plan, output, env)


def verify(value, state, args, env):
    from .s2s_products import verify_first_ten_days
    # Qualified full-window verification is preferred. The legacy analysis
    # scorer remains available when no ERA5 weekly observation manifest exists.
    if args.s2s_observations:
        return verify_weeks(value, state, args, env)
    analysis = verify_first_ten_days(value, state, args, env)
    return {**analysis, "weekly_era5": verify_weeks(value, state, args, env)}
