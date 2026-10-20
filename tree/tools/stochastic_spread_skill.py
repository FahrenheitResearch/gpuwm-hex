"""Plot and grade ten daily receipts from score_pair_gpu, without weather I/O."""
import argparse
import json
from pathlib import Path
import time

from hexcore.stochastic_verification import plot_daily, qualify_daily


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="+", type=Path)
    parser.add_argument("--cycle", required=True, help="initial time with explicit UTC offset")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    result = qualify_daily([json.loads(path.read_text()) for path in args.receipts], cycle=args.cycle)
    result["plots"] = plot_daily(result, args.out)
    result["receipt_paths"] = [str(path) for path in args.receipts]
    result["analysis_chart_wall_seconds"] = time.perf_counter()-started
    (args.out / "spread-skill.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(spread_skill_passed=result["spread_skill_passed"],
                          qualification_complete=False, plots=result["plots"])))
    return 0 if result["spread_skill_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
