#!/usr/bin/env python3
"""Capture the real admission-probe compile request for wheel construction."""
import argparse
import json
from pathlib import Path

from hexcore.cuda_backend.aot import capture_requests
from hexcore.cuda_backend.arch_admission import measure_numeric_route


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with capture_requests() as rows:
        result = measure_numeric_route()
    args.output.write_text(json.dumps({"requests": list(rows.values()), "probe": result}, indent=2) + "\n")


if __name__ == "__main__":
    main()
