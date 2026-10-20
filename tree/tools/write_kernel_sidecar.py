"""Mint the kernel sidecar after any package namespace transformation.

The aggregate wheel builder runs this once against its staged package.
Job startup only reads the resulting JSON.
"""
import argparse
import importlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default="hexcore")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    admission = importlib.import_module(args.package + ".cuda_backend.regional_admission")
    document = {"schema": "gpuwm-hex.kernel-set/v1", "kernel_set_sha256": admission.kernel_set_sha256()}
    args.output.write_text(json.dumps(document, sort_keys=True) + "\n")
    print(json.dumps(document, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
