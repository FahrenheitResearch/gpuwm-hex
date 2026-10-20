"""Run rw-mpas's actual proof-policy regressions in the Hex CPU tier.

These tests check receipt and admission integrity; they are not weather proof.
The source artifact is the delivered rw-mpas checkout, not a copied fixture.
"""

import importlib.util
import os
from pathlib import Path
import sys

import pytest


# A lane checkout has a sibling gpuwm checkout. Trial-merge runners can name
# their delivered Rust source explicitly without relying on lane directories.
rust_source = Path(os.environ.get(
    "RW_MPAS_SOURCE", str(Path(__file__).resolve().parents[3] / "gpuwm")))
oracle = rust_source / "tools/rustwx/crates/rw-mpas/tests/surface_oracle"
test_source = oracle / "test_surface_proof_policy.py"
if not test_source.is_file():
    pytest.skip("delivered rw-mpas source artifact unavailable; set RW_MPAS_SOURCE",
                allow_module_level=True)

spec = importlib.util.spec_from_file_location("i5_rw_mpas_proof_policy", test_source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
sys.path.insert(0, str(oracle))
try:
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)

# Pytest collects the original unittest class and executes the actual helpers.
SurfaceProofPolicy = module.SurfaceProofPolicy
