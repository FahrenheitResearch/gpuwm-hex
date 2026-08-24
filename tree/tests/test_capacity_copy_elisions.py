"""Static gates for the bit-preserving capacity copy-elision patch.

These tests do not make a hardware or output-equivalence claim. They ensure the
specific read-only bindings, named refusal, and exact source-pin migration do
not silently disappear before the required CUDA A/B runs are performed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "mpas_port"

EXPECTED_SOURCE_SHA256 = {
    SRC / "cuda_driver.py": (
        "9daf917a89b3b9dd6f013be3d971c76d255bcfbbb9c1027b9de0c8823cb49e66"
    ),
    SRC / "cuda_horizontal.py": (
        "97faf0869a0a5ea9ebbc4c67b3c2d6c68cefdfa10dece73cd204d818962efde4"
    ),
    SRC / "cuda_horizontal_v841.py": (
        "3fc0b860ebd67dfed453617c348810964ea1110e782fe85db10283afb406e2fe"
    ),
}


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _between(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def test_modified_execution_sources_match_the_migration_table() -> None:
    for path, expected in EXPECTED_SOURCE_SHA256.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_recovery_velocities_only_mode_refuses_incomplete_host_publish() -> None:
    source = _source(SRC / "cuda_backend" / "recovery.py")
    assert "include_pressure: bool = True" in source
    assert "if include_pressure:" in source
    assert "recover_state(include_pressure=False) did not compute" in source
    assert "to_host() would publish a field set that silently" in source
    assert '"pressure": pressure_timing' in source
    assert "if include_pressure\n            else" in source


def test_candidate_recovery_binds_scalars_and_skips_discarded_fields() -> None:
    source = _source(SRC / "cuda_driver.py")
    body = _between(source, "    def _recover_candidate(", "    def _copy_state(")
    assert "self._device_state(rho, rtheta, ru, rw, scalars)" in body
    assert "cp.array(scalars, copy=True)" not in body
    assert "include_pressure=False" in body


def test_dynamics_subcycle_uses_read_only_aliases_only_at_proven_lifetimes() -> None:
    source = _source(SRC / "cuda_driver.py")
    copy_body = _between(source, "    def _copy_state(", "    def _copy_saved(")
    assert "share_scalars: bool = False" in copy_body
    assert "source.scalars if share_scalars else cp.array(source.scalars, copy=True)" in copy_body

    subcycle = _between(
        source,
        "    def _advance_dynamics_subcycle_v841(",
        "    def _step_device_v841(",
    )
    assert "saved_state = self._copy_state(state, share_scalars=True)" in subcycle
    assert "current_state = saved_state" in subcycle
    assert "current_diag = saved_diag" in subcycle
    assert "current_state = self._copy_state(saved_state)" not in subcycle
    assert "current_diag = self._copy_saved(saved_diag)" not in subcycle


def test_cached_tangential_velocity_is_bound_not_copied() -> None:
    for name in ("cuda_horizontal.py", "cuda_horizontal_v841.py"):
        source = _source(SRC / name)
        # The window opens at the guard above the binding so that it covers the
        # comment as well as the call.  Anchoring it on the _edge_field argument
        # instead would start below the comment and could only ever see the
        # closing paren.
        cached = _between(
            source,
            'raise ValueError("RK1/RK2 diagnostics require cached tangential velocity")',
            "        vorticity =",
        )
        assert ").copy()" not in cached
        assert "read" in cached or "const float*" in cached
