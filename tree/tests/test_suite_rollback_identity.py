"""A suite or generic-seam backend's rollback can be verified.

THE BREAKAGE THIS FILE PREVENTS: the step transaction proves a rollback by
reading the frozen adapter's identity fields back from the backend's restart
identity and comparing them with its receipt
(``run_cuda_v841_full_physics_x4._verify_backend_rollback_boundary``).  The
column-suite backend inherits the receipt but published a restart identity
without those fields, so every failed step on a suite or generic-seam
backend ended "rollback was incomplete: adapter rollback restart identity
changed field 'adapter_contract_sha256'" instead of the refusal that caused
it (hex 30 km global at TEMPO, 2026-10-11, refused at step 1 for missing
aerosol fields).
"""

from __future__ import annotations

from types import SimpleNamespace

from hexcore import cuda_arwen_physics_v841 as frozen
from hexcore import cuda_suite_physics_v841 as suite

# The names _verify_backend_rollback_boundary reads from the restart identity.
VERIFIED = (
    "adapter_contract_sha256", "arwen_commit", "arwen_source_manifest",
    "contract_surface_sha256", "glacier_composed_tu_sha256",
    "constructor_identity_sha256", "prep_contract_sha256",
    "gwdo_contract_sha256",
)


def _backend(cls):
    backend = object.__new__(cls)
    backend._engine = SimpleNamespace(
        build_commit="abc123", manifest={"gpuwm/core/physics.py": "00"},
        glacier_composed_tu_sha256="11", version="2.8.9", root="/engine")
    backend._constructor = SimpleNamespace(
        identity_sha256="22", selection={"pbl_scheme": "ysu"})
    backend._profile = {"id": "engine-0123456789"}
    backend._gwdo_static = None
    return backend


def test_suite_restart_identity_carries_every_field_the_rollback_check_reads():
    base = frozen.PersistentTwoPhaseCudaPhysicsBackendV841._restart_identity(
        _backend(frozen.PersistentTwoPhaseCudaPhysicsBackendV841))
    identity = _backend(suite.PersistentSuiteCudaPhysicsBackendV841)._restart_identity()
    for name in VERIFIED:
        assert name in identity, name
        assert identity[name] == base[name], name
    assert identity["schema"] == suite.CUDA_SUITE_PHYSICS_V841_SCHEMA
    assert identity["profile"] == "engine-0123456789"


def test_the_verifier_reads_no_identity_field_outside_that_list():
    import inspect

    from hexcore.drivers import run_cuda_v841_full_physics_x4 as driver

    source = inspect.getsource(driver._verify_backend_rollback_boundary)
    block = source[source.index("expected_restart_identity = {"):source.index("for name, expected")]
    read = {line.split(":")[0].strip().strip('"') for line in block.splitlines()
            if line.strip().startswith('"')}
    assert read == set(VERIFIED)


def _receipt_and_identity(cls, *, drag):
    backend = _backend(cls)
    backend._gwdo_static = object() if drag else None
    identity = cls._restart_identity(backend)
    receipt = {
        "adapter_contract_sha256": frozen.CUDA_ARWEN_PHYSICS_V841_CONTRACT_SHA256,
        "arwen_commit": "abc123",
        "arwen_source_manifest": {"gpuwm/core/physics.py": "00"},
        "contract_surface_sha256": frozen.MPAS_SEAM_CONTRACT_SURFACE_SHA256,
        "glacier_composed_tu_sha256": "11",
        "constructor": {"identity_sha256": "22"},
        "dependencies": dict(frozen.ARWEN_V841_DEPENDENCY_CONTRACTS)
        if hasattr(frozen, "ARWEN_V841_DEPENDENCY_CONTRACTS") else {
            "prep_contract_sha256": frozen.CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256,
            "gwdo_contract_sha256": frozen.CUDA_GWDO_V841_CONTRACT_SHA256,
        },
        "gwdo": {"enabled": drag},
    }
    return receipt, identity


def test_the_rollback_identity_check_passes_with_and_without_hex_drag(monkeypatch):
    """Drive the verifier's identity comparison with the drag on and off."""
    from hexcore.drivers import run_cuda_v841_full_physics_x4 as driver

    for cls in (frozen.PersistentTwoPhaseCudaPhysicsBackendV841,
                suite.PersistentSuiteCudaPhysicsBackendV841):
        for drag in (True, False):
            receipt, identity = _receipt_and_identity(cls, drag=drag)
            receipt["phase"] = "boundary"
            receipt["start_time_seconds"] = 0.0
            receipt["schema"] = "s"

            class Backend:
                def step_receipt(self):
                    return receipt

                def restart_state(self):
                    return {"schema": "s", "identity": identity,
                            "seam": {"identity": {}, "arrays": {},
                                     "scalars": {"elapsed_seconds": 0.0}},
                            "adapter": {}}

            try:
                driver._verify_backend_rollback_boundary(
                    Backend(), expected_phase="boundary", start_time=0.0,
                    transaction_identity=driver._backend_transaction_identity(receipt))
            except RuntimeError as error:
                assert "identity changed field" not in str(error), (cls.__name__, drag, error)
