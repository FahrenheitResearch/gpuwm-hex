import json
import time

import pytest

from hexcore.cuda_backend import regional_admission as admission
from hexcore.speculative_contract import ENV, SCHEMA, validate_output


def arguments():
    klass = admission.ADMITTED_CLASSES["graded-869m-dt5-z7"]
    key = klass.key
    return dict(mesh_row="new-point", bdy_mask_sha256="a" * 64, n_cells=100,
                boundary_zone_width=key.boundary_zone_width, n_vert_levels=key.n_vert_levels,
                finest_edge_m=key.finest_edge_mm / 1000., dt_seconds=key.dt_ms / 1000.,
                kernel_set=key.kernel_set_sha256)


def test_ordinary_uncontracted_caller_retains_refusal(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(admission.CONTRACT_LEDGER_ENVIRONMENT, raising=False)
    with pytest.raises(admission.RegionalAdmissionRefusal, match="contract"):
        admission.require_regional_anchor(**arguments())


def test_speculative_ticket_binds_geometry_class_kernels_and_output(tmp_path, monkeypatch):
    args = arguments()
    contract, output = tmp_path / "contract", tmp_path / "quarantine"
    contract.mkdir()
    doc = dict(schema=SCHEMA, status="pending", expires_unix=time.time() + 60,
               contract_dir=str(contract), quarantine_dir=str(output), bdy_mask_sha256=args["bdy_mask_sha256"],
               n_cells=args["n_cells"], kernel_set_sha256=args["kernel_set"], mesh_row=args["mesh_row"],
               class_id="graded-869m-dt5-z7")
    path = tmp_path / "ticket.json"; path.write_text(json.dumps(doc))
    monkeypatch.setenv(ENV, str(path)); monkeypatch.setenv(admission.CONTRACT_LEDGER_ENVIRONMENT, str(contract))
    validate_output(output)
    assert admission.require_regional_anchor(**args).contract_route == "speculative"
    shallow = {key: args[key] for key in ("mesh_row", "bdy_mask_sha256", "n_cells", "kernel_set")}
    assert admission.require_regional_anchor(**shallow).contract_route == "speculative"
    with pytest.raises(ValueError, match="quarantine"):
        validate_output(tmp_path / "published")
    for key, value in (("n_cells", 101), ("kernel_set_sha256", "b" * 64), ("class_id", "other"), ("bdy_mask_sha256", "b" * 64)):
        bad = {**doc, key: value}; path.write_text(json.dumps(bad))
        with pytest.raises(admission.RegionalAdmissionRefusal, match="identity differs"):
            admission.require_regional_anchor(**args)
    path.write_text(json.dumps({**doc, "expires_unix": 0}))
    with pytest.raises(ValueError, match="expired"):
        validate_output(output)
