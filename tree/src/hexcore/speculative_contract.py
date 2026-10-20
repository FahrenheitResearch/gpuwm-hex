"""Explicit quarantine admission while a new geometry's deck runs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

ENV = "GPUWM_HEX_SPECULATIVE_CONTRACT"
SCHEMA = "gpuwm-hex.speculative-contract/v1"


def ticket():
    raw = os.environ.get(ENV)
    if not raw:
        return None
    path = Path(raw)
    document = json.loads(path.read_text())
    if document.get("schema") != SCHEMA or document.get("status") != "pending":
        raise ValueError(f"{path}: invalid speculative contract; unchecked geometry cannot be published")
    if float(document.get("expires_unix", 0)) <= time.time():
        raise ValueError(f"{path}: speculative contract expired; unchecked geometry cannot be published")
    directory = Path(document["contract_dir"]).resolve()
    configured = os.environ.get("GPUWM_HEX_REGIONAL_CONTRACT_DIR", "")
    if directory not in [Path(item).resolve() for item in configured.split(os.pathsep) if item]:
        raise ValueError(f"{path}: contract directory differs; refuses an unrelated deck")
    return document


def validate_output(output):
    document = ticket()
    if document is not None and Path(document["quarantine_dir"]).resolve() != Path(output).resolve():
        raise ValueError("speculative forecast destination differs from quarantine_dir; unchecked geometry cannot be published")


def pending_contract(*, digest, cells, kernel_set, measured_key, mesh_row):
    from .cuda_backend import regional_admission as admission
    document = ticket()
    if document is None:
        return None
    klass = (admission.admitted_class_for_key(measured_key) if measured_key is not None
             else admission.admitted_class(document.get("class_id")))
    expected = {"bdy_mask_sha256": digest, "n_cells": cells, "kernel_set_sha256": kernel_set,
                "mesh_row": mesh_row, "class_id": klass.class_id if klass else None}
    if any(document.get(key) != value for key, value in expected.items()) or klass is None:
        raise admission.RegionalAdmissionRefusal(
            "speculative contract identity differs from the measured geometry, class or kernels; "
            "refuses a forecast whose publication gate would check another configuration")
    if not klass.key.finest_edge_measured:
        raise admission.RegionalAdmissionRefusal("speculative class has no measured finest edge; refuses an unqualified resolution")
    return admission.RegionalContract(
        bdy_mask_sha256=digest, n_cells=int(cells), boundary_zone_width=(measured_key or klass.key).boundary_zone_width,
        class_id=klass.class_id, mesh_row=mesh_row, card="pending", admitted_on="pending",
        contract_receipt=os.environ[ENV], basis="SPECULATIVE: publication held in quarantine until the complete deck passes")
