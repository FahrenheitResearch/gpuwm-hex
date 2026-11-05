"""Price the MPAS route before warm-up, pool decode or device upload."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
from types import SimpleNamespace

from ..device_admission import model_for_card, route_required_free_bytes


def rank_requirements(local_cells, devices, profiles, levels):
    """Sum every context assigned to a card, including each rank's halos."""
    if len(local_cells) != len(devices):
        raise ValueError("one local cell count is required per rank device")
    required = defaultdict(int)
    for cells, device in zip(local_cells, devices):
        model = model_for_card(profiles[device], "global", levels=int(levels))
        required[device] += route_required_free_bytes(int(cells), model, route="mpas")
    return dict(required)


def amd_capacity_trial(local_cells, devices, *, global_cells, input_payload_bytes,
                       reserve_bytes=8 * 1024**3):
    """An experimental allocation trial, never a measured AMD capacity fit.

    CUDA local-frame and sampled whole-run fits do not describe ROCm. Until
    an AMD whole-run row exists, charge each context its halo-inclusive share
    of the actual input payload plus a declared 8 GiB startup reserve. This
    is a startup budget, NOT an upper bound on forecast memory. Normal HIP
    allocations still enforce physical free memory; no managed spill or
    allocator substitution is enabled. Whole-run telemetry measures the fit.
    """
    if global_cells <= 0 or input_payload_bytes < 0 or reserve_bytes < 0:
        raise ValueError('invalid AMD capacity-trial dimensions or reserve')
    if len(local_cells) != len(devices):
        raise ValueError('one local cell count is required per rank device')
    required = defaultdict(int)
    for cells, device in zip(local_cells, devices):
        if cells <= 0:
            raise ValueError('rank cell counts must be positive')
        required[device] += math.ceil(input_payload_bytes * int(cells) / global_cells) + reserve_bytes
    return dict(required)


def input_payload_bytes(path):
    """Metadata only: count the pinned init's numeric/S1 words, without decode."""
    from netCDF4 import Dataset
    import numpy as np
    with Dataset(path) as dataset:
        return sum(int(value.size) * np.dtype(value.dtype).itemsize
                   for value in dataset.variables.values()
                   if value.dtype is not str)


def admit(arguments):
    from ..mesh_rows import _dimensions
    from ..drivers.samehost_forecast_v841 import device_list, request_options
    from ..forecast_door import measure_device_memory, read_card_profile

    shape = _dimensions(Path(arguments.init))
    cells, levels = shape["nCells"], shape["nVertLevels"]
    named = getattr(arguments, "samehost_devices", None)
    devices = device_list(named) if named else (0,)
    if named:
        request = SimpleNamespace(grid=Path(getattr(arguments, "grid", None) or arguments.init),
                                  scratch=Path(arguments.out), cells=cells)
        options = request_options(arguments, request)
        local_cells = [int(layout.n_local_cells) for layout in options["layouts"]]
        # Keep the exact proven layout for the run; do not build it twice.
        arguments.samehost_layouts = options["layouts_path"]
        arguments.samehost_part = options["part_path"]
    else:
        local_cells = [cells]
    import cupy as cp
    profiles, memory = {}, {}
    for device in sorted(set(devices)):
        with cp.cuda.Device(device):
            # These probes read the current device; no model arrays or kernels
            # have been allocated when a request is refused here.
            profiles[device] = read_card_profile(device)
            memory[device] = measure_device_memory()
    physical_amd = bool(cp.cuda.runtime.is_hip)
    payload = input_payload_bytes(arguments.init) if physical_amd else None
    required = (amd_capacity_trial(local_cells, devices, global_cells=cells,
                                  input_payload_bytes=payload)
                if physical_amd else rank_requirements(local_cells, devices, profiles, levels))
    record = {"route": "mpas", "levels": levels, "local_cells": local_cells,
              "devices": list(devices), "required_bytes": required,
              "free_bytes": {device: value[0] for device, value in memory.items()}}
    record.update(backend='amd' if physical_amd else 'cuda',
                  qualified_capacity=not physical_amd,
                  capacity_trial=physical_amd,
                  basis=('experimental startup budget: halo-inclusive init payload share + 8 GiB/context; '
                         'forecast peak is unmeasured; normal physical HIP allocator'
                         if physical_amd else 'existing CUDA route footprint model'),
                  input_payload_bytes=payload)
    Path(arguments.out).mkdir(parents=True, exist_ok=True)
    import json
    (Path(arguments.out) / 'device-memory-admission.json').write_text(json.dumps(record, indent=2)+'\n')
    arguments.native_memory_admission = record
    for device, need in required.items():
        free = memory[device][0]
        if need > free:
            raise ValueError(f"MPAS device admission: card {device} needs {need} bytes for its ranks "
                             f"(halos included), but has {free} free; use more cards or a smaller mesh")
    return record
