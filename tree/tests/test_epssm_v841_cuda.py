"""The CUDA v8.4.1 step ignores a legacy ``config_epssm`` bitwise, as the CPU step does.

THE BREAKAGE THIS PREVENTS: the v8.4.1 lanes admit any real config_epssm
with upstream's warning, on the promise that no v8.4.1 numeric path reads
it.  tests/test_epssm_v841.py holds that promise by source and on the CPU;
this holds it on the card, where cuda_driver.py also carries v8.2.3 reads of
the knob.  A leaked value would change forecast bytes while the receipt said
the knob was ignored.

The deck is the v8.4.1 numeric ruler's (tools/run_cuda_v841_numeric_ruler.py)
on x1.2562; the mesh pair is an external asset and its absence skips.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import math
import pathlib
import sys
import warnings

import numpy as np
import pytest

from hexcore.config_v841 import LegacyEpssmIgnored

TREE = pathlib.Path(__file__).resolve().parents[1]
RULER_TOOL = TREE / "tools" / "run_cuda_v841_numeric_ruler.py"


def _ruler():
    name = "_epssm_ruler_tool"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RULER_TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _state_digest(state) -> str:
    digest = hashlib.sha256()
    for name in ("rho", "rho_theta", "rho_u", "rho_w", "scalars"):
        array = np.ascontiguousarray(np.asarray(getattr(state, name)))
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def test_the_cuda_v841_step_is_bitwise_identical_for_any_legacy_value():
    pytest.importorskip("cupy")
    from hexcore.cuda_driver import CudaDryDycoreDriver

    ruler = _ruler()
    missing = [str(p) for p in (ruler.DEFAULT_GRID, ruler.DEFAULT_STATIC) if not p.is_file()]
    if missing:
        pytest.skip(
            "the x1.2562 mesh pair is an external asset and is absent here "
            f"({', '.join(missing)})"
        )
    case = ruler.prepare_synthetic_case(config=ruler.ruler_config())
    if "defc_a" not in case.mesh.arrays:
        # A v8.4.1 init or static no longer carries the v8.2.3 deformation
        # coefficients (v8.4.1 builds its own at model start).  With mixing
        # off they are inactive, and the CUDA host contract takes zeros,
        # exactly as the production x4 driver attaches them.
        from hexcore.drivers.run_cuda_v841_full_physics_x4 import (
            attach_inactive_zero_deformation,
        )

        assert case.config.config_horiz_mixing == "off"
        attach_inactive_zero_deformation(case.mesh)

    def cuda_step(config):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LegacyEpssmIgnored)
            driver = CudaDryDycoreDriver.from_host(
                case.mesh,
                case.state.copy(),
                case.vertical,
                case.reference,
                config,
                saved_diagnostics=case.saved,
                terrain_metrics=case.terrain,
                advection_coefficients=case.coefficients,
                reference_wind_profiles=case.profiles,
            )
        return _state_digest(driver.step().state)

    base = case.config
    reference = cuda_step(base)
    for value in (0.1, 0.5, math.nan):
        assert cuda_step(dataclasses.replace(base, config_epssm=value)) == reference
    # Non-vacuous: the profile knob v8.4.1 DOES read moves the answer.
    assert cuda_step(dataclasses.replace(base, config_epssm_maximum=0.5)) != reference
