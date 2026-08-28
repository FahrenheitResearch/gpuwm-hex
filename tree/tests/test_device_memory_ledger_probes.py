"""The #264 instruments' self-validation controls are pinned to live premises.

Stale-guard audit #347, finding 6: reservation_probe's positive control was
pinned to ``gf_gfdrv_stage`` at 7,034.0 MiB -- the pre-cut in-run sum.  The
#294 Grell-Freitas frame cut shrank that frame to 88/72 B, so the instrument
declared its own technique invalid (``CONTROLS FAIL``, exit 1) against a
retired premise, and the sibling image probe never resolved the module that
now holds the widest launched frame.  These tests pin both instruments to
the post-cut premise (``wsm6_column``, 7,216 B -- STATE.md section 5) so the
next frame move fails HERE, by name, instead of inside a GPU session.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
LEDGER_TOOLS = ROOT / "tools" / "device_memory_ledger"


def _load(name: str):
    if str(LEDGER_TOOLS) not in sys.path:
        sys.path.insert(0, str(LEDGER_TOOLS))
    spec = importlib.util.spec_from_file_location(name, LEDGER_TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_reservation_probe_control_is_the_post_cut_widest_frame() -> None:
    probe = _load("reservation_probe")
    controls = [
        row for row in probe.DEFAULT_SWEEP_TARGETS if row[3] is not None
    ]
    kinds = {row[3] for row in controls}
    # One derived-bound positive control on the post-cut widest launched
    # frame, at its shipped call-site block; one exact negative control.
    assert ("gpuwm.core.kernels", "wsm6_column", 32, "bound") in controls
    assert 0.0 in kinds  # the zero-local negative control stays exact
    # The retired 7,034.0 MiB gf pin is gone from the control set.
    assert not any(row[3] == 7034.0 for row in probe.DEFAULT_SWEEP_TARGETS)
    assert not any(
        row[1] == "gf_gfdrv_stage" and row[3] is not None
        for row in probe.DEFAULT_SWEEP_TARGETS
    )


def test_reservation_probe_reaches_wsm6_through_the_registry_route() -> None:
    probe = _load("reservation_probe")
    assert probe.BUILDERS["gpuwm.core.kernels"] == ("load_module", ("wsm6",))


def test_reservation_probe_names_its_repin_and_follow_up() -> None:
    source = (LEDGER_TOOLS / "reservation_probe.py").read_text(encoding="utf-8")
    assert "#294" in source, "the control must cite the frame cut that re-pinned it"
    assert "7,216" in source, "the control must cite the post-cut frame it pins"
    assert "named follow-up" in source, (
        "the derived-bound control ships without a hardware run; the run "
        "that restores a measured-equality control must be named"
    )


def test_image_probe_resolves_the_module_holding_the_widest_frame() -> None:
    image = _load("module_image_probe")
    assert (
        "gpuwm.core.kernels",
        "load_module",
        ("wsm6",),
        "wsm6_column",
    ) in image.TARGETS
