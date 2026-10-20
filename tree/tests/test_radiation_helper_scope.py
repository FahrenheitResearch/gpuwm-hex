"""Complete source sets and owning-thread scopes on the real imported runtime."""
from contextvars import copy_context
from pathlib import Path
import shutil
import threading

import pytest

from hexcore import cuda_arwen_helper_scope_v2 as scope


def runtime_root():
    import gpuwm.core.rrtmg_legacy as legacy
    return Path(legacy.__file__).resolve().parents[2]


def test_real_runtime_sources_and_receipt():
    bound = scope.bind_arwen_helpers_v2(runtime_root())
    assert bound.mode in scope.ADMITTED_SOURCE_SETS
    assert bound.files == scope.ADMITTED_SOURCE_SETS[bound.mode]
    assert bound.receipt()['physical_restart_identity_changed'] is False
    with bound.phase1(n_columns=3, n_levels=4):
        if bound.candidate:
            from gpuwm.core.rrtmg_lw import hex_busy_current_event_v2
            from gpuwm.ingest.ozone_pressure_device import current_backend_provider
            assert current_backend_provider() is bound.provider
            assert hex_busy_current_event_v2() is not None
            context = copy_context()
            result = []
            worker = threading.Thread(target=lambda: context.run(lambda: result.append(
                (hex_busy_current_event_v2(), current_backend_provider()))))
            worker.start()
            worker.join()
            assert result == [(None, None)]


@pytest.mark.parametrize('changed', list(scope.DEVICE_SOURCES))
def test_mixed_source_set_refuses_before_imports(tmp_path, changed):
    root = runtime_root()
    for name in scope.DEVICE_SOURCES:
        source = root / name
        if source.exists():
            target = tmp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    target = tmp_path / changed
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(target.read_bytes() + b'\n' if target.exists() else b'not a provider')
    with pytest.raises((RuntimeError, FileNotFoundError)):
        scope.bind_arwen_helpers_v2(tmp_path)


def test_cuda_source_escape_refuses(tmp_path):
    root = runtime_root()
    for name in scope.PUBLISHED_SOURCES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name == scope.SW_CUDA:
            target.symlink_to(root / name)
        else:
            shutil.copyfile(root / name, target)
    with pytest.raises(RuntimeError, match='escapes its engine tree'):
        scope.bind_arwen_helpers_v2(tmp_path)


def test_published_engine_rows_bind_the_whole_set_and_run_the_published_path():
    """A published-engine row (data/profiles/engine-helper-sets.v1.json).

    THE BREAKAGE: the 2.8.8 line changed one non-arithmetic line of
    rrtmg_lw.py and every hex forecast on it refused here as an unknown
    source set.  Each row must bind the baseline's six sources exactly, and
    must select the published path: the candidate path enters
    ``hex_busy_phase1_event_v2``, which only the 2.8.7 overlay carries.
    """

    rows = [mode for mode in scope.PUBLISHED_MODES if mode != "published-2.8.7"]
    assert "published-2.8.8-line" in rows
    for mode in rows:
        files = scope.ADMITTED_SOURCE_SETS[mode]
        assert set(files) == set(scope.PUBLISHED_SOURCES)
        assert mode in scope.RUNTIME_DESCRIPTIONS
    moved = {name for name, digest in scope.ADMITTED_SOURCE_SETS["published-2.8.8-line"].items()
             if digest != scope.PUBLISHED_SOURCES[name]}
    assert moved == {"gpuwm/core/rrtmg_lw.py"}
    assert "sw32-ozone-device" not in scope.PUBLISHED_MODES
