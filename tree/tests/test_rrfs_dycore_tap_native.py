"""Verify the complete dycore native capture interface without numeric edits."""
import difflib
import os
from pathlib import Path

import site_support
from rrfs_native.prepare_dycore_tap import ROUTINES, SOURCE, generate
from rrfs_native.native_source import read_pinned_source


def test_complete_native_overlay_only_inserts_observers(tmp_path):
    root = site_support.rrfs_source('RRFS_MPAS_SOURCE')
    manifest = generate(root, tmp_path)
    assert tuple(b['key'].removeprefix('dycore__') for b in manifest['boundaries']) == ROUTINES
    assert manifest['native_arithmetic_edits'] == []
    original = read_pinned_source(root, root / SOURCE)
    instrumented = (tmp_path / SOURCE).read_text()
    changes = list(difflib.unified_diff(original.splitlines(), instrumented.splitlines()))
    assert not any(line.startswith('-') and not line.startswith('---') for line in changes)
    for boundary in manifest['boundaries']:
        labels = {f['label'] for f in boundary['fields']}
        assert {'argument.' + arg for arg in boundary['arguments'] if arg not in boundary['non_numeric_arguments']} <= labels
        assert {'module.' + n for n in boundary['private_module_fields']} <= labels
        for label in labels:
            for phase in ('pre', 'post'):
                assert instrumented.count(f"'{boundary['key']}','{phase}','{label}',") == 1
    dynamics = next(b for b in manifest['boundaries'] if b['key'] == 'dycore__atm_compute_dyn_tend_work')
    assert {'qtot', 'dpdz', 'tend_ru_physics', 'tend_rtheta_physics', 'tend_rho_physics'} <= set(dynamics['private_module_fields'])
    assert {'nvertlevels', 'num_scalars', 'maxedges', 'maxedges2'} <= set(dynamics['dimensions'])
    mono = next(b for b in manifest['boundaries'] if b['key'].endswith('atm_advance_scalars_mono_work'))
    assert mono['non_numeric_arguments'] == ['block', 'exchange_halo_group', 'state']
    assert {'argument.' + n for n in ('scalars_old','scalars_new','scalar_tend','scalar_old','scalar_new','s_max','s_min','wdtn','scale_arr','flux_arr','flux_tmp','flux_upwind_tmp')} <= {f['label'] for f in mono['fields']}


def test_complete_cold_core_overlay_preserves_original_source(tmp_path):
    from rrfs_native.prepare_core_tap import SOURCE, ROUTINES, generate
    root = site_support.rrfs_source('RRFS_MPAS_SOURCE')
    manifest = generate(root, tmp_path)
    original = read_pinned_source(root, root / SOURCE)
    instrumented = (tmp_path / SOURCE).read_text()
    changes = list(difflib.unified_diff(original.splitlines(), instrumented.splitlines()))
    assert not any(line.startswith('-') and not line.startswith('---') for line in changes)
    assert manifest['native_arithmetic_edits'] == []
    assert tuple(b['key'].removeprefix('core__') for b in manifest['boundaries']) == ROUTINES
    for boundary in manifest['boundaries']:
        for field in boundary['fields']:
            for phase in ('pre', 'post'):
                assert instrumented.count(f"'{boundary['key']}','{phase}','{field['label']}',") == 1
    assert 'oracle_set_context(r3_tap_ids,0,dminfo%my_proc_id)' in instrumented
