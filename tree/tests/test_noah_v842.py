"""Integration checks against the compiled, pristine MPAS v8.4.2 Noah oracle."""
from pathlib import Path
import os
import subprocess
import sys
import pytest
import site_support

ROOT = Path(__file__).resolve().parents[1]
ORACLE = Path(__file__).with_name('noah_v842_oracle')


def _run(script, *args, cwd=ROOT):
    subprocess.run([sys.executable, str(ORACLE / script), *args], cwd=cwd, check=True)


@pytest.mark.skipif(not (ORACLE/'libnoah_oracle.so').exists(), reason='build strict MPAS standalone oracle on assigned CPU host')
def test_complete_noah_and_initialization_against_native():
    _run('check_columns.py')


@pytest.mark.skipif(not (ORACLE/'table_decode').exists(), reason='build Rust Noah table decoder on assigned CPU host')
def test_rust_table_decoding_against_native():
    _run('check_tables.py')


@pytest.mark.skipif(not (ORACLE/'libnoah_oracle.so').exists(), reason='build strict MPAS field conversion oracle on assigned CPU host')
def test_mpas_from_to_field_conversions_against_native():
    _run('check_io.py')


@pytest.mark.skipif(not (ORACLE/'libnoah_oracle.so').exists(), reason='build strict MPAS standalone oracle on assigned CPU host')
def test_native_fatal_conditions():
    _run('check_errors.py')


@pytest.mark.skipif(not (ORACLE/'libnoah_oracle.so').exists(), reason='build strict supplemental oracle on assigned CPU host')
def test_remaining_noah_boundary_outcomes():
    _run('check_helpers.py')


@pytest.mark.skipif(not (ORACLE/'libnoah_oracle.so').exists(), reason='build strict subnormal MPAS oracle on assigned CPU host')
def test_native_subnormal_evaporation():
    _run('check_subnormal.py')


@pytest.mark.skipif(not (ORACLE/'libnoah_host.so').exists() or not (ORACLE/'compare_noah_call').exists(),
                    reason='build whole-module host bridge and Rust word comparator')
def test_native_snowpatch_melt_compiler_fold_regression(tmp_path):
    _run('check_snowpatch.py', '--receipt', str(tmp_path/'snowpatch.json'))


def test_immutable_mpas_source_citations(tmp_path):
    site_support.mpas_source()
    _run('check_citations.py', cwd=tmp_path)


@pytest.mark.skipif(not (ORACLE/'libnoah_oracle.so').exists(), reason='build strict MPAS standalone oracle on assigned CPU host')
def test_native_complete_driver_and_conversion_pipeline():
    _run('check_pipeline.py')


@pytest.mark.skipif(not (ORACLE/'dump_transport').exists(), reason='build Rust f1 dump transport on assigned CPU host')
def test_f1_typed_transport_and_inactive_halos():
    _run('check_dump_transport.py')


@pytest.mark.skipif(not (ORACLE/'dump_transport').exists(), reason='build Rust f1 dump transport on assigned CPU host')
def test_f1_native_module_table_snapshots():
    _run('check_dump_transport.py', '--state')


def _written_proof_checker():
    import runpy
    return runpy.run_path(str(ORACLE / 'written_comparison.py'))['written_failures']


def _written_proof_inventory():
    written = dict(stream='output', field='temperature', step=72,
                   status='compared', compared_words=100, differing_words=0)
    missing = dict(stream='output', field='native_only', step=72,
                   status='field missing in hex', compared_words=100, differing_words=0)
    stored = dict(files=[dict(file='output.final.nc', errors=['variable set differs: native_only'])],
                  words=100, differing_words=0)
    return [written, missing], stored


def test_written_proof_accepts_only_original_approved_omissions():
    fields, stored = _written_proof_inventory()
    assert _written_proof_checker()(fields, fields, stored) == []


@pytest.mark.parametrize('defect', ['numeric', 'stored', 'missing_growth', 'lost_written',
                                    'new_field', 'duplicate', 'missing_sample', 'new_stored_omission'])
def test_written_proof_rejects_nonzero_words_or_changed_coverage(defect):
    import copy
    baseline, stored = _written_proof_inventory()
    fields = copy.deepcopy(baseline)
    if defect == 'numeric':
        fields[0]['differing_words'] = 1
    elif defect == 'stored':
        stored['differing_words'] = 1
    elif defect == 'missing_growth':
        fields[1]['compared_words'] += 1
    elif defect == 'lost_written':
        fields[0]['status'] = 'field missing in hex'
    elif defect == 'new_field':
        fields[1]['field'] = 'unapproved'
    elif defect == 'duplicate':
        fields.append(copy.deepcopy(fields[0]))
    elif defect == 'missing_sample':
        fields.pop()
    else:
        stored['files'][0]['errors'].append('variable set differs: unapproved')
    assert _written_proof_checker()(fields, baseline, stored)


def test_single_deck_subset_never_permits_new_missing_words():
    import copy
    baseline, stored = _written_proof_inventory()
    extra = copy.deepcopy(baseline[0])
    extra['field'] = 'suite_only'
    assert _written_proof_checker()(baseline, [*baseline, extra], stored, subset=True) == []
    fields = copy.deepcopy(baseline)
    fields[0]['status'] = 'field missing in hex'
    assert _written_proof_checker()(fields, [*baseline, extra], stored, subset=True)
