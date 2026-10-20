"""Reject incomplete proof receipts even if an earlier summary says equal."""
import copy
import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).parent/'rrfs_radiation_oracle/replay_validation.py'
_spec = importlib.util.spec_from_file_location('r9_replay_validation', _path)
validation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validation)


@pytest.fixture
def proof():
    # Two real-boundary-shaped fields exercise a scientific carrier and a clock.
    receipt = dict(schema='test-proof/v1', checkout_tip='clean-tip',
        native_manifest_sha256='native-bank', replay_source_sha256='replay-source',
        production_sha256='production', bitwise_equal=True, comparison_complete=True,
        arrays=2, words=4, scientific_output_words=3, preserved_words=1,
        different_words=0, cases=[{'case': 'sw-call'}], comparisons=[
            dict(case='sw-call', field='global.rthratensw_p', words=3,
                 different_words=0, bitwise_equal=True, comparison_complete=True),
            dict(case='sw-call', field='calendar.julday', words=1,
                 different_words=0, bitwise_equal=True, comparison_complete=True)])
    authority = dict(tip='clean-tip', schema='test-proof/v1', bank_sha='native-bank',
        replay_sha='replay-source', production_sha='production',
        expected={('sw-call', 'global.rthratensw_p'), ('sw-call', 'calendar.julday')})
    return receipt, authority


def test_complete_module_receipt(proof):
    receipt, authority = proof
    validation.validate_receipt(receipt, **authority)


@pytest.mark.parametrize('field,value', [
    ('checkout_tip', 'earlier-tip'), ('schema', 'another-schema'),
    ('native_manifest_sha256', 'another-bank'), ('replay_source_sha256', 'another-replay'),
    ('production_sha256', 'another-port'), ('bitwise_equal', False),
    ('comparison_complete', False), ('arrays', 1), ('words', 3),
    ('different_words', None), ('different_words', True),
    ('scientific_output_words', 0), ('preserved_words', 0)])
def test_inconsistent_summary_cannot_seal(proof, field, value):
    receipt, authority = proof
    receipt[field] = value
    with pytest.raises(ValueError):
        validation.validate_receipt(receipt, **authority)


@pytest.mark.parametrize('field,value', [
    ('different_words', 1), ('different_words', None), ('different_words', False),
    ('words', 0), ('words', 3.0), ('comparison_complete', False),
    ('bitwise_equal', False), ('schema_mismatch', ['shape'])])
def test_one_bad_record_cannot_be_hidden_by_equal_summary(proof, field, value):
    receipt, authority = proof
    receipt['comparisons'][0][field] = value
    with pytest.raises(ValueError):
        validation.validate_receipt(receipt, **authority)


def test_omitted_preserved_input_cannot_reduce_claim(proof):
    receipt, authority = proof
    receipt['comparisons'].pop()
    receipt.update(arrays=1, words=3, preserved_words=0)
    with pytest.raises(ValueError, match='every native POST'):
        validation.validate_receipt(receipt, **authority)


def test_duplicate_cannot_replace_missing_output(proof):
    receipt, authority = proof
    receipt['comparisons'][0] = copy.deepcopy(receipt['comparisons'][1])
    receipt.update(words=2, scientific_output_words=1)
    with pytest.raises(ValueError, match='Duplicate'):
        validation.validate_receipt(receipt, **authority)


def test_stale_case_cannot_be_relabelled(proof):
    receipt, authority = proof
    receipt['cases'] = [{'case': 'earlier-call'}]
    with pytest.raises(ValueError, match='case inventory'):
        validation.validate_receipt(receipt, **authority)


def test_native_inventory_includes_every_post():
    manifest = {'files': {'sw-call.scope.txt': 'scope',
        'sw-call.pre.a.bin': 'pre-a', 'sw-call.post.a.bin': 'post-a',
        'sw-call.pre.b.bin': 'pre-b', 'sw-call.post.b.bin': 'post-b'}}
    assert validation.inventory(manifest) == {('sw-call', 'a'), ('sw-call', 'b')}
    del manifest['files']['sw-call.post.b.bin']
    with pytest.raises(ValueError, match='PRE/POST'):
        validation.inventory(manifest)


def test_loaded_other_checkout_is_rejected(tmp_path):
    supplied, stale = tmp_path/'supplied.py', tmp_path/'stale.py'
    supplied.write_text('same production')
    stale.write_text('same production')
    entry = dict(path=str(stale), sha256=validation.digest(stale))
    with pytest.raises(ValueError, match='another source path'):
        validation.loaded_source(entry, supplied)
