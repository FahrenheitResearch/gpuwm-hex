import importlib.util
from pathlib import Path
import sys
import pytest

TOOLS = Path(__file__).parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('operational_compare_endpoints', TOOLS / 'operational_compare_endpoints.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def record(clock):
    value = {'dtype': '<f4', 'shape': [1], 'sha256': 'a' * 64, 'bytes_sha256': 'b' * 64}
    return {'atmosphere': {'model_time_seconds': clock,
             'state': {'fields': {name: dict(value) for name in compare.STATE_FIELDS}},
             'saved_diagnostics': {'fields': {name: dict(value) for name in compare.SIDECAR_FIELDS}}},
            'backend': {'arrays': {'carrier': dict(value)}, 'scalars': {}}}


def test_sparse_endpoint_evidence_does_not_need_or_claim_intermediate_states():
    compare.validate_endpoints({0: record(0), 720: record(3600)}, 720, 5)


@pytest.mark.parametrize('records', [{0: record(0)}, {0: record(0), 719: record(3595)},
                                   {0: record(0), 720: record(3605)}])
def test_truncation_wrong_endpoint_and_wrong_clock_refuse(records):
    with pytest.raises(ValueError):
        compare.validate_endpoints(records, 720, 5)


def test_partial_fingerprint_census_refuses_even_when_remaining_hashes_match():
    endpoint = record(3600)
    endpoint['atmosphere']['state']['fields'].pop(next(iter(compare.STATE_FIELDS)))
    with pytest.raises(ValueError, match='census'):
        compare.validate_endpoints({0: record(0), 720: endpoint}, 720, 5)


def test_invalid_backend_digest_refuses():
    endpoint = record(3600)
    endpoint['backend']['arrays']['carrier']['sha256'] = 'not-a-digest'
    with pytest.raises(ValueError, match='sha256'):
        compare.validate_endpoints({0: record(0), 720: endpoint}, 720, 5)
