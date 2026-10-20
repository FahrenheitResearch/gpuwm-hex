#!/usr/bin/env python3
"""Compare complete forecasts with sparse endpoint fingerprints, honestly scoped."""
import argparse
import json
from pathlib import Path

from operational_proof import summarize
from compare_samehost_forecasts_v841 import _fingerprint, _FORCING_FIELDS
from hexcore.cuda_dualrun import STATE_FIELDS, SIDECAR_FIELDS
from hexcore.drivers.samehost_checks_v841 import read_boundary_jsonl, compare_boundary_files


def validate_endpoints(records, steps, dt):
    if set(records) != {0, steps}:
        raise ValueError('endpoint proof requires exactly steps zero and the completed forecast endpoint')
    for step, record in records.items():
        atmosphere = record.get('atmosphere') or {}
        if atmosphere.get('model_time_seconds') != step * dt:
            raise ValueError('endpoint clock does not match the complete forecast')
        for group, names in [('state', STATE_FIELDS), ('saved_diagnostics', SIDECAR_FIELDS)]:
            fields = (atmosphere.get(group) or {}).get('fields')
            if not isinstance(fields, dict) or set(fields) != set(names):
                raise ValueError('complete atmosphere endpoint fingerprint census is absent')
            for name, value in fields.items():
                _fingerprint(value, name, atmosphere=True)
        backend = record.get('backend') or {}
        if not isinstance(backend.get('arrays'), dict) or not backend['arrays'] or not isinstance(backend.get('scalars'), dict):
            raise ValueError('complete backend endpoint fingerprints are absent')
        for name, value in backend['arrays'].items():
            _fingerprint(value, name)


def compare(reference, candidate):
    arms = []
    for run in [reference, candidate]:
        manifest = json.loads((run / 'manifest.json').read_text())
        proof = summarize(run, manifest)
        if proof['status'] != 'PASS':
            raise ValueError(f'{run.name}: complete forecast and sanity proof failed: {proof["errors"]}')
        measurement = json.loads((run / 'measurement.json').read_text())
        for key in ['driver_completed', 'run_identity_eligible', 'protected_bytes_unchanged', 'ordinary_door_restored']:
            if measurement.get(key) is not True:
                raise ValueError(f'{run.name}: {key} is not true')
        validate_endpoints(read_boundary_jsonl(run / 'fc/boundary-fingerprints.jsonl'),
                           proof['expected_steps'], manifest['dt_seconds'])
        forcing = measurement.get('final_gf_tendency_hashes')
        if not isinstance(forcing, dict) or set(forcing) != _FORCING_FIELDS:
            raise ValueError('both final forcing fingerprints are required')
        for name, value in forcing.items():
            _fingerprint(value, name)
        arms.append((proof, json.loads((run / 'case.json').read_text()), forcing))
    for key in ['hours', 'dt_seconds', 'arithmetic']:
        if arms[0][0]['manifest'][key] != arms[1][0]['manifest'][key]:
            raise ValueError(f'forecast controls differ: {key}')
    for key in ['mesh', 'grid', 'static', 'init', 'lbc', 'start', 'init_source', 'forecast_flags', 'engine', 'cells']:
        if arms[0][1].get(key) != arms[1][1].get(key):
            raise ValueError(f'prepared case controls differ: {key}')
    boundaries = compare_boundary_files(reference / 'fc/boundary-fingerprints.jsonl',
                                        candidate / 'fc/boundary-fingerprints.jsonl', require_steps=2)
    histories = arms[0][0]['history_files'] == arms[1][0]['history_files']
    forcing = arms[0][2] == arms[1][2]
    passed = boundaries['status'] == 'PASS' and histories and forcing
    return {'schema': 'hex-operational-endpoint-identity/v1', 'status': 'PASS' if passed else 'FAIL',
            'expected_steps': arms[0][0]['expected_steps'], 'fingerprinted_steps': [0, arms[0][0]['expected_steps']],
            'scope': 'all endpoint atmosphere and persisted backend hashes with declared artifact exceptions; all history file bytes; final forcing',
            'all_step_state_identity_claim': False, 'all_step_health_complete': True,
            'boundary_comparison': boundaries, 'history_identity': histories, 'final_forcing_identity': forcing,
            'proofs': [arm[0] for arm in arms]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(args.reference, args.candidate)
    except Exception as error:
        result = {'schema': 'hex-operational-endpoint-identity/v1', 'status': 'FAIL',
                  'error': str(error), 'all_step_state_identity_claim': False}
    with args.receipt.open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(f'{result["status"]} endpoint/history/forcing identity: {args.receipt}')
    return 0 if result['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
