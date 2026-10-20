"""Witness an actual forecast attempt, retaining failures without a zero grade."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('deck', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    deck = json.loads(args.deck.read_text())
    checkout = Path(__file__).resolve().parents[3]
    if subprocess.run(['git', '-C', str(checkout), 'diff', '--quiet', 'HEAD']).returncode:
        raise ValueError('forecast proof requires committed source')
    commit = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    sources = [*sorted((checkout/'tree/src/hexcore/mpasdiag').glob('*.py')),
               *sorted(Path(__file__).parent.glob('*.py'))]
    source_hashes = {str(path.relative_to(checkout)):
                     hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    artifact_path = os.environ.get('HEX_MPASDIAG_CUDA_ARTIFACT')
    artifact = None if not artifact_path else dict(path=artifact_path,
        sha256=hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest(),
        manifest=json.loads(Path(artifact_path+'.json').read_text()))
    companion_path = os.environ.get('HEX_MPASDIAG_NATIVE_COMPANIONS')
    companions = None if not companion_path else dict(path=companion_path,
        sha256=hashlib.sha256(Path(companion_path).read_bytes()).hexdigest(),
        manifest=json.loads(Path(companion_path).with_suffix('.json').read_text()))
    if deck['process_count'] != 1:
        raise ValueError('this actual-door attempt uses one process')
    bindings = {}
    for name, item in deck['inputs'].items():
        path = Path(item['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('changed pinned input '+name)
        bindings[name] = str(path)
    for name, item in deck.get('auxiliary_inputs', {}).items():
        if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('changed pinned auxiliary input '+name)
    for name, expected in deck['native_provider'].items():
        root = Path(deck['hex_command'][deck['hex_command'].index('--native-physics-library')+1]).parent
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('changed native provider '+name)
    bindings['out'] = str(args.output.resolve())
    command = [sys.executable, str(Path(__file__).with_name('forecast.py')),
               *(part.format_map(bindings) for part in deck['hex_command'][1:])]
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    log = args.receipt.with_suffix('.log')
    with log.open('w') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    forecast = args.output/'native-physics-receipt.json'
    state = json.loads(forecast.read_text()) if forecast.exists() else {}
    if subprocess.run(['git', '-C', str(checkout), 'diff', '--quiet', 'HEAD']).returncode:
        raise ValueError('source changed during forecast')
    if subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip() != commit:
        raise ValueError('commit changed during forecast')
    receipt = dict(schema='mpas-o2.actual-forecast-attempt/v1', process_count=1,
        model_execution=True, identity_checked=False, passed=False,
        completed=state.get('completed',False), returncode=result.returncode,
        requested_steps=deck['duration_steps'], steps_completed=state.get('steps_completed',0),
        diagnostics_calls=state.get('isobaric_calls',[]), diagnostics_backend='hex-cuda',
        complete_manager_binding=state.get('diagnostics_manager_complete', False), argv=command,
        run_host=platform.node(), source_sha256=source_hashes, cuda_artifact=artifact,
        native_companions=companions,
        same_input_referee=state.get('isobaric_same_input_referee'),
        output_boundary_comparison=state.get('output_boundary_comparison'),
        hex_commit=commit, clean_checkout=True, deck_path=str(args.deck.absolute()),
        output_directory=str(args.output.absolute()),
        deck_sha256=hashlib.sha256(args.deck.read_bytes()).hexdigest(),
        native=deck['native'], log=str(log),
        word_diff_per_stream_and_field=None,
        word_diff_unavailable_reason='No full native/hex stream comparison has run.',
        failure=log.read_text()[-6000:] if result.returncode else None)
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({key:receipt[key] for key in
                    ('completed','returncode','steps_completed','identity_checked')}),flush=True)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
