"""Restart across a move (A4): replay the first hours of a long run's move segment from a copy of the
carrier it started from, with the same member, operator and flags, and compare the per-frame restart
identity (SHA-256 of rho, rho_theta, rho_u, rho_w and every class's physics identity) with the run's own.

usage: replay_move.py --run RUN --segment N --carrier COPY.pkl --out NEWDIR [--hours 1] [--engine ENGINE]
       replay_move.py --compare --run RUN --segment N --out NEWDIR

The carrier a move consumes is deleted by the controller once the next segment closes, so copy
RUN/segment-(N-1)/boundary.pkl (and its .json) while segment N is still running. One card (the one CUDA
makes visible; --launcher prefixes a host's own card-queue command). Pass: every frame both runs wrote
has identical hashes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

TREE = Path(__file__).resolve().parents[2]


def identity_rows(fc: Path):
    path = fc / 'fullstep-lts-restart-identity.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
    return {round(float(r['model_seconds'])): r for r in rows}


def compare(run: Path, segment: int, out: Path) -> dict:
    ref = identity_rows(run / f'segment-{segment:03d}' / 'fc')
    new = identity_rows(out / 'fc')
    common = sorted(set(ref) & set(new))
    frames = []
    for s in common:
        a, b = dict(ref[s]), dict(new[s])
        for row in (a, b):
            row.pop('step', None)
        frames.append(dict(model_seconds=s, identical=json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)))
    result = dict(run=str(run), segment=segment, replay=str(out), frames_compared=len(frames),
                  identical=bool(frames) and all(f['identical'] for f in frames), frames=frames)
    (out / 'replay-identity.json').write_text(json.dumps(result, indent=1) + '\n')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--segment', type=int, required=True)
    p.add_argument('--carrier', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--hours', type=float, default=1.0)
    p.add_argument('--engine', type=Path)
    p.add_argument('--compare', action='store_true')
    p.add_argument('--launcher', default=None,
                   help='optional command that admits a GPU job on this host, given as one string; '
                        'the replay command follows it')
    a = p.parse_args(argv)
    if a.compare:
        print(json.dumps(compare(a.run, a.segment, a.out), indent=1))
        return
    seg = a.run / f'segment-{a.segment:03d}'
    record = json.loads((seg / 'command.json').read_text())
    state = json.loads((a.run / 'controller.json').read_text())
    member = state['members'][record['member']]
    a.out.mkdir(parents=True)
    control = a.out / 'control'
    control.mkdir()
    cmd = list(record['command'])
    # drop the /usr/bin/time and timeout wrappers; set the replay length and folders
    cmd = cmd[cmd.index('-m') - 1:]
    for flag, value in (('--hours', f'{a.hours:g}'), ('--out', str(a.out / 'fc')), ('--scratch', str(a.out / 'scratch'))):
        cmd[cmd.index(flag) + 1] = value
    if a.engine:
        if '--gpuwm-checkout' in cmd:
            cmd[cmd.index('--gpuwm-checkout') + 1] = str(a.engine)
        else:
            cmd += ['--gpuwm-checkout', str(a.engine)]
    engine = cmd[cmd.index('--gpuwm-checkout') + 1] if '--gpuwm-checkout' in cmd else None
    env = dict(os.environ, GPUWM_HEX_MOVING_SHOCK='1', GPUWM_WRF_EXACT='0', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', GPUWM_HEX_MESH_ROWS=member['mesh_rows'],
               GPUWM_HEX_MOVING_CONTROL=str(control), GPUWM_HEX_MOVING_HEIGHT_REMAP='absolute',
               GPUWM_HEX_MOVING_CHECKPOINT_IN=str(a.carrier),
               PYTHONPATH=os.pathsep.join([str(TREE / 'src')] + ([engine] if engine else [])))
    for key in ('GPUWM_HEX_MOVING_OPERATOR', 'GPUWM_HEX_MOVING_SOURCE_GRID', 'GPUWM_HEX_MOVING_ZONE',
                'GPUWM_HEX_MOVING_SHOCK_ZONE', 'GPUWM_HEX_MOVING_CHECKPOINT_OUT', 'GPUWM_HEX_MOVING_CHECKPOINT_WAIT'):
        env.pop(key, None)
    operator = record.get('operator')
    if operator and str(operator).endswith('.zone.npz'):
        env.update(GPUWM_HEX_MOVING_ZONE=str(operator), GPUWM_HEX_MOVING_SHOCK_ZONE=str(operator))
    elif operator:
        env.update(GPUWM_HEX_MOVING_OPERATOR=str(operator), GPUWM_HEX_MOVING_SOURCE_GRID=str(record['source_grid']))
    (a.out / 'replay-command.json').write_text(json.dumps(dict(command=cmd, operator=operator,
                                                               carrier=str(a.carrier)), indent=1) + '\n')
    wrapped = [*(a.launcher.split() if a.launcher else []), *cmd]
    with (a.out / 'forecast.log').open('w') as log:
        rc = subprocess.call(wrapped, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(TREE))
    result = compare(a.run, a.segment, a.out)
    result['rc'] = rc
    print(json.dumps(result, indent=1))
    sys.exit(0 if rc == 0 and result['identical'] else 1)


if __name__ == '__main__':
    main()
