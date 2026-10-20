"""Measure and replay the complete strict MPAS driver for exceptional timesteps.

cu_ntiedtke.F90:528,684-687,715-716,319-320. Fortran constructs every input
column and every expected word. Every strict invocation is repeated; a
bounds-checked binary must reproduce the same complete CSV sets. No words
are excluded and this column measurement never counts as a forecast proof.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_mpas_ntiedtke_v842 import replay
from ntiedtke_native.host import HostTiedtke
from hexcore.mpasphys.ntiedtke_v842 import source_receipt

WORDS = ('42700000', 'c2700000', '00000000', '80000000', '00000001',
         '80000001', '7f800000', 'ff800000', '7fc00000', 'ffc00000')
CSVS = ('nt-levels.csv', 'nt-surface.csv', 'nt-isolation.csv')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    launcher = HostTiedtke(args.library)
    rows = []
    for variant in args.variant:
        for word in WORDS:
            index = len(rows)
            setting = dict(variant=variant, hfx_word='42c80000',
                           lapse_word='3c03126f', dt_word=word)
            invocations = {}
            for mode, binary in (('strict-a', args.strict),
                                 ('strict-b', args.strict), ('bounds', args.bounds)):
                fixture = args.output/'fixtures'/mode/str(index)
                fixture.mkdir(parents=True)
                with (fixture/'native.log').open('w') as log:
                    result = subprocess.run([str(binary.resolve()), '0', '1', '0',
                                             str(variant), setting['hfx_word'],
                                             setting['lapse_word'], word], cwd=fixture,
                                            stdout=log, stderr=subprocess.STDOUT,
                                            timeout=30, check=True)
                invocations[mode] = dict(exit_code=result.returncode,
                                         csv_sha256={n: digest(fixture/n) for n in CSVS})
            expected = invocations['strict-a']['csv_sha256']
            if any(r['csv_sha256'] != expected for r in invocations.values()):
                raise ValueError('strict repetition or bounds output varies; no word exclusion: '
                                 + str(setting))
            fixture = args.output/'fixtures/strict-a'/str(index)
            (fixture/'settings.json').write_text(json.dumps(dict(
                grav=9.80616, initflag=0, stepcu=1, flag_mode=0,
                native_host=socket.gethostname()) | setting)+'\n')
            receipt = args.output/(str(index)+'.json')
            result = replay(fixture, receipt, host_library=args.library, launcher=launcher)
            rows.append(dict(index=index, setting=setting, native_invocations=invocations,
                             receipt=receipt.name, receipt_sha256=digest(receipt),
                             differing_words=result['differing_words'],
                             compared_words=sum(a['words'] for g in result['results']
                                                for a in g['arrays'].values())))
            summary = dict(schema='mpas-ntiedtke-whole-timestep-matrix/v1',
                           complete=len(rows)==len(args.variant)*len(WORDS),
                           passed=all(r['differing_words']==0 for r in rows),
                           native_process_count=1, native_host=socket.gethostname(),
                           strict_sha256=digest(args.strict), bounds_sha256=digest(args.bounds),
                           host_library_sha256=digest(args.library), source=source_receipt(),
                           no_words_excluded=True, device_executed=False,
                           forecast_proof=False, runs=rows)
            (args.output/'matrix.json').write_text(json.dumps(summary, indent=2)+'\n')
            print(json.dumps({k: rows[-1][k] for k in
                              ('index', 'setting', 'differing_words', 'compared_words')}), flush=True)
    return 0 if summary['complete'] and summary['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('strict', 'bounds', 'library', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--variant', type=int, action='append', required=True)
    raise SystemExit(main(parser.parse_args()))
