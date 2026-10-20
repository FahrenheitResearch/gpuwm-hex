"""Search native whole-driver forcing words; traces are diagnostic, never an oracle.

MPAS cu_ntiedtke.F90:1430-1479,1652-1689. Python inspects integer trace
metadata only. Fortran constructs all columns and computes every expected word.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instrument(source, destination):
    original = source.read_text()
    marker = '          kup(jl,jk)  = (atop1*kup(jl,jk+1) + atop2) / abot'
    assert original.count(marker) == 2
    chunks = original.split(marker)
    result = chunks[0]
    for phase, tail in enumerate(chunks[1:], 1):
        result += marker + f'''
          if (klon == 18) then
            write(*,'(a,4(1x,i0),3(1x,z8.8),1x,i0)') &
              'NTENERGY', {phase}, {'klev' if phase == 1 else 'levels'}, jl, jk, transfer(kup(jl,jk),0), &
              transfer(kup(jl,jk+1),0), transfer(plu(jl,jk),0), klab(jl,jk+1)
          endif
''' + tail
    destination.write_text(result)
    destination.with_suffix('.provenance.json').write_text(json.dumps(dict(
        authority=str(source), authority_sha256=sha(source),
        diagnostic_source_sha256=sha(destination), output_statements_only=True,
        acceptance_authority=False), indent=2)+'\n')


def search(args):
    args.work.mkdir(parents=True, exist_ok=True)
    cache = {}
    invocations = 0
    def run(variant, word):
        nonlocal invocations
        key = (variant, word)
        if key not in cache:
            context = dict(hfx_word='42c80000', lapse_word='3c03126f',
                           dt_word='42700000', dz_word=args.geometry_word,
                           qv_word=args.vapor_word)
            context[args.parameter + '_word'] = f'{word:08x}'
            inputs = [context[name] for name in
                      ('hfx_word', 'lapse_word', 'dt_word', 'dz_word', 'qv_word')]
            proc = subprocess.run([str(args.native.resolve()), '0', '1', '0',
                                   str(variant), *inputs], cwd=args.work,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, check=True)
            rows = {}
            for line in proc.stdout.splitlines():
                if not line.startswith('NTENERGY '):
                    continue
                _, phase, departure, col, level, current, previous, liquid, label = line.split()
                rows.setdefault((int(phase), int(departure), int(col), int(level)), dict(
                    current=int(current,16), previous=int(previous,16),
                    liquid=int(liquid,16), previous_label=int(label)))
            cache[key] = rows
            invocations += 1
        return cache[key]
    brackets = []
    selected = set()
    lower, upper = {'hfx': (0, 0x49742400),
                    'lapse': (0x3b03126f, 0x3c23d70a),
                    'qv': (0, 0x41800000)}[args.parameter]
    for variant in (0, 2, 5, 1000, 1200):
        low = run(variant, lower)
        high = run(variant, upper)
        selected.update(((variant, lower), (variant, upper)))
        field = args.energy_word
        eligible = [key for key, row in high.items()
                    if row['previous_label'] == 1
                    and (row['liquid'] == 0 if args.parcel_state == 'unsaturated'
                         else 0 < row['liquid'] < 0x7f800000)
                    and 0 < row[field] < 0x7f800000
                    and (key not in low or low[key][field] >= 0x80000000
                         or low[key][field] == 0)]
        # Spread searches across native departure levels and columns in both
        # parcel phases. This selects input probes, never plume state values.
        keys = []
        for phase in ((args.phase,) if args.phase else (1, 2)):
            candidates = sorted(key for key in eligible if key[0] == phase)
            count = min(len(candidates), args.keys_per_phase)
            if count:
                keys.extend(candidates[(2*i+1)*len(candidates)//(2*count)]
                            for i in range(count))
        for key in keys:
            left, right = lower, upper
            while right-left > 1:
                mid = (left+right)//2
                row = run(variant, mid).get(key)
                if row is not None and row[field] < 0x80000000:
                    right = mid
                else:
                    left = mid
            endpoints = []
            for word in range(max(0,left-args.neighbors), right+args.neighbors+1):
                row = run(variant,word).get(key)
                selected.add((variant,word))
                endpoints.append(dict(input_word=f'{word:08x}', trace=row))
            brackets.append(dict(variant=variant, phase=key[0], departure=key[1],
                                 column=key[2], level=key[3], endpoints=endpoints))
        print(json.dumps(dict(variant=variant, invocations=invocations,
                              brackets=len(brackets))), flush=True)
    summary = dict(schema='ntiedtke-native-energy-input-search/v1',
                   diagnostic_only=True, acceptance_proven=False,
                   native_process_count=1, binary_sha256=sha(args.native),
                   parameter=args.parameter, phase=args.phase,
                   geometry_word=args.geometry_word, vapor_word=args.vapor_word,
                   parcel_state=args.parcel_state,energy_word=args.energy_word,
                   keys_per_phase=args.keys_per_phase, neighbors=args.neighbors,
                   invocations=invocations, complete=True, brackets=brackets,
                   exact_zero_observed=any(e['trace'] and
                       e['trace']['current'] & 0x7fffffff == 0
                       for b in brackets for e in b['endpoints']),
                   target_zero_observed=any(e['trace'] and
                       e['trace'][field] & 0x7fffffff == 0
                       for b in brackets for e in b['endpoints']),
                   settings=[dict(variant=v, hfx_word='42c80000',
                                  lapse_word='3c03126f', dt_word='42700000',
                                  dz_word=args.geometry_word, qv_word=args.vapor_word)
                             | {args.parameter + '_word': f'{w:08x}'}
                             for v,w in sorted(selected)])
    args.receipt.write_text(json.dumps(summary,indent=2)+'\n')


def replay(args):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import test_mpas_ntiedtke_v842 as tests
    selected = json.loads(args.settings.read_text())
    settings = selected['settings']
    if not selected.get('complete') or not settings:
        raise ValueError('a finished native input search is required')
    args.fixtures.mkdir(parents=True,exist_ok=True)
    args.receipts.mkdir(parents=True,exist_ok=True)
    if args.host_library:
        from ntiedtke_native.host import HostTiedtke
        launcher = HostTiedtke(args.host_library)
    elif args.repeat:
        launcher = None
    else:
        from hexcore.mpasphys.ntiedtke_v842 import NewTiedtkeV842
        launcher = NewTiedtkeV842()
    names = ('nt-levels.csv','nt-surface.csv','nt-isolation.csv')
    expected = json.loads(args.repeat.read_text()) if args.repeat else None
    if expected and (not expected['complete'] or not expected['passed']
                     or len(expected['runs']) != len(settings)):
        raise ValueError('native repeat requires the complete passing replay')
    stop = len(settings) if args.stop is None else args.stop
    if not 0 <= args.start < stop <= len(settings):
        raise ValueError('replay range is outside the declared settings')
    requested = stop - args.start
    rows = []
    for index in range(args.start, stop):
        setting = settings[index]
        root = args.fixtures / str(index)
        if args.native:
            root.mkdir(exist_ok=False)
            optional = [('lapse_word', '3c03126f'), ('dt_word', '42700000'),
                        ('dz_word', '3f800000'), ('qv_word', '3f800000'),
                        ('dz_first', '1'), ('dz_last', '0'),
                        ('interface_word', '00000000'),
                        ('interface_first', '2'), ('interface_last', '2')]
            last = max((i for i, (name, _) in enumerate(optional)
                        if name in setting), default=-1)
            inputs = [str(setting.get(name, default))
                      for name, default in optional[:last+1]]
            with (root/'native.log').open('w') as log:
                subprocess.run([str(args.native.resolve()),str(setting.get('initflag',0)),'1','0',
                                str(setting['variant']),setting['hfx_word'],
                                *inputs],
                               cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
            (root/'settings.json').write_text(json.dumps(dict(
                grav=9.80616,initflag=0,stepcu=1,flag_mode=0,
                native_host=socket.gethostname()) | setting)+'\n')
        hashes = {name:sha(root/name) for name in names}
        row = dict(index=index,setting=setting,native_csv_sha256=hashes)
        if args.repeat:
            previous = expected['runs'][index]
            row.update(identical=previous['setting']==setting and
                       previous['native_csv_sha256']==hashes)
        else:
            path = args.receipts / f'{index}.json'
            result = tests.replay(root,path,host_library=args.host_library,
                                  launcher=launcher)
            row.update(receipt=str(path),sha256=sha(path),
                       differing_words=result['differing_words'],
                       compared_words=sum(a['words'] for r in result['results']
                                          for a in r['arrays'].values()))
        rows.append(row)
        summary = dict(schema='ntiedtke-full-driver-energy-boundary-replay/v1',
                       settings_sha256=sha(args.settings),
                       execution=('strict-native-repeat' if args.repeat else
                                  'host-glibc-witness' if args.host_library else 'hex-cuda'),
                       native_process_count=1,requested=requested,
                       replay_host=socket.gethostname(),
                       settings_range=[args.start, stop],
                       complete=len(rows)==requested,runs=rows,
                       native_binary_sha256=sha(args.native) if args.native else None,
                       passed=len(rows)==requested and all(
                           r.get('identical',r.get('differing_words')==0) for r in rows))
        (args.receipts/'matrix.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        if (row.get('differing_words',0) or row.get('identical') is False) and not args.keep_going:
            return 1
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    generate = modes.add_parser('instrument')
    generate.add_argument('source', type=Path)
    generate.add_argument('destination', type=Path)
    seek = modes.add_parser('search')
    seek.add_argument('--parameter',choices=('hfx','lapse','qv'),default='hfx')
    seek.add_argument('--geometry-word', default='3f800000')
    seek.add_argument('--vapor-word', default='3f800000')
    seek.add_argument('--phase', type=int, choices=(1, 2))
    seek.add_argument('--parcel-state',choices=('unsaturated','cloud-base'),default='unsaturated')
    seek.add_argument('--energy-word',choices=('current','previous'),default='current')
    seek.add_argument('--keys-per-phase',type=int,default=1)
    seek.add_argument('--neighbors',type=int,default=2)
    for name in ('native','work','receipt'):
        seek.add_argument('--'+name, type=Path, required=True)
    compare = modes.add_parser('replay')
    for name in ('settings','fixtures','receipts'):
        compare.add_argument('--'+name,type=Path,required=True)
    for name in ('native','host-library','repeat'):
        compare.add_argument('--'+name,type=Path)
    compare.add_argument('--keep-going', action='store_true',
                         help='compare the entire matrix while retaining every failure')
    compare.add_argument('--start', type=int, default=0)
    compare.add_argument('--stop', type=int)
    args = parser.parse_args()
    if args.mode == 'instrument':
        instrument(args.source,args.destination)
    elif args.mode == 'search':
        if args.keys_per_phase < 1 or args.neighbors < 0:
            parser.error('search needs positive key count and nonnegative neighbors')
        search(args)
    else:
        if args.repeat and not args.native:
            parser.error('strict repeat needs a native executable')
        if args.native and not (args.repeat or args.host_library):
            parser.error('native generation needs the host witness')
        raise SystemExit(replay(args))
