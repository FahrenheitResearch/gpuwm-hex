#!/usr/bin/env python3
"""Replay real coupling, number carriers and lifecycle from a committed checkout.

MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_todynamics.F:65-550
at 1d6e370e. This entry point does not certify uncaptured driver boundaries.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


def run(checkout, bank, out, library, lifecycle_bank, carrier_bank, interface_bank, control_bank, micro_interface_bank, output_bank, surface_bank, pbl_bank, convection_bank, lsm_bank, lw_bank, sw_bank, init_bank):
    sys.path.insert(0, str(checkout / 'tree/tests'))
    sys.path.insert(0, str(checkout / 'tree/src'))
    from rrfs_driver_dump_replay import replay
    manifest = json.loads((bank / 'manifest.json').read_text())
    passive = json.loads((bank / 'passivity.json').read_text())
    if (not manifest.get('observer_passivity_proven') or not passive.get('bitwise_equal')
            or passive.get('different_words') != 0
            or hashlib.sha256((bank/'passivity.json').read_bytes()).hexdigest()
                != manifest.get('passivity_receipt_sha256')):
        raise ValueError('Measured native observer passivity is missing or changed')
    if out.exists():
        if out.is_symlink():
            raise ValueError('Replay scratch must not be a symlink')
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rows = {}
    for i, prefix in enumerate(sorted(manifest['captures'])):
        rows[prefix] = replay(bank/'dump', prefix, out/str(i), library, bank/'manifest.json')
    from rrfs_driver_lifecycle_replay import replay as lifecycle_replay
    from rrfs_driver_carrier_replay import replay as carrier_replay
    lifecycle = lifecycle_replay(checkout, lifecycle_bank)
    carrier_manifest = json.loads((carrier_bank/'manifest.json').read_text())
    carriers = {prefix:carrier_replay(carrier_bank/'dump',prefix,out/'carriers'/str(i),
        library,carrier_bank/'manifest.json') for i,prefix in enumerate(sorted(carrier_manifest['captures']))}
    from rrfs_driver_interface_replay import replay as interface_replay
    interface_manifest = json.loads((interface_bank/'manifest.json').read_text())
    interfaces = {prefix:interface_replay(interface_bank/'dump',prefix,out/'interface'/str(i),
        library,interface_bank/'manifest.json') for i,prefix in enumerate(sorted(interface_manifest['captures']))}
    from rrfs_driver_control_replay import replay as control_replay
    control_manifest = json.loads((control_bank/'manifest.json').read_text())
    controls = {prefix:control_replay(control_bank/'dump',prefix,out/'control'/str(i),
        library,control_bank/'manifest.json') for i,prefix in enumerate(sorted(control_manifest['captures']))}
    from rrfs_driver_micro_interface_replay import replay as micro_interface_replay
    micro_manifest = json.loads((micro_interface_bank/'manifest.json').read_text())
    micro_interfaces = {prefix:micro_interface_replay(micro_interface_bank/'dump',prefix,out/'micro-interface'/str(i),
        library,micro_interface_bank/'manifest.json') for i,prefix in enumerate(sorted(micro_manifest['captures']))}
    from rrfs_driver_output_replay import replay as output_replay
    output_manifest = json.loads((output_bank/'manifest.json').read_text())
    physics_outputs = {prefix:output_replay(output_bank/'dump',prefix,out/'physics-output'/str(i),
        library,output_bank/'manifest.json') for i,prefix in enumerate(sorted(output_manifest['captures']))}
    from rrfs_driver_surface_replay import replay as surface_replay
    surface_manifest=json.loads((surface_bank/'manifest.json').read_text())
    surfaces={prefix:surface_replay(surface_bank/'dump',prefix,out/'surface'/str(i),
        library,surface_bank/'manifest.json') for i,prefix in enumerate(sorted(surface_manifest['captures']))}
    from rrfs_driver_pbl_replay import replay as pbl_replay
    pbl_manifest=json.loads((pbl_bank/'manifest.json').read_text())
    pbls={prefix:pbl_replay(pbl_bank/'dump',prefix,out/'pbl'/str(i),
        library,pbl_bank/'manifest.json') for i,prefix in enumerate(sorted(pbl_manifest['captures']))}
    from rrfs_driver_convection_replay import replay as convection_replay
    convection_manifest=json.loads((convection_bank/'manifest.json').read_text())
    convections={prefix:convection_replay(convection_bank/'dump',prefix,out/'convection'/str(i),
        library,convection_bank/'manifest.json') for i,prefix in enumerate(sorted(convection_manifest['captures']))}
    from rrfs_driver_lsm_replay import replay as lsm_replay
    lsm_manifest=json.loads((lsm_bank/'manifest.json').read_text())
    lsms={prefix:lsm_replay(lsm_bank/'dump',prefix,out/'lsm'/str(i),
        library,lsm_bank/'manifest.json') for i,prefix in enumerate(sorted(lsm_manifest['captures']))}
    from rrfs_driver_lw_replay import replay as lw_replay
    lw_manifest=json.loads((lw_bank/'manifest.json').read_text())
    lws={prefix:lw_replay(lw_bank/'dump',prefix,out/'lw'/str(i),
        library,lw_bank/'manifest.json') for i,prefix in enumerate(sorted(lw_manifest['captures']))}
    from rrfs_driver_sw_replay import replay as sw_replay
    sw_manifest=json.loads((sw_bank/'manifest.json').read_text())
    sws={prefix:sw_replay(sw_bank/'dump',prefix,out/'sw'/str(i),
        library,sw_bank/'manifest.json') for i,prefix in enumerate(sorted(sw_manifest['captures']))}
    from rrfs_driver_init_replay import replay as init_replay
    init_manifest=json.loads((init_bank/'manifest.json').read_text())
    inits={prefix:init_replay(init_bank/'dump',prefix,out/'init'/str(i),
        library,init_bank/'manifest.json') for i,prefix in enumerate(sorted(init_manifest['captures']))}
    outputs = (*rows.values(),*carriers.values(),*interfaces.values(),*controls.values(),*micro_interfaces.values(),*physics_outputs.values(),*surfaces.values(),*pbls.values(),*convections.values(),*lsms.values(),*lws.values(),*sws.values(),*inits.values())
    capture_hosts = {}
    bank_paths = {'coupling':bank, 'lifecycle':lifecycle_bank, 'carriers':carrier_bank,
                  'preparation':interface_bank, 'control':control_bank,
                  'microphysics_handoffs':micro_interface_bank, 'output':output_bank, 'surface_handoffs':surface_bank, 'pbl_handoffs':pbl_bank, 'convection_handoffs':convection_bank, 'lsm_handoffs':lsm_bank, 'longwave_handoffs':lw_bank, 'shortwave_handoffs':sw_bank, 'numerical_init_preparation':init_bank}
    for phase, path in bank_paths.items():
        authority = json.loads((path/'passivity.json').read_text())
        capture_hosts[phase] = authority['host']
    equal = bool(rows) and bool(carriers) and bool(interfaces) and bool(controls) and bool(micro_interfaces) and bool(physics_outputs) and bool(surfaces) and bool(pbls) and bool(convections) and bool(lsms) and bool(lws) and bool(sws) and bool(inits) and lifecycle['bitwise_equal'] and all(
        row['bitwise_equal'] for row in outputs)
    return {'schema':'r4.supplied-checkout-driver-replay/v14',
        'scope':'real whole coupling, complete MPAS_to_physics, full control check, both complete microphysics, surface-layer, PBL, convection, LSM, longwave and shortwave handoffs and physics output preparation/reset, independent number carriers and complete driver-owned numerical initialization plus lifecycle/schedule orchestration; delegated initialization/private state not replayed',
        'cases':len(rows), 'output_records':sum(r['output_records'] for r in outputs),
        'words':sum(r['words'] for r in outputs)+lifecycle['compared_schedule_words'],
        'different_words':(sum(r['different_words'] for r in outputs)+lifecycle['different_words']
                            if all(r['different_words'] is not None for r in outputs) else None),
        'bitwise_equal':equal, 'process_count':1, 'coupling_capture_host':passive['host'],
        'native_capture_hosts':sorted(set(capture_hosts.values())),
        'native_capture_host_by_phase':capture_hosts,
        'bank_manifest_sha256':{phase:hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest()
                                for phase,path in bank_paths.items()},
        'manifest_sha256':hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest(),
        'observer_passivity':{'words':passive['words'],'different_words':passive['different_words'],
                              'scope':passive['scope']},
        'code_sha256':{str(p.relative_to(checkout)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [checkout/'tree/tests/rrfs_driver_init_replay.py',
                                checkout/'tree/tests/rrfs_driver_sw_replay.py',
                                checkout/'tree/src/hexcore/mpasphys/rrtmg_sw_v842.py',
                                checkout/'tree/tests/rrfs_driver_lw_replay.py',
                                checkout/'tree/src/hexcore/mpasphys/rrtmg_lw_v842.py',
                                checkout/'tree/tests/rrfs_driver_lsm_replay.py',
                                checkout/'tree/tests/rrfs_driver_convection_replay.py',
                                checkout/'tree/tests/rrfs_driver_pbl_replay.py',
                                checkout/'tree/tests/rrfs_driver_surface_replay.py',
                                checkout/'tree/tests/rrfs_driver_output_replay.py',
                                checkout/'tree/src/hexcore/mpasphys/diagnostics_v842.py',
                                checkout/'tree/tests/rrfs_driver_micro_interface_replay.py',
                                checkout/'tree/src/hexcore/mpasphys/kessler_v842.py',
                                checkout/'tree/src/hexcore/native_powf.py',
                                checkout/'tree/tests/rrfs_driver_control_replay.py',
                                checkout/'tree/tests/rrfs_driver_interface_replay.py',
                                checkout/'tree/src/hexcore/mpasphys/math_v842.py',
                                checkout/'tree/tests/rrfs_driver_lifecycle_replay.py',
                                checkout/'tree/tests/rrfs_driver_carrier_replay.py',
                                checkout/'tree/src/hexcore/cuda_physics_prep_v841.py',
                                checkout/'tree/src/hexcore/mpasphys/rrfs/driver_rrfs.py',
                                checkout/'tree/src/hexcore/cuda_physics_v841.py',
                                checkout/'tree/tests/rrfs_driver_dump_replay.py',
                                checkout/'tree/tests/rrfs_driver_dump_bridge.rs']},
        'inits':inits,'sws':sws,'lws':lws,'lsms':lsms,'replays':rows,'lifecycle':lifecycle,'carriers':carriers,'interfaces':interfaces,'controls':controls,'micro_interfaces':micro_interfaces,'physics_outputs':physics_outputs,'surfaces':surfaces,'pbls':pbls,'convections':convections}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('checkout','bank','out','library','receipt','lifecycle-bank','carrier-bank','interface-bank','control-bank','micro-interface-bank','output-bank','surface-bank','pbl-bank','convection-bank','lsm-bank','lw-bank','sw-bank','init-bank'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.checkout.resolve(), args.bank.resolve(), args.out.resolve(), args.library.resolve(),
                 args.lifecycle_bank.resolve(), args.carrier_bank.resolve(), args.interface_bank.resolve(), args.control_bank.resolve(), args.micro_interface_bank.resolve(), args.output_bank.resolve(), args.surface_bank.resolve(), args.pbl_bank.resolve(), args.convection_bank.resolve(), args.lsm_bank.resolve(), args.lw_bank.resolve(), args.sw_bank.resolve(), args.init_bank.resolve())
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('replays','code_sha256','lifecycle','carriers','interfaces','controls','micro_interfaces','physics_outputs','surfaces','pbls','convections','lsms','lws','sws','inits')}))
    raise SystemExit(0 if result['bitwise_equal'] else 1)
