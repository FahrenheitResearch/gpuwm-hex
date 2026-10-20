#!/usr/bin/env python3
"""Replay real driver orchestration and manager schedule scalars.

MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver.F:127-419;
mpas_atmphys_manager.F:137-832 at 1d6e370e. Callbacks below record invocation
only: they do not claim to replay the delegated schemes or initialization.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct
import sys


def replay(checkout, bank):
    sys.path.insert(0, str(checkout/'tree/src'))
    from hexcore.mpasphys.rrfs.driver_rrfs import (
        PhysicsDriver, PhysicsManager, SlotRegistry, resolve_namelist, specification)
    manifest = json.loads((bank/'manifest.json').read_text())
    if manifest['model_commit'] != specification()['model_commit'] or manifest['process_count'] != 1:
        raise ValueError('Real lifecycle producer identity mismatch')
    if manifest.get('schema') != 'r4.real-driver-lifecycle-bank/v1' or manifest.get('input_origin') != 'native-tap':
        raise ValueError('Lifecycle bank has no native capture provenance')
    for source in ('mpas_atmphys_manager.F','mpas_atmphys_driver.F','mpas_atmphys_interface.F'):
        key = 'src/core_atmosphere/physics/'+source
        if manifest['source_sha256'].get(key) != specification()['source_files'][key]:
            raise ValueError('Lifecycle fork source authority changed')
    producer = json.loads((bank/'producer.json').read_text())
    if (producer['native_sha256'] != manifest['native_binary_sha256']
            or producer['input_request_sha256'] != manifest['input_request_sha256']
            or producer['process_count'] != manifest['process_count']
            or any(run['exit'] != 0 for run in producer['runs'].values())):
        raise ValueError('Native lifecycle producer did not complete the bound real case')
    for name, digest in manifest['files'].items():
        if Path(name).name != name or hashlib.sha256((bank/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Real lifecycle member hash mismatch: '+name)
    passive = json.loads((bank/'passivity.json').read_text())
    if not passive['bitwise_equal'] or passive['different_words'] != 0:
        raise ValueError('Native lifecycle observer passivity was not proven')
    rows = [json.loads(line) for line in (bank/'trace.jsonl').read_text().splitlines()]
    pairs = {}; stack = []
    for row in rows:
        if any(len(bytes.fromhex(v)) != 4 for v in row['words'].values()):
            raise ValueError('Invalid scalar word width')
        if row['phase'] == 'pre':
            if row['id'] in pairs or row['depth'] != len(stack):
                raise ValueError('Duplicate or malformed native entry nesting')
            pairs[row['id']] = {'pre':row}; stack.append(row['id'])
        elif row['phase'] == 'post':
            if not stack or stack.pop() != row['id'] or row['symbol'] != pairs[row['id']]['pre']['symbol']:
                raise ValueError('Malformed native return nesting')
            pairs[row['id']]['post'] = row
        else:
            raise ValueError('Unknown native lifecycle phase')
    if stack or any(set(pair) != {'pre','post'} for pair in pairs.values()):
        raise ValueError('Incomplete native lifecycle observation')
    def named(name):
        return [pair for pair in pairs.values() if pair['pre']['symbol'].split('_MOD_')[1] == name]
    trackers = named('physics_timetracker')
    if len(trackers) != 1:
        raise ValueError('This stated real lifecycle packet must contain one complete timestep')
    config = resolve_namelist((bank/'namelist.atmosphere').read_text())
    dt = struct.unpack('<f', bytes.fromhex(trackers[0]['pre']['words']['argument.dt']))[0]
    manager = PhysicsManager(config, dt, streams=manifest['stream_intervals'])
    comparisons = []
    def compare(row, key, value):
        actual = struct.pack('<i' if type(value) in (bool,int) else '<f', value).hex()
        expected = row['words']['__mpas_atmphys_vars_MOD_'+key]
        comparisons.append({'native_call':row['id'],'field':key,'native_hex':expected,
                            'hex_hex':actual,'equal':actual == expected})
    initialized = named('physics_run_init')
    if len(initialized) != 1:
        raise ValueError('Expected one native initialization')
    for key, value in manager.native_timestep_state().items():
        compare(initialized[0]['post'], key, value)
    class InvocationRecorder:
        def invoke(self, operation, call):
            return None
    registry = SlotRegistry()
    for slot, scheme in config.selectors.items():
        if config.active(slot) and not config.native_embedded(slot):
            registry.register(slot, scheme, config.backends[slot], lambda call: None)
    driver = PhysicsDriver(config, manager, registry, InvocationRecorder())
    driver.initialize({})
    native_initialization = [pair['pre']['symbol'].split('_MOD_')[1] for pair in pairs.values()
        if pair['pre']['depth'] == 0 and pair['pre']['symbol'].split('_MOD_')[1]
            in ('physics_tables_init','physics_registry_init','physics_run_init','physics_init')]
    order = [{'phase':'initialization delegation', 'native':native_initialization,
              'hex':[call['operation'] for call in driver.calls]}]
    translate = {slot:specification()['slots'][slot]['native_entries'][scheme].lower()
                 for slot,scheme in config.selectors.items() if config.active(slot)}
    flags_map = {'radiation_lw':'l_radtlw','radiation_sw':'l_radtsw','convection':'l_conv',
                 'cam_abs':'doabsems','cam_restart':'l_camlw','rain_bucket':'l_acrain',
                 'radiation_bucket':'l_acradt','diagnostics':'l_diags'}
    for pair in trackers:
        entry = pair['pre']
        step = struct.unpack('<i', bytes.fromhex(entry['words']['argument.itimestep']))[0]
        time = struct.unpack('<f', bytes.fromhex(entry['words']['argument.xtime_s']))[0]
        flags = driver.begin_step(itimestep=step, time_seconds=time, pools={})
        for key, value in asdict(flags).items():
            if key in flags_map:
                compare(pair['post'], flags_map[key], value)
        native_driver = [p for p in named('physics_driver')
                         if p['pre']['words']['argument.itimestep'] == entry['words']['argument.itimestep']]
        if len(native_driver) != 1:
            raise ValueError('Native manager/driver pairing mismatch')
        body = native_driver[0]['pre']
        expected = [p['pre']['symbol'].split('_MOD_')[1] for p in pairs.values()
                    if p['pre']['depth'] == body['depth']+1
                    and body['id'] < p['pre']['id']
                    and rows.index(p['pre']) < rows.index(native_driver[0]['post'])]
        actual = [translate.get(call['operation'],call['operation']).lower()
                  for call in driver.calls]
        start = actual.index('allocate_forall_physics')
        order.append({'phase':'complete pre-dynamics physics body', 'native':expected,'hex':actual[start:]})
        expected_updates = [p['pre']['symbol'].split('_MOD_')[1] for p in pairs.values()
                    if p['pre']['depth'] == entry['depth']+1 and entry['id'] < p['pre']['id']
                    and rows.index(p['pre']) < rows.index(pair['post'])]
        update_calls = [op for op in actual[:start] if op != 'physics_timetracker_state']
        order.append({'phase':'manager update delegation','native':expected_updates,'hex':update_calls})
        driver.physics_get_tend(rk_step=1,dynamics_substep=1,pools={})
        driver.finish_step(pools={})
    # This real one-step packet has a single successful diagnostics write/reset.
    # The stream owner supplies the successful alarm; it is not inferred from l_diags.
    resets = len(named('atm_reset_diagnostics'))
    for _ in range(resets):
        driver.after_output(pools={},diagnostics_alarm=True,itimestep=step,time_seconds=time+dt)
    driver.finalize({})
    actual_tail = [translate.get(c['operation'],c['operation']).lower() for c in driver.calls[start+len(expected):]]
    actual_tail = [c for c in actual_tail if c != 'post_transport_prepare']
    expected_tail = [p['pre']['symbol'].split('_MOD_')[1] for p in pairs.values()
        if p['pre']['depth'] == 0 and p['pre']['id'] > body['id']
        and p['pre']['symbol'].split('_MOD_')[1] not in ('physics_timetracker','physics_driver')]
    order.append({'phase':'coupling, microphysics, successful-output reset and finalization delegation',
                  'native':expected_tail,'hex':actual_tail})
    equal = all(c['equal'] for c in comparisons) and all(c['native'] == c['hex'] for c in order)
    return {'schema':'r4.real-driver-lifecycle-replay/v1','bitwise_equal':equal,
        'scope':'real scalar schedule words and complete orchestration; delegated kernel/private state and unimplemented scalar context are observed, not replayed',
        'observed_calls':len(pairs),'observed_scalar_words':sum(len(r['words']) for r in rows),
        'compared_schedule_words':len(comparisons),'different_words':sum(not c['equal'] for c in comparisons),
        'comparisons':comparisons,'order':order,'delegated_native_reset_calls':resets,
        'native_passivity_words':passive['words'],'manifest_sha256':hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('checkout','bank','receipt'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    result = replay(args.checkout.resolve(),args.bank.resolve())
    args.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('comparisons','order')}))
    raise SystemExit(0 if result['bitwise_equal'] else 1)
