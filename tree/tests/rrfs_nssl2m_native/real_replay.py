#!/usr/bin/env python3
"""Replay complete real NSSL argument records; Rust owns binary I/O/comparison.

Fork: src/core_atmosphere/physics/physics_wrf/module_mp_nssl_2mom.F:1995-3031
at 1d6e370e7b43a332074809f170532a216c2212fb. Only PRE feeds the numerical
module. Every captured POST word, including presence and halos, is compared.
Host-glibc evidence is explicitly distinct from the required CUDA evidence.
"""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np

from prepare import STATE, ENVIRONMENT, SURFACE, DIAGNOSTICS, load_target


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    decoder = Path(__file__).parents[2]/'tools/native_physics_reference/oracle/oracle_words.rs'
    bridge = Path(__file__).with_name('record_bridge.rs')
    source = decoder.read_text().replace('//!','//')+'\n'+bridge.read_text()
    (root/'bridge.rs').write_text(source)
    (root/'Makefile').write_text('all: bridge.so oracle_words\nbridge.so: bridge.rs\n\trustc --edition=2021 --crate-type cdylib -O $< -o $@\noracle_words: oracle_words.rs\n\trustc --edition=2021 -O $< -o $@\n')
    (root/'oracle_words.rs').write_bytes(decoder.read_bytes())
    (root/'sources.json').write_text(json.dumps(dict(decoder_sha256=sha(decoder),
        bridge_sha256=sha(bridge), combined_sha256=sha(root/'bridge.rs')),indent=2)+'\n')


class Records:
    def __init__(self, path):
        self.lib=ct.CDLL(str(path.resolve()))
        self.lib.nssl_record_open.argtypes=[ct.c_char_p,ct.POINTER(ct.c_int32),ct.POINTER(ct.POINTER(ct.c_uint32))]
        self.lib.nssl_record_open.restype=ct.c_void_p
        self.lib.nssl_record_close.argtypes=[ct.c_void_p]
        self.lib.nssl_record_post.argtypes=[ct.c_char_p,ct.c_char_p,ct.POINTER(ct.c_uint32),ct.c_size_t]

    def load(self, path):
        header=(ct.c_int32*21)(); payload=ct.POINTER(ct.c_uint32)()
        handle=self.lib.nssl_record_open(os.fsencode(path),header,ct.byref(payload))
        if not handle: raise ValueError('Rust native record decoding failed: '+str(path))
        try:
            shape=tuple(header[3:3+header[2]])
            size=int(np.prod(shape)) if shape else 1
            array=np.ctypeslib.as_array(payload,shape=(size,)).copy()
            return array.view('f4' if header[0]==1 else 'i4').reshape(shape,order='F'),tuple(header[8:8+header[2]])
        finally:
            self.lib.nssl_record_close(handle)

    def save(self, template, target, array):
        words=np.ascontiguousarray(array.ravel(order='F')).view('u4')
        result=self.lib.nssl_record_post(os.fsencode(template),os.fsencode(target),
            words.ctypes.data_as(ct.POINTER(ct.c_uint32)),words.size)
        if result: raise ValueError('Rust native record encoding failed: '+str(target))


# Independent reference: pristine fork mpas_atmphys_driver_microphysics.F:750-770.
# Keep the native call's argument binding independent of the CUDA port table.
NATIVE_PREPARED_ARGUMENTS = dict(
    th='th_p', qv='qv_p', qc='qc_p', qr='qr_p', qi='qi_p', qs='qs_p',
    qh='qg_p', qhl='qh_p', cn='nccn_p', ccw='nc_p', crw='nr_p', cci='ni_p',
    csw='ns_p', chw='ng_p', chl='nh_p', vhw='volg_p', vhl='volh_p',
    dn='rho_p', pii='pi_p', p='pres_p', dz='dz_p', w='w_p',
    rainnc='rainnc_p', rainncv='rainncv_p', snownc='snownc_p', snowncv='snowncv_p',
    grplnc='graupelnc_p', grplncv='graupelncv_p', sr='sr_p',
    rainprod='rainprod_p', evapprod='evapprod_p', re_cloud='recloud_p',
    re_ice='reice_p', re_snow='resnow_p', dbz='refl10cm_p',
    refl_diagnostic='refl10cm_1km_p')
NATIVE_SCALAR_ARGUMENTS = dict(n_microp='ntmul', l_diags='diagflag',
    has_reqc='has_reqc', has_reqi='has_reqi', has_reqs='has_reqs', dt_microp='dtp',
    kms='kms', kts='kts', kte='kte')


def replay_native_boundary(module, runtime, native, raw, lower, groups, keywords, nc, verify):
    """Real prepared storage through the owner adapter; fork microphysics.F:750-770.

    This starts after native preparation and ends before native postprocessing.
    It does not install the proposed production descriptor or execute the door.
    """
    from dataclasses import replace
    from types import SimpleNamespace
    from hexcore.mpasphys.device_pools_v842 import NativeDevicePoolViews
    from hexcore.mpasphys.driver_v842 import Backend, SlotCall
    cp = runtime.cp
    assert int(raw['its'].item()) == 1
    for argument, prepared in NATIVE_PREPARED_ARGUMENTS.items():
        assert lower[argument][0] == 1
        assert raw[argument].shape[-1] == 1
        native.put_variable(prepared, raw[argument], lower=lower[argument])
    for prepared, argument in NATIVE_SCALAR_ARGUMENTS.items():
        native.put_variable(prepared, raw[argument])
    native.put_field('diag_physics', 'do_diag_dbz', raw['diag_dbz'])
    verify('native-prepared', {argument: native.get_variable(prepared)[0]
        for argument, prepared in NATIVE_PREPARED_ARGUMENTS.items()})
    layout = SimpleNamespace(pools=('diag_physics',),
        configs={'config_nssl_moments': 'nssl2m'}, dimensions={'nCellsSolve': nc},
        fields=[dict(pool='diag_physics', name='do_diag_dbz',
                     pool_name='do_diag_dbz', dimensions=())])
    provider = NativeDevicePoolViews(SimpleNamespace(native=native, layout=layout), cp)
    provider.nssl_runtime = runtime
    call = SlotCall(slot='microphysics', scheme='mp_nssl2m', backend=Backend.HEX_CUDA,
        itimestep=keywords['nt'], time_seconds=0., dt=keywords['dt'],
        flags=None, pools=provider, options={})
    descriptor = dict(module='rrfs.mp_nssl2m', bind='bind_native', call='cuda_entry')
    binding = provider.begin(call, descriptor)
    assert binding['options'] == keywords, 'Native callback scalar binding differs from real arguments'
    packed_inputs = {}
    for group, names in groups.items():
        for index, argument in enumerate(names):
            packed = binding['arrays'][group]
            slab = packed[index] if group in ('state', 'environment', 'surface', 'diagnostics') else packed
            value = raw[argument].copy(order='F')
            if value.ndim == 3:
                kstart = int(raw['kts'].item()) - lower[argument][1]
                kstop = int(raw['kte'].item()) - lower[argument][1] + 1
                value[:nc, kstart:kstop, 0] = slab.get().T
            else:
                value[:nc, 0] = slab.get()
            packed_inputs[argument] = value
    verify('cuda-packed', packed_inputs)
    module.cuda_entry(replace(call, pools=binding))
    provider.commit()
    assert not provider.buffers
    for argument, prepared in NATIVE_PREPARED_ARGUMENTS.items():
        value, bounds = native.get_variable(prepared)
        assert bounds == lower[argument] and value.shape == raw[argument].shape
        raw[argument] = value
    for prepared, argument in NATIVE_SCALAR_ARGUMENTS.items():
        value, bounds = native.get_variable(prepared)
        assert value.shape == raw[argument].shape and bounds == lower[argument]
        raw[argument] = value
    raw['diag_dbz'] = native.get_field('diag_physics', 'do_diag_dbz')


def execute(a):
    capture=json.loads((a.capture/'summary.json').read_text())
    assert capture['passed'] and capture['passivity_all_output_words_equal']
    assert capture['complete_entry_return_pairs'] and capture['process_count']==1
    members_path=a.capture/'capture-members.json'
    assert sha(members_path)==capture['capture_member_manifest_sha256']
    members=json.loads(members_path.read_text())
    dump=a.capture/'captures'
    assert {p.name for p in dump.iterdir() if p.is_file()}==set(members)
    for name,digest in members.items():
        assert sha(dump/name)==digest,'Capture changed: '+name
    assert capture['observer_spec_sha256']==sha(a.observer_spec)
    spec=json.loads(a.observer_spec.read_text())
    module=load_target()
    assert spec['source_sha256']['src/core_atmosphere/physics/physics_wrf/module_mp_nssl_2mom.F']==module.SOURCE_SHA256
    checkout=Path(__file__).parents[3]
    dirty=subprocess.check_output(['git','-C',str(checkout),'status','--porcelain','--','tree/src','tree/tests','tree/tools'],text=True)
    assert not dirty,'Replay requires committed code'
    a.out.mkdir(parents=True,exist_ok=False)
    result=dict(schema='r16-nssl2m.real-module-replay/v1',backend=a.backend,
        host=socket.gethostname(),source_commit=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip(),
        runner_sha256=sha(Path(__file__)),translation_sha256=sha(Path(module.__file__)),
        capture_summary_sha256=sha(a.capture/'summary.json'),capture_member_manifest_sha256=sha(members_path),
        observer_forecast_passivity_proven=True,scope='All captured arguments of every NSSL call in the stated short native run',
        forecast_acceptance=False,hidden_state_complete=False,global_cell_ids_known=False,rows=[],passed=False)
    trace=None
    native_boundary=None
    try:
        if a.backend=='hex-cuda':
            if a.resources is None: raise ValueError('An authenticated shared device math resource packet is required')
            result['resources_sha256']=sha(a.resources)
            packet=json.loads(a.resources.read_text())
            result['math_packet_path']=packet['math']['path']
            result['math_sha256']=packet['math']['sha256']
            result['packet_certification']=packet.get('certification')
            if a.math_header is not None:
                assert sha(a.math_header)==packet['math']['sha256'],'Staged math header changed'
                packet['math']['path']=str(a.math_header)
            result['math_loaded_path']=packet['math']['path']
            import cupy as cp
            runtime=module.load_resources(packet,cp.cuda.Device())
            if a.cuda_native_boundary_library is not None:
                from hexcore.mpasphys.native_backend import NativePhysics
                native_boundary = NativePhysics(a.cuda_native_boundary_library, test_only=True, pin_set='rrfs')
                result['native_boundary'] = dict(
                    library_sha256=sha(native_boundary.path),
                    abi_sha256=sha(native_boundary.path.with_name('abi.json')),
                    build_receipt_sha256=sha(native_boundary.path.with_name('build-receipt.json')),
                    adapter_sha256=sha(checkout/'tree/src/hexcore/mpasphys/device_pools_v842.py'),
                    transaction='NativePhysics PRE -> NativeDevicePoolViews.begin -> bind_native -> cuda_entry -> commit -> NativePhysics POST',
                    prepared_arrays_per_call=len(NATIVE_PREPARED_ARGUMENTS),
                    production_registration_tested=False, native_driver_prefix_suffix_tested=False)
        elif a.backend=='native-fortran':
            from replay import Native
            runtime=Native(a.native_library)
            result['native_library_sha256']=sha(a.native_library)
            result['scope']='Standalone pristine native Fortran on every authenticated real call in the stated capture; supporting passivity/coverage evidence, not CUDA acceptance'
        else:
            from replay import Host
            if a.math_trace_library:
                trace=ct.CDLL(str(a.math_trace_library.resolve()))
                trace.nssl_trace_reset()
                trace.nssl_trace_flush.argtypes=[ct.c_char_p]
                result['math_trace_library_sha256']=sha(a.math_trace_library)
            runtime=Host(a.host_library)
            result['host_library_sha256']=sha(a.host_library)
        records=Records(a.bridge)
        calls=[json.loads(line) for line in (dump/'rank0-calls.jsonl').read_text().splitlines()]
        entries=[call for call in calls if call['phase']=='pre']
        for call in entries:
            prefix=call['prefix'];raw={};lower={}
            for name in members:
                marker=prefix+'.pre.'
                if name.startswith(marker) and name.endswith('.bin'):
                    leaf=name[len(marker):-4];raw[leaf],lower[leaf]=records.load(dump/name)
            scalar=lambda name:raw[name].item()
            assert scalar('ntcnt')==1 and scalar('lastloop')==1
            assert not any(scalar(name+'.present') for name in ('zrw','zhw','zhl'))
            its,ite,kts,kte,jts,jte=(int(scalar(n)) for n in ('its','ite','kts','kte','jts','jte'))
            assert jts==jte and kts==1
            nc,nz=ite-its+1,kte-kts+1
            def slices(name):
                lo=lower[name]
                if raw[name].ndim==3:return (slice(its-lo[0],ite-lo[0]+1),slice(kts-lo[1],kte-lo[1]+1),jts-lo[2])
                if raw[name].ndim==2:return (slice(its-lo[0],ite-lo[0]+1),jts-lo[1])
                raise ValueError('Native array rank unsupported: '+name)
            groups=dict(state=STATE,environment=ENVIRONMENT,surface=SURFACE,diagnostics=DIAGNOSTICS,
                        refl1km=('refl_diagnostic',),rainprod=('rainprod',),evapprod=('evapprod',))
            data={}
            for group,names in groups.items():
                slabs=[raw[n][slices(n)].T for n in names]
                data[group]=np.ascontiguousarray(np.stack(slabs) if group in ('state','environment','surface','diagnostics') else slabs[0])
            keywords=dict(nt=int(scalar('itimestep')),nmicro=int(scalar('ntmul')),
                diagnose=int(scalar('diagflag')),dbz_due=int(scalar('diag_dbz')),
                reqc=int(scalar('has_reqc')),reqi=int(scalar('has_reqi')),reqs=int(scalar('has_reqs')),dt=float(scalar('dtp')))
            if native_boundary is not None:
                def verify(stage, candidates):
                    for leaf, value in candidates.items():
                        template = dump/(prefix+'.pre.'+leaf+'.bin')
                        target = a.out/('boundary-'+leaf+'.bin')
                        records.save(template, target, value)
                        compared = subprocess.run([str(a.words.resolve()), 'compare',
                            str(template.resolve()), str(target.resolve())], capture_output=True, text=True)
                        target.unlink()
                        assert compared.returncode == 0, stage+': '+leaf+': '+compared.stdout+compared.stderr
                replay_native_boundary(module, runtime, native_boundary, raw, lower, groups, keywords, nc, verify)
            elif a.backend=='hex-cuda':
                device={n:cp.asarray(v) for n,v in data.items()}
                errors=runtime.run(device,**keywords)
                assert not bool(cp.any(errors).get()),'Native STOP reached by CUDA replay'
                data={n:v.get() for n,v in device.items()}
            else:
                runtime.run(data,**keywords)
            for group,names in (groups.items() if native_boundary is None else ()):
                for i,name in enumerate(names):
                    slab=data[group][i] if group in ('state','environment','surface','diagnostics') else data[group]
                    raw[name][slices(name)]=slab.T
            arrays={}
            for name in sorted(members):
                marker=prefix+'.post.'
                if not name.startswith(marker) or not name.endswith('.bin'):continue
                leaf=name[len(marker):-4]
                target=a.out/name
                records.save(dump/(prefix+'.pre.'+leaf+'.bin'),target,raw[leaf])
                compare=subprocess.run([str(a.words.resolve()),'compare',str((dump/name).resolve()),str(target.resolve())],capture_output=True,text=True)
                assert compare.returncode in (0,1),compare.stderr
                arrays[leaf]=json.loads(compare.stdout)
                target.unlink()
            result['rows'].append(dict(prefix=prefix,columns=nc,levels=nz,arguments=keywords,arrays=arrays,
                compared_words=sum(v['words'] for v in arrays.values()),
                comparison_complete=bool(arrays) and all(v['different_words'] is not None for v in arrays.values()),
                differing_words=sum(v['different_words'] for v in arrays.values()) if arrays and all(v['different_words'] is not None for v in arrays.values()) else None,
                passed=bool(arrays) and all(v['bitwise_equal'] and v.get('comparison_complete',True) for v in arrays.values())))
        assert len(result['rows'])==capture['observed_calls'] and result['rows']
        assert 2*sum(len(row['arrays']) for row in result['rows'])==capture['captured_argument_arrays']
        result['passed']=all(row['passed'] for row in result['rows'])
    except Exception as error:
        result['error']=type(error).__name__+': '+str(error)
    if native_boundary is not None:
        native_boundary.close()
    if trace is not None:
        trace_out=a.out/'math-trace'
        assert trace.nssl_trace_flush(os.fsencode(trace_out))==0
        result['math_trace']=json.loads((trace_out/'inventory.json').read_text())
        result['math_trace_inventory_sha256']=sha(trace_out/'inventory.json')
    (a.out/'replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(backend=result['backend'],passed=result['passed'],calls=len(result['rows']),
        compared_words=sum(row['compared_words'] for row in result['rows']),
        error=result.get('error'))))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',type=Path)
    parser.add_argument('--retire-build',type=Path)
    for name in ('capture','observer-spec','bridge','words','host-library','native-library','resources','math-header','math-trace-library','cuda-native-boundary-library','out'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--backend',choices=('hex-cuda','host-glibc','native-fortran'),default='hex-cuda')
    args=parser.parse_args()
    if args.prepare:prepare(args.prepare)
    elif args.retire_build:
        import shutil
        path=args.retire_build.resolve()
        assert path.parent==Path(os.environ['TMPDIR']).resolve()
        assert path.name.startswith('real-cuda-replay-')
        shutil.rmtree(path,ignore_errors=True)
    else:
        assert all(getattr(args,name) is not None for name in ('capture','observer_spec','bridge','words','out'))
        raise SystemExit(execute(args))
