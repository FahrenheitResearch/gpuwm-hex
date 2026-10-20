#!/usr/bin/env python3
"""Replay complete r1 MYJ captures using production launchers and Rust record I/O.

MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:141-782
and module_sf_myjsfc.F:76-366 at 1d6e370e. This is supporting column evidence.
"""
from pathlib import Path
import ctypes as ct
import hashlib
import json
import os
import sys
import numpy as np
from hexcore.mpasphys.rrfs import myj


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    decoder = Path(__file__).parents[1] / 'tools/native_physics_reference/oracle/oracle_words.rs'
    bridge = Path(__file__).with_suffix('.rs').with_name('rrfs_myj_dump_bridge.rs')
    source = decoder.read_text().replace('//!', '//') + '\n' + bridge.read_text()
    (root / 'bridge.rs').write_text(source)
    (root / 'Makefile').write_text('all: bridge.so\nbridge.so: bridge.rs\n\trustc --edition=2021 --crate-type cdylib -O $< -o $@\n')
    (root / 'source.json').write_text(json.dumps({
        'decoder_sha256': hashlib.sha256(decoder.read_bytes()).hexdigest(),
        'bridge_sha256': hashlib.sha256(bridge.read_bytes()).hexdigest(),
        'generated_sha256': hashlib.sha256(source.encode()).hexdigest()}, indent=2)+'\n')


class Records:
    def __init__(self, library):
        self.lib = ct.CDLL(str(library.resolve()))
        self.lib.myj_record_open.argtypes = [ct.c_char_p, ct.POINTER(ct.c_int32), ct.POINTER(ct.POINTER(ct.c_uint32))]
        self.lib.myj_record_open.restype = ct.c_void_p
        self.lib.myj_record_close.argtypes = [ct.c_void_p]
        self.lib.myj_record_post.argtypes = [ct.c_char_p, ct.c_char_p, ct.POINTER(ct.c_uint32), ct.c_size_t]

    def load(self, path):
        header = (ct.c_int32*21)()
        payload = ct.POINTER(ct.c_uint32)()
        handle = self.lib.myj_record_open(os.fsencode(path), header, ct.byref(payload))
        if not handle:
            raise ValueError(f'Rust oracle decode failed: {path}')
        try:
            shape = tuple(header[3:3+header[2]])
            size = int(np.prod(shape)) if shape else 1
            array = np.ctypeslib.as_array(payload, shape=(size,)).copy()
            array = array.view('f4' if header[0]==1 else 'i4').reshape(shape, order='F')
            return array, tuple(header[8:8+header[2]])
        finally:
            self.lib.myj_record_close(handle)

    def save(self, template, target, array):
        words = np.ascontiguousarray(array.ravel(order='F')).view('u4')
        status = self.lib.myj_record_post(os.fsencode(template), os.fsencode(target),
                                         words.ctypes.data_as(ct.POINTER(ct.c_uint32)), words.size)
        if status:
            raise ValueError(f'Rust oracle encode failed: {target}')


def replay(input_prefix, output_prefix, library):
    import cupy as cp
    records = Records(library)
    root, prefix = input_prefix.parent, input_prefix.name
    manifest = json.loads((root/'manifest.json').read_text())
    site_key = manifest['captures'][prefix]['site']
    site = next(s for s in manifest['sites'] if s['site']==site_key)
    assert site['scheme'] in ('myjpbl','myjsfc')
    source_hash = manifest['source_sha256']
    assert source_hash['src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F']=='45075a6e2622d3f4ecf555c16307689a4e8097f376e7386ebe1e14f14e530738'
    assert source_hash['src/core_atmosphere/physics/physics_wrf/module_sf_myjsfc.F']=='44f3d6eec777b5064ad2fc9995556d878b0fca90d4f98287fa96faafa6242d29'
    raw, lower = {}, {}
    for path in root.glob(prefix+'.pre.*.bin'):
        name = path.name[len(prefix+'.pre.'):-4]
        # Framework allocation-entry records accompany real driver captures.
        # They are not scheme arguments or outputs; keep them in the sealed
        # dump, while decoding every argument and captured private-state word.
        if name.startswith('allocation.'):
            continue
        raw[name], lower[name] = records.load(path)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    # Pristine external captures expose the native initialized SAVE packet.
    # Authenticate the actual generated CUDA lookup resources against it,
    # instead of merely returning unchanged private PRE words as POST.
    if (manifest.get('schema') == 'r15-myj.pristine-module-records/v1'
            and 'state.module_sf_myjsfc.psim1' in raw):
        tables, controls = myj.device_tables(cp.cuda.Device().id)
        table_names = ('psim1', 'psih1', 'psim2', 'psih2')
        control_names = ('dzeta1', 'dzeta2', 'ztmin1', 'ztmax1',
                         'ztmin2', 'ztmax2', 'fh01', 'fh02')
        for name, table in zip(table_names, tables):
            key = 'state.module_sf_myjsfc.'+name
            records.save(root/(prefix+'.pre.'+key+'.bin'),
                         output_prefix.parent/(output_prefix.name+'.resource.'+key+'.bin'),table.get())
        for name, value in zip(control_names, controls):
            key = 'state.module_sf_myjsfc.'+name
            records.save(root/(prefix+'.pre.'+key+'.bin'),
                         output_prefix.parent/(output_prefix.name+'.resource.'+key+'.bin'),np.asarray(value,dtype='f4'))
    scalar = lambda key: raw[key].item()
    slot = 'pbl' if site['scheme']=='myjpbl' else 'surface_layer'
    bindings = myj.PBL_NATIVE_ARRAYS if slot=='pbl' else myj.SURFACE_NATIVE_ARRAYS
    metadata = {key: scalar(key) for key in
                ('its','ite','kts','kte','jts','jte','ims','kms','jms')}
    if slot=='pbl':
        metadata.update(dt_pbl=scalar('dt'),stepbl=scalar('stepbl'))
    else:
        metadata.update(p1000mb=scalar('p1000mb'))
    device = {native: cp.asarray(raw[key],order='C')
              for key,native in bindings.items() if key in raw}
    native_lower = {native: lower[key] for key,native in bindings.items() if key in raw}
    # Raw module PRE follows the driver cold-TKE initialization. Surface
    # ITIMESTEP is the driver's initflag, not the model timestep number.
    myj.run_prepared(device,metadata,slot=slot,options={'config_frac_seaice':False},
                     itimestep=1,lower=native_lower,
                     initflag_override=2 if slot=='pbl' else scalar('itimestep'))
    for key,native in bindings.items():
        if native in device:
            raw[key] = device[native].get()
    # Emit exactly the sealed output record set, including untouched INOUT and
    # private state. Every payload derives from PRE plus actual CUDA results.
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for name in manifest['files']:
        marker = prefix+'.post.'
        if not name.startswith(marker):
            continue
        leaf = name[len(marker):-4]
        records.save(root/(prefix+'.pre.'+leaf+'.bin'),
                     output_prefix.parent/(output_prefix.name+'.post.'+leaf+'.bin'),raw[leaf])
    cp.cuda.get_current_stream().synchronize()


def replay_surface_initializer(capture, prefix, output, library):
    """Replay the complete pristine MYJSFCINIT entry, including generated SAVE.

    MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_sf_myjsfc.F:1066-1296
    at 1d6e370e. MPAS does not enable the NMM_CORE roughness branch. Full
    storage starts from PRE, including untouched Z0 and inactive words.
    """
    import cupy as cp
    records = Records(library)
    manifest = json.loads((capture/'manifest.json').read_text())
    assert manifest['schema']=='r15-myj.pristine-initializer-records/v1'
    assert manifest['captures'][prefix]['site']=='myjsfcinit__external_entry'
    raw, lower = {}, {}
    for path in capture.glob(prefix+'.pre.*.bin'):
        name = path.name[len(prefix+'.pre.'):-4]
        raw[name], lower[name] = records.load(path)
    scalar = lambda name: int(raw[name].item())
    its, itf = scalar('its'), min(scalar('ite'),scalar('ide')-1)
    jts, jtf, jte = scalar('jts'), min(scalar('jte'),scalar('jde')-1), scalar('jte')
    restart = bool(scalar('restart'))
    if its<=itf:
        for j in range(jts,jte+1):
            def row(name):
                value, lo = raw[name], lower[name]
                return value[its-lo[0]:itf-lo[0]+1,j-lo[1]]
            lowlyr = cp.asarray(np.ascontiguousarray(row('lowlyr')))
            ustar = cp.asarray(np.ascontiguousarray(row('ustar')))
            myj.init_surface(lowlyr,ustar,restart=restart)
            # Native LOWLYR ends at JTF; USTAR ends at JTE. A computed
            # temporary LOWLYR outside JTF never replaces native storage.
            if j<=jtf:
                lo=lower['lowlyr']
                raw['lowlyr'][its-lo[0]:itf-lo[0]+1,j-lo[1]]=lowlyr.get()
            lo=lower['ustar']
            raw['ustar'][its-lo[0]:itf-lo[0]+1,j-lo[1]]=ustar.get()
    # Force a fresh table launch so cold PRE tables cannot masquerade as
    # initialized resources. No expected POST payload enters the calculation.
    myj.device_tables.cache_clear()
    tables, controls = myj.device_tables(cp.cuda.Device().id)
    for name,value in zip(('psim1','psih1','psim2','psih2'),tables):
        raw['state.module_sf_myjsfc.'+name]=value.get()
    for name,value in zip(('dzeta1','dzeta2','ztmin1','ztmax1',
                           'ztmin2','ztmax2','fh01','fh02'),controls):
        raw['state.module_sf_myjsfc.'+name]=np.asarray(value,dtype='f4')
    output.mkdir(parents=True,exist_ok=True)
    for name in manifest['files']:
        marker=prefix+'.post.'
        if name.startswith(marker):
            leaf=name[len(marker):-4]
            records.save(capture/(prefix+'.pre.'+leaf+'.bin'),output/name,raw[leaf])
    cp.cuda.get_current_stream().synchronize()


def replay_surface_pair(capture, prefixes, output, library, *, poison_unused_sea_shared=False):
    """Replay the complete native main/sea pair, using only captured PRE data.

    MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F:1464-1514
    at 1d6e370e. ZNT/PBLH are the same native buffers in both calls. The
    caller authenticates their native POST-to-PRE continuity before launch.
    Native packing and sfclayer_to_MPAS accumulation are outside this proof.
    """
    import cupy as cp
    records = Records(library)
    manifest = json.loads((capture/'manifest.json').read_text())
    assert manifest['schema']=='r15-myj.pristine-module-records/v1'
    assert len(prefixes)==2
    packets = []
    for prefix in prefixes:
        assert manifest['captures'][prefix]['site']=='myjsfc__external_entry'
        raw, lower = {}, {}
        for path in capture.glob(prefix+'.pre.*.bin'):
            name = path.name[len(prefix+'.pre.'):-4]
            raw[name], lower[name] = records.load(path)
        packets.append((prefix,raw,lower))
    main, sea = packets
    bindings = myj.SURFACE_NATIVE_ARRAYS
    sea_bindings = dict(bindings, **myj.SURFACE_SEA_ARRAYS)
    prepared = {name: cp.asarray(main[1][key],order='C') for key,name in bindings.items()}
    native_lower = {name: main[2][key] for key,name in bindings.items()}
    for key,name in myj.SURFACE_SEA_ARRAYS.items():
        prepared[name] = cp.asarray(sea[1][key],order='C')
        native_lower[name] = sea[2][key]
    if poison_unused_sea_shared:
        # Full driver allocations include unused sea-side shared placeholders.
        # The production mapper must ignore them, including poisoned storage.
        for name in ('znt_sea','hpbl_sea'):
            prepared[name] = cp.full_like(prepared['znt_p'],cp.nan)
    metadata = {key: main[1][key].item() for key in
                ('its','ite','kts','kte','jts','jte','ims','kms','jms','p1000mb')}
    myj.run_prepared(prepared,metadata,slot='surface_layer',
                     options={'config_frac_seaice':True},itimestep=1,
                     lower=native_lower,initflag_override=main[1]['itimestep'].item())
    for packet,mapping in zip(packets,(bindings,sea_bindings)):
        prefix,raw,lower = packet
        for name in myj.SFC_STATE+myj.SFC_OUTPUTS:
            raw[name] = prepared[mapping[name]].get()
        output.mkdir(parents=True,exist_ok=True)
        for name in manifest['files']:
            marker = prefix+'.post.'
            if name.startswith(marker):
                leaf = name[len(marker):-4]
                records.save(capture/(prefix+'.pre.'+leaf+'.bin'),output/name,raw[leaf])
    cp.cuda.get_current_stream().synchronize()


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='prepare':
        prepare(Path(sys.argv[2]))
    else:
        replay(Path(sys.argv[1]),Path(sys.argv[2]),Path(os.environ['MYJ_DUMP_BRIDGE']))
