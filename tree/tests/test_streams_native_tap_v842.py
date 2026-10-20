"""Real f1 output-diagnostics pool capture through the complete stream module.

MPAS mpas_atm_core.F:663-705,894-902; mpas_io_streams.F:1659-1857.
The stated input set is every registered stream field present in the sealed
step-zero POST snapshot. Uncaptured physics pools are explicitly inventoried;
complete published-file replay separately covers every native output field.
No diagnostic or physics computation is being claimed here.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest

from hexcore.mpasphys import resolve_namelist
from hexcore.mpasphys.native_backend import NativePhysics
from hexcore.mpasphys.pool_v842 import RegistryLayout
from hexcore.restart_v842 import NativePoolStreams, inspect_stream, read_stream, write_stream
from hexcore.streams_v842 import field_definition


def test_real_native_tap_streams_and_restart(tmp_path):
    root = os.environ.get('MPAS_STREAM_REAL_TAP')
    if not root:
        pytest.skip('sealed real f1 pool capture not supplied')
    from oracle_replay import Capture, transport
    root = Path(root)
    publication = json.loads((root/'PUBLICATION.json').read_text())
    assert publication['owner'] == 'f1-native-tap'
    seal = json.loads((root/'manifest.json').read_text())
    prefix = 'boundary__atm_compute_output_diagnostics.s0000000000.e0000000001.r000000.post.'
    reference = Path(os.environ['MPAS_NATIVE_POOL_REAL_PUBLICATION'])
    text = (reference/'namelist.atmosphere').read_text()
    metadata = inspect_stream(sorted(reference.glob('output.*.nc'))[0], scratch=tmp_path/'metadata')
    layout = RegistryLayout(Path(os.environ['MPAS_SOURCE']), resolve_namelist(text),
                            text, metadata['dimensions'])
    library = Path(os.environ['MPAS_NATIVE_LIBRARY'])
    decoder = transport(library)
    selection = {}
    def capture(row):
        suffix = '.t1' if row['nlevels'] > 1 else ''
        return prefix + row['pool']+'.'+row['pool_name']+suffix+'.bin'
    receipt = dict(process_count=1, transport_process_count=1,
        scope='All registered stream fields present in real f1 step-zero diagnostic POST pools',
        capture_root=str(root), publication_sha256=hashlib.sha256((root/'PUBLICATION.json').read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        captures=[], streams=[], different_words=0, passed=False)
    with NativePhysics(library, test_only=True) as native:
        probe = NativePoolStreams(native, layout, reference/'streams.atmosphere', tmp_path/'unused',
            list_directory=Path(os.environ['MPAS_NATIVE_STREAM_DEFAULTS']))
        streams = [s for s in probe.streams.values()
                   if s.name in ('output','diagnostics','restart','surface','diag_ugwp')]
        unavailable = {}
        for stream in streams:
            absent = []
            for name in stream.fields(layout.packages, streams=probe.streams):
                if name in ('xtime','initial_time'):
                    continue
                row, _ = probe.rows[name]
                filename = capture(row)
                if filename not in seal['files']:
                    absent.append(name)
                else:
                    selection[row['pool'],row['pool_name']] = row
            unavailable[stream.name] = absent
        assert len(selection) == 182
        for row in selection.values():
            filename = capture(row); path = root/filename
            expected = seal['files'][filename]['sha256']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
            with Capture(decoder, path) as captured:
                assert captured.shape == tuple(row['shape'])
                assert captured.lower == (1,)*len(row['shape'])
                for level in range(1, row['nlevels']+1):
                    if row['kind'] == 'character':
                        native.put_text_field(row['pool'],row['pool_name'],captured.scalar(),
                            level=level,nlevels=row['nlevels'])
                    else:
                        native.put_field(row['pool'],row['pool_name'],captured.array(),
                            level=level,nlevels=row['nlevels'])
            receipt['captures'].append(dict(file=filename,sha256=expected))
        writer = NativePoolStreams(native, layout, reference/'streams.atmosphere', tmp_path/'out',
            list_directory=Path(os.environ['MPAS_NATIVE_STREAM_DEFAULTS']),
            global_attrs=metadata['global_attrs'], unavailable_fields=unavailable)
        for stream in streams:
            fields = writer.fields(stream)
            if not fields:
                receipt['streams'].append(dict(name=stream.name,fields=0,words=0,
                    different_words=0,scope='No fields from this pool are in the stated tap capture'))
                continue
            target = tmp_path/(stream.name+'.nc')
            write_stream(target, writer.publication_stream(stream), writer.mesh, fields,
                writer.start, initial_time=writer.initial, active_packages=layout.packages,
                streams=writer.streams, global_attrs=writer.attrs, scratch=tmp_path/'encode')
            words = 0
            with read_stream(target, scratch=tmp_path/'decode',
                    layouts={name:dict(dimensions=field.dimensions,
                        expected_dimensions=tuple(reversed(field_definition(name)['dimensions'])),
                        connectivity_offset=0, dtype='f4' if field.values.dtype.kind == 'f' else 'i4')
                        for name,field in fields.items()
                             if not isinstance(field,str)}) as decoded:
                assert set(decoded.variables) == set(writer.publication_stream(stream).fields(
                    layout.packages,streams=writer.streams))
                for name, field in fields.items():
                    actual = decoded.variables[name]
                    if isinstance(field,str):
                        assert actual.text == (field,)
                        words += len(field)
                    else:
                        assert actual.values.dtype == field.values.dtype
                        assert actual.values.shape == field.values.shape
                        assert actual.values.tobytes() == field.values.tobytes(),name
                        words += field.values.size
            receipt['streams'].append(dict(name=stream.name,fields=len(fields),words=words,different_words=0))
            if stream.name == 'restart':
                restart = target
            else:
                target.unlink()
        receipt.update(passed=True, unavailable_capture_fields=unavailable,
            written_words=sum(row['words'] for row in receipt['streams']))
        restart.unlink()
    Path(os.environ['MPAS_STREAM_REAL_TAP_RECEIPT']).write_text(json.dumps(receipt,indent=2)+'\n')
