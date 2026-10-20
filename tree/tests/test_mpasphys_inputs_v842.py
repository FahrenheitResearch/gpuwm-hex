"""Native table selection and global SST words at the distributed boundary."""
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.mpasphys.inputs_v842 import HostField, ProjectedSurfaceRecordReader, stage_tables


class Collectives:
    """Replay root and peer calls, retaining the actual projected packets."""
    def __init__(self):
        self.broadcasts = []
        self.packets = None
        self.clocks = None

    def comm(self, rank):
        bus = self
        class Comm:
            cursor = 0
            def Get_rank(self): return rank
            def allgather(self, value): return bus.clocks or [value, value]
            def bcast(self, value, root):
                if rank == root:
                    bus.broadcasts.append(value)
                result = bus.broadcasts[self.cursor]
                self.cursor += 1
                return result
            def scatter(self, values, root):
                if rank == root: bus.packets = values
                return bus.packets[rank]
        return Comm()


class GlobalReader:
    def __init__(self, library, path, *, n_cells):
        assert n_cells == 5
        self.times = [datetime(2026, 10, 17)]
        self.calls = 0
        self.last_record = None
        self.fields = dict(
            sst=np.array([0x80000000, 0x7fc12345, 0x3f800000, 0x7fc54321, 0x40000000], np.uint32).view(np.float32),
            xice=np.arange(5, dtype=np.float32))

    def __call__(self, clock, pool):
        self.calls += 1
        for name, words in self.fields.items(): pool[name].set(words)
        self.last_record = 0


def readers(factory=GlobalReader):
    parts = [SimpleNamespace(_global_cells=5, cell_l2g=np.array(ids))
             for ids in ([4, 0, 3], [2, 1, 4, 0])]
    bus = Collectives()
    ranks = [ProjectedSurfaceRecordReader(bus.comm(rank), parts, factory, 'unused', 'unused')
             for rank in range(2)]
    return bus, parts, ranks


def pool(size):
    fields = {name: HostField(size+1) for name in ('sst', 'xice')}
    for field in fields.values(): field.words.fill(-99)
    return fields


def test_surface_global_words_project_once_to_noncontiguous_owned_and_halo_cells():
    bus, parts, ranks = readers()
    assert ranks[0].reader.calls == 0 and ranks[1].reader is None
    for part, rank in zip(parts, ranks):
        fields = pool(len(part.cell_l2g))
        rank('2026-10-17_00:00:00', fields)
        for name, expected in ranks[0].reader.fields.items():
            np.testing.assert_array_equal(fields[name].words[:-1].view(np.uint32),
                expected[part.cell_l2g].view(np.uint32))
            assert fields[name].words[-1] == -99
        assert rank.last_record == 0 and rank.times == [datetime(2026, 10, 17)]
        assert rank.reads[0]['seconds'] >= 0
    assert ranks[0].reader.calls == 1


def test_surface_bad_second_target_does_not_publish_first():
    _, parts, ranks = readers()
    fields = pool(len(parts[0].cell_l2g))
    fields['xice'].words = np.zeros(1, np.float64)
    with pytest.raises(ValueError, match='xice rank extent or dtype'):
        ranks[0]('2026-10-17_00:00:00', fields)
    np.testing.assert_array_equal(fields['sst'].words, -99)
    assert ranks[0].last_record is None


def test_surface_rank_clock_mismatch_refuses_before_decode():
    bus, parts, ranks = readers()
    bus.clocks = ['2026-10-17_00:00:00', '2026-10-17_00:01:00']
    with pytest.raises(ValueError, match='clocks differ'):
        ranks[0](bus.clocks[0], pool(len(parts[0].cell_l2g)))
    assert ranks[0].reader.calls == 0


def test_surface_decode_error_reaches_each_rank_without_publishing():
    class FailedReader(GlobalReader):
        def __call__(self, clock, pool):
            raise ValueError('Rust surface sst decoder failed: -10002')
    _, parts, ranks = readers(FailedReader)
    for part, rank in zip(parts, ranks):
        fields = pool(len(part.cell_l2g))
        with pytest.raises(ValueError, match='Rust surface sst decoder failed: -10002'):
            rank('2026-10-17_00:00:00', fields)
        np.testing.assert_array_equal(fields['sst'].words, -99)


def test_shared_rrfs_staging_uses_native_selector_overrides_and_initialization_namelist(tmp_path):
    from hexcore.mpasphys.rrfs import driver_rrfs
    special = next(name for name in driver_rrfs.specification()['native_table_names']
                   if not name.endswith(('.TBL', '.DBL')) and not name.startswith(('RRTMG_', 'CCN_')))
    base, override, out = [tmp_path/name for name in ('base', 'override', 'out')]
    for path in (base, override, out): path.mkdir()
    (base/special).write_bytes(b'base bytes')
    (override/special).write_bytes(b'override bytes')
    (base/'SURFACE.TBL').write_bytes(b'opaque table')
    (base/'unselected.txt').write_bytes(b'unselected')
    namelist = tmp_path/'physics.nml'
    namelist.write_text('&mp_nssl\nopaque_option = 5\n/\n')
    row = dict(native_table_selector='hexcore.mpasphys.rrfs.driver_rrfs:native_table_inputs',
               native_table_override_keyword='override_roots',
               native_initialization_namelist='hexcore.mpasphys.rrfs.driver_rrfs:stage_native_initialization_namelist')
    runtime = SimpleNamespace(native_table_roots=lambda *args: [base, override])
    result = stage_tables(runtime, SimpleNamespace(physics_namelist=namelist),
                          SimpleNamespace(source=tmp_path), None, out, row)
    assert set(result['files']) == {special, 'SURFACE.TBL'}
    assert (out/special).resolve() == (override/special).resolve()
    assert (out/'namelist.atmosphere').read_bytes() == namelist.read_bytes()
    assert len(result['initialization_namelist_sha256']) == 64
    # Each rank's existing selector sees the exact same completed staging.
    for table in driver_rrfs.native_table_inputs([base], override_roots=[override]):
        assert (out/table.name).resolve() == table.resolve()
    assert driver_rrfs.stage_native_initialization_namelist(namelist, out) == result['initialization_namelist_sha256']


def test_table_staging_refuses_an_existing_different_destination(tmp_path):
    base, out = tmp_path/'base', tmp_path/'out'
    base.mkdir(); out.mkdir()
    (base/'SURFACE.TBL').write_bytes(b'new')
    (out/'SURFACE.TBL').write_bytes(b'existing')
    with pytest.raises(ValueError, match='native table destination already differs'):
        stage_tables(SimpleNamespace(native_table_roots=lambda *args: [base]),
            SimpleNamespace(), SimpleNamespace(source=base), None, out, {})
    assert (out/'SURFACE.TBL').read_bytes() == b'existing'
