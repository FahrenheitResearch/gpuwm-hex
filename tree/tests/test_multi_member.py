"""Resident-member isolation and admission, using real file and NetCDF payloads."""
from concurrent.futures import ThreadPoolExecutor
import errno
import json
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from hexcore import multi_member as mm, time_ledger
from hexcore.device_admission import admit_co_tenancy


def test_admission_sums_solo_floors_and_names_midrun_breakage():
    args = dict(mesh_kind="point-cull", cells=41145, levels=55,
                card_bytes=96 * 1024**3, free_bytes=80 * 1024**3,
                solo_required_bytes=10 * 1024**3)
    assert admit_co_tenancy(members=4, **args)["required_free_bytes"] == 40 * 1024**3
    with pytest.raises(MemoryError, match="first radiation workspace allocation"):
        admit_co_tenancy(members=4, **{**args, "free_bytes": 39 * 1024**3})
    # Global meshes have their own rows since the resident global ensemble (B5); a mesh past the
    # 30 km x1.655362 still has none, so it stays one member per card.
    assert admit_co_tenancy(members=4, **{**args, "mesh_kind": "global"})["maximum_members"] == 4
    with pytest.raises(MemoryError, match="table limit 1"):
        admit_co_tenancy(members=2, **{**args, "mesh_kind": "global", "cells": 2_621_442})
    with pytest.raises(MemoryError, match="table limit 1"):
        admit_co_tenancy(members=2, **{**args, "levels": 59})


def test_resident_admission_preserves_heterogeneous_stochastic_arrays_on_readmit():
    args = dict(mesh_kind="global", cells=40962, levels=55, members=3,
                card_bytes=96 * 1024**3, solo_required_bytes=10 * 1024**3)
    # Off, SPPT-only, and SKEB with a different degree, without a sharing credit.
    from hexcore.forecast_door import stochastic_array_bytes
    extra = (0, stochastic_array_bytes(40962, '{"sppt":true}'),
             stochastic_array_bytes(40962, '{"skeb":true,"degree":24}'))
    required = 3 * args['solo_required_bytes'] + sum(extra)
    with pytest.raises(MemoryError, match=f"required {required} bytes"):
        admit_co_tenancy(**args, free_bytes=required-1, member_additional_bytes=extra)
    receipt = admit_co_tenancy(**args, free_bytes=required, member_additional_bytes=extra)
    receipt2 = admit_co_tenancy(**{**args, 'solo_required_bytes': receipt['solo_required_bytes']},
        free_bytes=required, member_additional_bytes=tuple(receipt['member_additional_bytes']))
    assert receipt2['required_free_bytes'] == required
    with pytest.raises(ValueError, match='one nonnegative byte count per member'):
        admit_co_tenancy(**args, free_bytes=required, member_additional_bytes=(1,))


def test_private_proof_globals_keep_original_numerical_code():
    module = ModuleType("member_example")
    exec("pin = {'count': 0}\ndef call(value):\n    pin['count'] += value\n    return pin['count']", vars(module))
    left, right = mm.private_module(module), mm.private_module(module)
    assert left.call.__code__ is right.call.__code__ is module.call.__code__
    assert left.call(7) == 7
    assert right.call(3) == 3
    assert module.pin == {"count": 0}


def test_real_driver_and_proof_can_be_isolated():
    from hexcore.drivers import run_cuda_v841_forecast as driver
    for module in (driver, driver.proof):
        clone = mm.private_module(module)
        for name, value in vars(module).items():
            if hasattr(value, "__code__") and value.__globals__ is vars(module):
                assert getattr(clone, name).__code__ is value.__code__


def test_table_pool_shares_only_immutable_exact_bytes(tmp_path):
    pool = mm.SharedTables(tmp_path / "files", policy="nvme")
    source = np.arange(100, dtype=np.float32)
    prepared = SimpleNamespace(mesh={"x": source}, state=source.copy(), saved_diagnostics=source.copy())
    left, right = pool.host({"prepared": prepared}), pool.host({"prepared": prepared})
    assert left["prepared"].mesh["x"] is right["prepared"].mesh["x"]
    assert left["prepared"].state is prepared.state
    with pytest.raises(ValueError):
        left["prepared"].mesh["x"][0] = 99
    assert np.array_equal(left["prepared"].mesh["x"], source)
    assert type(left["prepared"].mesh["x"]) is np.ndarray
    from hexcore.prep_cache import _Encoder
    _Encoder().add(left["prepared"].mesh["x"])
    assert pool.rows[0]["storage"] == "NVMe-backed file"
    pool.close()
    assert not (tmp_path / "files").exists()
    assert pool.deleted[0]["bytes"] > source.nbytes


def test_slotted_prepared_and_vertical_tables_keep_types_and_scalar_shape(tmp_path):
    from dataclasses import dataclass
    @dataclass(frozen=True, slots=True)
    class Vertical:
        zz: np.ndarray
    @dataclass(frozen=True, slots=True)
    class Prepared:
        mesh: dict
        state: np.ndarray
        vertical: Vertical
    state = np.arange(5, dtype=np.float32)
    prepared = Prepared({"scalar": np.array(4, dtype=np.int32)}, state, Vertical(state))
    pool = mm.SharedTables(tmp_path / "tables", policy="nvme")
    left, right = (pool.host({"prepared": prepared})["prepared"] for _ in range(2))
    assert type(left) is Prepared and type(left.vertical) is Vertical
    assert left.vertical.zz is right.vertical.zz
    assert left.state is state
    assert left.mesh["scalar"].shape == ()
    pool.close()


def test_shm_capacity_and_allocation_race_fall_back(tmp_path, monkeypatch):
    shm = tmp_path / "shm"
    shm.mkdir()
    source = np.arange(100, dtype=np.float32)
    pool = mm.SharedTables(tmp_path / "files", shm_root=shm)
    monkeypatch.setattr(pool, "available", lambda root: 0)
    assert np.array_equal(pool.intern(source), source)
    assert pool.rows[0]["fallback"] == "available shm smaller than shared buffer"
    pool.close()


def test_shared_buffer_manifest_tracks_creation_and_directory_deletion(tmp_path):
    manifest = tmp_path / "created-paths.txt"
    pool = mm.SharedTables(tmp_path / "files", policy="nvme", creation_manifest=manifest)
    pool.intern(np.arange(10, dtype=np.float32))
    created = set(manifest.read_text().splitlines())
    assert str(manifest) in created and str(pool.root) in created
    assert {row["path"] for row in pool.rows}.issubset(created)
    pool.close()
    assert all(row["path"] in created for row in pool.deleted)
    assert any(row.get("kind") == "directory" and row["bytes"] == 0 for row in pool.deleted)


def test_shm_reservation_race_falls_back_without_a_bus_error(tmp_path, monkeypatch):
    shm = tmp_path / "shm"
    shm.mkdir()
    source = np.arange(100, dtype=np.float32)
    pool = mm.SharedTables(tmp_path / "race", shm_root=shm)
    monkeypatch.setattr(pool, "available", lambda root: 10**9)
    def full(*args):
        raise OSError(errno.ENOSPC, "capacity changed")
    monkeypatch.setattr(mm.os, "posix_fallocate", full, raising=False)
    assert np.array_equal(pool.intern(source), source)
    assert "exhausted" in pool.rows[0]["fallback"]
    pool.close()


def test_kernel_family_has_one_compiler_and_private_scrub_observers():
    class Cache:
        constructed = 0
        def __init__(self, **kwargs):
            type(self).constructed += 1
            self._kernels, self._observed = {}, {}
            self.post_launch = None
        def raw_kernel(self, name):
            return self._kernels.setdefault(name, object())
    family = mm.kernel_cache_family(Cache)
    left, right = family(), family()
    assert Cache.constructed == 1
    assert left.raw_kernel("dycore") is right.raw_kernel("dycore")
    left.post_launch = object()
    assert right.post_launch is None
    assert left._observed is not right._observed


def test_warm_member_lookup_does_not_wait_for_another_members_compiler():
    class Cache:
        def __init__(self):
            self.post_launch = None
            self.calls = []

        def raw_kernel(self, name, source, *, module_key, options=()):
            self.calls.append((name, source, module_key, tuple(options), self.post_launch))
            return object()

    family = mm.kernel_cache_family(Cache)
    member = family()
    args = ("kernel", "source")
    kwargs = dict(module_key="module", options=["--fmad=false"])
    resolved = member.raw_kernel(*args, **kwargs)
    # A different member can be compiling while this member queues its
    # already-resolved kernels. The old family lock blocked this lookup.
    with ThreadPoolExecutor(1) as pool:
        with family.family_lock:
            result = pool.submit(member.raw_kernel, *args, **kwargs)
            assert result.result(timeout=2) is resolved
    assert len(member.calls) == 1
    assert member.raw_kernel(*args, module_key="module", options=[]) is not resolved
    member.post_launch = object()
    assert member.raw_kernel(*args, **kwargs) is not resolved


def test_device_table_families_share_only_equal_input_bytes_and_restore():
    class Host:
        def __init__(self, offset=0):
            self.dimensions = {"nCells": 3}
            self.value = np.arange(3, dtype=np.float32) + offset
        def __getattr__(self, name):
            return self.value
    class Device:
        @classmethod
        def from_host(cls, host, **kwargs):
            return SimpleNamespace(buffer=host.value.copy())
    classes = [type(name, (Device,), {"from_host": classmethod(Device.from_host.__func__)})
               for name in ("DeviceMesh", "DeviceVerticalGrid", "DeviceReferenceState", "DeviceTerrainMetrics")]
    originals = [cls.__dict__["from_host"] for cls in classes]
    containers = SimpleNamespace(**{cls.__name__: cls for cls in classes},
                                 _mesh_value=lambda host, field: getattr(host, field))
    with mm.shared_device_tables(containers=containers) as counters:
        for cls in classes:
            left, right = cls.from_host(Host()), cls.from_host(Host())
            different = cls.from_host(Host(1))
            assert left is not right and left.buffer is right.buffer
            assert left.buffer is not different.buffer
            assert counters[cls.__name__] == {"constructed": 2, "reused": 1}
    assert [cls.__dict__["from_host"] for cls in classes] == originals


def test_member_ledger_ids_remain_distinct_on_one_card(tmp_path, monkeypatch):
    pytest.importorskip("gpuwm.timeline.ledger", reason="the engine ships no gpuwm.timeline (gpuwm 2.8.7 and earlier)")
    monkeypatch.delenv("GPUWM_TIME_LEDGER", raising=False)
    def worker(run_id):
        out = tmp_path / run_id
        with time_ledger.member(run_id, card=7, root=out):
            time_ledger.emit("forecast.construct", "begin", t_mono=100, t_unix=1700000000)
            out.mkdir()
            for stage, phase in (("forecast.construct", "end"),
                                 ("forecast.first_step", "begin"), ("forecast.first_step", "end"),
                                 ("forecast.integrate", "begin"), ("forecast.integrate", "end")):
                time_ledger.emit(stage, phase)
        return [json.loads(line) for line in (out / "timeline.jsonl").read_text().splitlines()]
    with ThreadPoolExecutor(2) as pool:
        rows = list(pool.map(worker, ["left", "right"]))
    for run_id, events in zip(("left", "right"), rows):
        assert len(events) == 6
        assert {row["run_id"] for row in events} == {run_id}
        assert {row["card"] for row in events} == {7}
        assert events[0]["t_mono"] == 100


def test_netcdf_members_serialize_real_library_and_restore_alias(tmp_path):
    import netCDF4
    original = netCDF4.Dataset
    barrier = threading.Barrier(2)
    def worker(index):
        barrier.wait(timeout=10)
        path = tmp_path / f"member-{index}.nc"
        with netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC") as dataset:
            dataset.createDimension("cell", 10)
            dataset.createVariable("value", "f4", ("cell",))[:] = np.arange(10, dtype=np.float32) + index
        with netCDF4.Dataset(path) as dataset:
            return dataset.variables["value"][:].copy()
    with mm.serial_netcdf(), ThreadPoolExecutor(2) as pool:
        values = list(pool.map(worker, range(2)))
    assert netCDF4.Dataset is original
    assert np.array_equal(values[0], np.arange(10))
    assert np.array_equal(values[1], np.arange(10) + 1)


def test_roster_presets_use_the_existing_door_resolver(tmp_path):
    path = tmp_path / "roster.json"
    path.write_text(json.dumps([{"run_id": "left", "preset": "reference"},
                                {"run_id": "right", "preset": "tempo"}]))
    resolved = []
    def resolve(member):
        resolved.append(member)
        return SimpleNamespace(mesh="point", grid="grid", static="static", preflight=False)
    args = SimpleNamespace(members=path, out=tmp_path / "out", scratch=tmp_path / "scratch")
    assert len(mm.roster_requests(args, SimpleNamespace(resolve_request=resolve))) == 2
    assert [member.preset for member in resolved] == ["reference", "tempo"]
    assert all(member.members is None for member in resolved)


@pytest.mark.parametrize('encoded', [False, True])
@pytest.mark.parametrize('count', [1, 2])
def test_roster_stochastic_overrides_inherit_shared_settings(tmp_path, encoded, count):
    from hexcore.stochastic import StochasticConfig, member_number
    shared = dict(sppt=True, skeb=True, seed=711, degree=8, time_s=600)
    records = [dict(run_id='left', stochastic=dict(sppt=False)),
               dict(run_id='right', stochastic=dict(skeb=False, seed=712))][:count]
    if encoded:
        for record in records:
            record['stochastic'] = json.dumps(record['stochastic'])
    path = tmp_path/'roster.json'
    path.write_text(json.dumps(records))
    resolved = []
    def resolve(member):
        resolved.append(StochasticConfig.parse(member.stochastic))
        return SimpleNamespace(mesh='global', grid='grid', static='static', preflight=False)
    args = SimpleNamespace(members=path, out=tmp_path/'out', scratch=tmp_path/'scratch',
                           stochastic=json.dumps(shared) if encoded else shared.copy())
    mm.roster_requests(args, SimpleNamespace(resolve_request=resolve))
    assert resolved[0].seed == 711 and resolved[0].degree == 8 and resolved[0].time_s == 600
    assert not resolved[0].sppt and resolved[0].skeb
    if count == 2:
        assert resolved[1].seed == 712 and resolved[1].degree == 8 and resolved[1].time_s == 600
        assert resolved[1].sppt and not resolved[1].skeb
    assert [value.member for value in resolved] == [member_number(name) for name in ('left', 'right')[:count]]
    assert args.stochastic == (json.dumps(shared) if encoded else shared)


@pytest.mark.parametrize('encoded', [False, True])
@pytest.mark.parametrize('identity_override', [None, 'shared', 'record'])
def test_roster_stochastic_identity_survives_singleton_regrouping(tmp_path, encoded, identity_override):
    from hexcore.stochastic import StochasticConfig, member_number
    shared = dict(sppt=True, skeb=True, seed=711)
    if identity_override == 'shared':
        shared['member'] = 29
    records = [dict(run_id=name) for name in ('alpha', 'beta', 'gamma', 'delta')]
    records[1]['stochastic'] = dict(seed=712, skeb=False)
    if identity_override == 'record':
        shared['member'] = 29
        for index, record in enumerate(records):
            record.setdefault('stochastic', {})['member'] = 101+index
    if encoded:
        for record in records:
            if 'stochastic' in record:
                record['stochastic'] = json.dumps(record['stochastic'])
    path = tmp_path/'roster.json'
    args = SimpleNamespace(members=path, out=tmp_path/'out', scratch=tmp_path/'scratch',
                           stochastic=json.dumps(shared) if encoded else shared.copy())
    def resolve(member):
        return SimpleNamespace(mesh='global', grid='grid', static='static', preflight=False,
                               config=StochasticConfig.parse(member.stochastic))
    audit = []
    def group(rows):
        path.write_text(json.dumps(rows))
        result = {name: request.config for name, request in
                  mm.roster_requests(args, SimpleNamespace(resolve_request=resolve))}
        audit.append({name: dict(config=json.loads(value.json()), member_seed_hex=f'{value.member_seed:016x}')
                      for name, value in result.items()})
        return result
    reference = group(records)
    for selection in (records[::-1], records[1:3], *([record] for record in records)):
        for name, config in group(selection).items():
            assert config == reference[name]
            assert config.member_seed == reference[name].member_seed
    for index, record in enumerate(records):
        expected = (29 if identity_override == 'shared' else 101+index
                    if identity_override == 'record' else member_number(record['run_id']))
        assert reference[record['run_id']].member == expected
    if identity_override != 'shared':
        assert len({config.member_seed for config in reference.values()}) == 4
    assert not reference['beta'].skeb and reference['beta'].seed == 712
    assert args.stochastic == (json.dumps(shared) if encoded else shared)
    (tmp_path/'singleton-identity-audit.json').write_text(json.dumps(audit, indent=2)+'\n')


def test_singleton_roster_identity_keeps_single_forecast_defaults_off(tmp_path):
    from hexcore.stochastic import StochasticConfig, member_number
    path = tmp_path/'roster.json'
    path.write_text(json.dumps([dict(run_id='alpha')]))
    args = SimpleNamespace(members=path, out=tmp_path/'out', scratch=tmp_path/'scratch')
    def resolve(member):
        return SimpleNamespace(mesh='global', grid='grid', static='static', preflight=False,
                               config=StochasticConfig.parse(member.stochastic))
    requests = mm.roster_requests(args, SimpleNamespace(resolve_request=resolve))
    config = requests[0][1].config
    assert not config.enabled and config.member == member_number('alpha')
    assert not hasattr(args, 'stochastic')


def test_complete_identical_preparation_has_private_state_and_proof(tmp_path):
    boundary = tmp_path / "lbc.nc"
    boundary.write_bytes(b"boundary-v1")
    proof = ModuleType("proof_example")
    exec("pin = 0\ndef read():\n    return pin", vars(proof))
    calls = []
    def prepare(paths, authority, **kwargs):
        calls.append(1)
        member.proof.pin = 7
        return {"state": np.arange(8, dtype=np.float32)}
    cache = mm.PreparedMembers()
    member = SimpleNamespace(proof=mm.private_module(proof))
    first = cache.prepare(member, prepare, {"init": tmp_path / "init.nc"},
                          {"sha256": "initial-v1"}, lbc_paths=[str(boundary)], pbl_cadence="auto")
    member = SimpleNamespace(proof=mm.private_module(proof))
    second = cache.prepare(member, prepare, {"init": tmp_path / "init.nc"},
                           {"sha256": "initial-v1"}, lbc_paths=[str(boundary)], pbl_cadence="auto")
    assert len(calls) == 1
    assert member.proof.read() == 7 and proof.read() == 0
    second["state"][0] = 99
    assert first["state"][0] == 0
    boundary.write_bytes(b"boundary-v2")
    cache.prepare(member, prepare, {"init": tmp_path / "init.nc"},
                  {"sha256": "initial-v1"}, lbc_paths=[str(boundary)], pbl_cadence="auto")
    assert len(calls) == 2


@pytest.mark.parametrize("run_id", ["../other", "", "..", "/absolute"])
def test_roster_refuses_path_escape_before_door(tmp_path, run_id):
    path = tmp_path / "roster.json"
    path.write_text(json.dumps([{"run_id": run_id}]))
    with pytest.raises(mm.MultiMemberRefusal, match="safe path component"):
        mm.roster_requests(SimpleNamespace(members=path), None)


def test_roster_out_is_the_members_parent_before_they_resolve(tmp_path):
    # The breakage: the members were resolved before <out> existed, so the door refused each one for a
    # missing parent, and an <out> made beforehand was refused as existing: no roster could run at all.
    path = tmp_path / "roster.json"
    path.write_text(json.dumps([{"run_id": "m1"}]))
    seen = []

    def resolve(member):
        seen.append(member.out.parent.is_dir())
        raise mm.MultiMemberRefusal("stop after the first member")

    args = SimpleNamespace(members=path, out=tmp_path / "out", scratch=None)
    with pytest.raises(mm.MultiMemberRefusal, match="stop after"):
        mm.run_members(args, SimpleNamespace(resolve_request=resolve))
    assert seen == [True]
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.forecast-scratch").exists()
