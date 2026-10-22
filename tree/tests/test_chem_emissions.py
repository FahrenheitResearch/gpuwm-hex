"""Source binding refuses wrong mesh, unknown air, gaps and duplicate mass."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from hexcore import chem_emissions as source


def binding(tmp_path, monkeypatch, *, entries=None, regional=False, species=("smoke",)):
    monkeypatch.setitem(sys.modules, "cupy", np)
    entries = entries or [(0, 10, 1.), (10, 10, 3.)]
    initial = tmp_path / "initial.nc"
    initial.write_bytes(b"initial from Rust")
    data = {str(initial): {n: np.full((2, 3), 4e-9, np.float32) for n in species}}
    emissions = []
    for i, (at, span, value) in enumerate(entries):
        path = tmp_path / f"emissions-{i}.nc"
        path.write_bytes(b"flux from Rust")
        data[str(path)] = {"smoke": np.full(3, value, np.float32)}
        emissions.append(dict(path=path.name, valid_time=at, interval_s=span))
    config = dict(valid_time=0, mesh_sha256="a"*64, initial=dict(path=initial.name), emissions=emissions)
    atmosphere = NS(state=NS(time_seconds=0., scalars=np.zeros((len(species), 2, 3), np.float32)))
    runtime = NS(raw=dict(inputs=config), path=tmp_path/"chemistry.json", nlev=2, ncells=3,
                 species=[NS(name=n) for n in species], solved=3,
                 surface_flux={n: np.zeros(3, np.float32) for n in species},
                 row=NS(index_of=lambda n: species.index(n)), driver=NS(atmosphere=atmosphere))
    host = dict(prepared=NS(input_bytes=dict(grid=dict(sha256="a"*64)),
                            mesh=NS(arrays=dict(bdyMaskCell=np.array([0,6,7])))),
                regional={} if regional else None)
    monkeypatch.setattr(source, "_frame", lambda path, *args, **kwargs: data[path])
    return NS(backend=NS(chemistry=runtime), runtime=runtime, host=host,
              config=config, data=data, atmosphere=atmosphere)


def test_arbitrary_table_row_and_global_inventory():
    rows = json.loads(source.TABLE.read_text())["sources"]
    assert len({r["id"] for r in rows}) == len(rows)
    global_row = source.source_row("gbbepx-global-pm25")
    assert global_row["variables"][0]["units"] == "kg s-1 m-2"
    assert global_row["grid"]["valid_time_offset_s"] == -43200
    assert source.source_row("merra2-aerosol")["source_basis"] == "moist_air"
    assert source.source_row("merra2-aerosol")["humidity"] == "QV"


def test_source_time_has_explicit_timezone():
    assert source.instant("2025-06-05T23:00:00Z") == 1749164400
    with pytest.raises(ValueError, match="UTC offset"):
        source.instant("2025-06-05T23:00:00")
    with pytest.raises(ValueError):
        source.instant(True)
    with pytest.raises(ValueError, match="integral UTC second"):
        source.instant("2025-06-05T23:00:00.5Z")


def test_initial_and_flux_binding_integrates_across_frame_boundary(tmp_path, monkeypatch):
    b = binding(tmp_path, monkeypatch)
    receipt = source.attach(b.backend, b.host)
    assert receipt["initial_species"] == ["smoke"]
    assert np.all(b.runtime.initial["smoke"] == np.float32(4e-9))
    b.runtime.boundary(b.atmosphere, b.runtime, 5, 10)
    assert np.all(b.runtime.surface_flux["smoke"] == 2.)
    with pytest.raises(ValueError, match="complete timestep"):
        b.runtime.boundary(b.atmosphere, b.runtime, 15, 10)


def test_missing_frame_does_not_mean_zero_emission(tmp_path, monkeypatch):
    b = binding(tmp_path, monkeypatch, entries=[(0, 5, 1.), (10, 10, 3.)])
    source.attach(b.backend, b.host)
    with pytest.raises(ValueError, match="coverage"):
        b.runtime.boundary(b.atmosphere, b.runtime, 6, 0)
    with pytest.raises(ValueError, match="complete timestep"):
        b.runtime.boundary(b.atmosphere, b.runtime, 0, 10)


def test_duplicate_sources_cannot_double_mass(tmp_path, monkeypatch):
    b = binding(tmp_path, monkeypatch, entries=[(0, 10, 1.), (5, 10, 3.)])
    with pytest.raises(ValueError, match="double count"):
        source.attach(b.backend, b.host)


def test_forecast_mesh_authority_must_match(tmp_path, monkeypatch):
    b = binding(tmp_path, monkeypatch)
    b.config["mesh_sha256"] = "b"*64
    with pytest.raises(ValueError, match="grid authority"):
        source.attach(b.backend, b.host)


@pytest.mark.parametrize("bad", [-1., float("nan"), float("inf")])
def test_invalid_initial_mass_is_refused(tmp_path, monkeypatch, bad):
    b = binding(tmp_path, monkeypatch)
    b.data[str(tmp_path/"initial.nc")]["smoke"][0, 0] = bad
    with pytest.raises(ValueError, match="initial smoke"):
        source.attach(b.backend, b.host)


def test_explicit_zero_source_declaration_for_nonemitting_species(tmp_path, monkeypatch):
    b = binding(tmp_path, monkeypatch, species=("smoke", "dust_1"))
    with pytest.raises(ValueError, match="coverage"):
        source.attach(b.backend, b.host)
    b.config["zero_emission_species"] = ["dust_1"]
    source.attach(b.backend, b.host)
    assert np.all(b.runtime.surface_flux["dust_1"] == 0)


def test_regional_boundary_needs_bracketing_composition(tmp_path, monkeypatch):
    b = binding(tmp_path, monkeypatch, regional=True)
    with pytest.raises(ValueError, match="bracketing"):
        source.attach(b.backend, b.host)


def test_regional_composition_drives_specified_cells_only(tmp_path, monkeypatch):
    b = binding(tmp_path, monkeypatch, regional=True)
    frames=[]
    for at,value in [(0,1e-9),(20,3e-9)]:
        path=tmp_path/f"boundary-{at}.nc";path.write_bytes(b"boundary from Rust")
        b.data[str(path)]={"smoke":np.full((2,3),value,np.float32)}
        frames.append(dict(path=path.name,valid_time=at))
    b.config["boundary"]=frames
    source.attach(b.backend,b.host)
    b.runtime.boundary(b.atmosphere,b.runtime,10,1)
    assert np.all(b.atmosphere.state.scalars[0,:,0]==0)
    assert np.allclose(b.atmosphere.state.scalars[0,:,1:],2e-9,rtol=1e-7,atol=0)


def test_research_license_is_named_opt_in(tmp_path):
    path=tmp_path/"sources.json"
    path.write_text(json.dumps(dict(schema="gpuwm-hex.chem-sources/v1",sources=[
        dict(id="arbitrary-product",research_only=True,license=dict(id="named-terms"))])))
    with pytest.raises(ValueError,match="--research-license named-terms"):
        source.source_row("arbitrary-product",path)
    assert source.source_row("arbitrary-product",path,research_license="named-terms")["id"]=="arbitrary-product"


def test_fetch_preserves_existing_partial(tmp_path, monkeypatch):
    destination = tmp_path / "source.nc"
    part = tmp_path / "source.nc.part"
    part.write_bytes(b"earlier producer")
    monkeypatch.setattr(source, "discover", lambda *a: "https://example.test/source.nc")
    with pytest.raises(FileExistsError):
        source.fetch({"id":"arbitrary"}, 0, destination)
    assert part.read_bytes() == b"earlier producer"


def test_fetch_publication_does_not_replace_concurrent_output(tmp_path, monkeypatch):
    from io import BytesIO
    destination = tmp_path / "source.nc"
    monkeypatch.setattr(source, "discover", lambda *a: "https://example.test/source.nc")
    def response(*args, **kwargs):
        destination.write_bytes(b"concurrent producer")
        return BytesIO(b"new download")
    monkeypatch.setattr(source, "urlopen", response)
    with pytest.raises(FileExistsError):
        source.fetch({"id":"arbitrary"}, 0, destination)
    assert destination.read_bytes() == b"concurrent producer"
    assert not (tmp_path / "source.nc.part").exists()
    assert json.loads((tmp_path / "deleted-files.jsonl").read_text())["bytes"] == 12


def test_fetch_rejects_silent_http_truncation(tmp_path, monkeypatch):
    from io import BytesIO
    destination = tmp_path / "source.nc"
    stream = BytesIO(b"short body")
    stream.headers = {"Content-Length": "100"}
    stream.status = 200
    monkeypatch.setattr(source, "discover", lambda *a: "https://example.test/source.nc")
    monkeypatch.setattr(source, "urlopen", lambda *a, **kw: stream)
    with pytest.raises(ValueError, match="truncated chemistry response"):
        source.fetch({"id": "arbitrary"}, 0, destination)
    assert not destination.exists()
    assert not (tmp_path / "source.nc.part").exists()
    assert json.loads((tmp_path / "deleted-files.jsonl").read_text())["bytes"] == 10


def test_interval_start_selects_midpoint_file_across_day_boundary(monkeypatch):
    from io import BytesIO
    seen = []
    def listing(url, timeout):
        seen.append(url)
        return BytesIO(b'<a href="GEOS.fp.asm.tavg3_2d_adg_Nx.20261020_0030.V01.nc4">one</a>'
                       b'<a href="GEOS.fp.asm.tavg3_2d_adg_Nx.20261020_0030.V02.nc4">two</a>')
    monkeypatch.setattr(source, 'urlopen', listing)
    row = source.source_row('geos-fp-gocart-dust')
    url = source.discover(row, '2026-10-19T23:00:00Z')
    assert seen == ['https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y2026/M10/D20/']
    assert url.endswith('20261020_0030.V02.nc4')
