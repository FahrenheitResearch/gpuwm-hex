"""A convection-permitting nest is emitted with the HRRR-shape suite, not the gfs/era5-class Morrison default.

The defect: `gpuwm domain --source hex-global` with no profile gave every
3 km nest Morrison, whose wet-snow reflectivity showed a 55-65 dBZ shield over the stratiform (Kansas trigger nest
+6 h: 42,822 km2 >= 50 dBZ against MRMS 1,107).
"""

from hexcore import add_nest_door as door


def _capture(tmp_path, monkeypatch, mp):
    seen = []

    def fake_run(argv, log):
        seen.append([str(a) for a in argv])
        if "domain" in seen[-1]:
            (tmp_path / "n.toml").write_text(f"[experiment]\nrestart_interval_s = 3600.0\n\n[physics]\nmp_physics = {mp}\n")
    monkeypatch.setattr(door, "_run", fake_run)
    return seen


def test_a_3km_nest_is_emitted_with_the_convection_permitting_suite_and_its_aerosol_climatology(tmp_path, monkeypatch):
    seen = _capture(tmp_path, monkeypatch, 28)
    door.nest_config("gpuwm", tmp_path, "n", lat=39.5, lon=-96.7, dx_km=3, size_km=600, hours=6,
                     cycle="2025-05-19T21", cadence_h=1.0, geog="/data/WPS_GEOG")
    domain = next(a for a in seen if "domain" in a)
    assert domain[domain.index("--physics-profile") + 1] == door.CONVECTION_PERMITTING_PROFILE
    assert ["gpuwm", "fetch-tables", "--wif-only"] in seen


def test_a_named_profile_wins_and_a_coarse_nest_keeps_the_engine_default(tmp_path, monkeypatch):
    seen = _capture(tmp_path, monkeypatch, 10)
    door.nest_config("gpuwm", tmp_path, "n", lat=39.5, lon=-96.7, dx_km=3, size_km=600, hours=6,
                     cycle="2025-05-19T21", cadence_h=1.0, geog="/g",
                     physics_profile="morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1")
    domain = next(a for a in seen if "domain" in a)
    assert domain[domain.index("--physics-profile") + 1] == "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"
    assert not any("fetch-tables" in a for a in seen)
    seen.clear()
    door.nest_config("gpuwm", tmp_path, "n", lat=39.5, lon=-96.7, dx_km=9, size_km=900, hours=6,
                     cycle="2025-05-19T21", cadence_h=1.0, geog=None)
    assert "--physics-profile" not in next(a for a in seen if "domain" in a)
