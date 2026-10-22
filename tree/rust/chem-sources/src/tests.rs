use super::*;
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    },
};
static NETCDF: Mutex<()> = Mutex::new(());
static COUNTER: AtomicU64 = AtomicU64::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!(
            "hex-chem-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        Self(dir)
    }
    fn path(&self, name: &str) -> String {
        self.0.join(name).to_str().unwrap().into()
    }
    fn mesh(&self) {
        let f = File::create(&self.path("mesh.nc")).unwrap();
        let c = f.dim("nCells", 2).unwrap();
        let a = f.define("latCell", &[c], "radians").unwrap();
        let b = f.define("lonCell", &[c], "radians").unwrap();
        let d = f.define("areaCell", &[c], "m2").unwrap();
        f.end().unwrap();
        f.put(a, &[0., 0.]).unwrap();
        f.put(b, &[0., 0.017453292]).unwrap();
        f.put(d, &[10., 20.]).unwrap();
        f.close().unwrap();
    }
    fn request(&self, row: Value) -> Request {
        serde_json::from_value(json!({"row":row,"input":self.path("source.nc"),"mesh":self.path("mesh.nc"),"output":self.path("output.nc"),"receipt":self.path("receipt.json"),"valid_time":0,"max_distance_m":30_000_000.,"mesh_sha256":"test"})).unwrap()
    }
    fn source(&self, values: &[f32], fill: f64) {
        let f = File::create(&self.path("source.nc")).unwrap();
        let t = f.dim("time", 1).unwrap();
        let y = f.dim("y", 1).unwrap();
        let x = f.dim("x", 3).unwrap();
        let tv = f
            .define("time", &[t], "seconds since 1970-01-01 00:00:00")
            .unwrap();
        let lat = f.define("lat", &[y, x], "degrees_north").unwrap();
        let lon = f.define("lon", &[y, x], "degrees_east").unwrap();
        let mass = f.define("mass", &[t, y, x], "kg").unwrap();
        f.numericattr(mass, "_FillValue", fill).unwrap();
        let qa = f.define("QA", &[t, y, x], "1").unwrap();
        f.end().unwrap();
        for (v, data) in [
            (tv, vec![0.]),
            (lat, vec![0., 0., 0.]),
            (lon, vec![0., 0.2, 1.]),
            (mass, values.to_vec()),
            (qa, vec![3., -1., 1.]),
        ] {
            f.put(v, &data).unwrap();
        }
        f.close().unwrap();
    }
    fn row() -> Value {
        json!({"id":"arbitrary-inventory","kind":"emission","interval_s":10,"grid":{"time":"time","latitude":"lat","longitude":"lon","latitude_units":"degrees_north","longitude_units":"degrees_east","angle":"degrees"},"variables":[{"species":"arbitrary_species","selector":"mass","units":"kg","no_fire_sentinel":-1,"masks":[{"selector":"QA","minimum":2,"maximum":3}]}]})
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        for e in fs::read_dir(&self.0).unwrap() {
            let p = e.unwrap().path();
            eprintln!(
                "deleted fixture {} {} bytes",
                p.display(),
                p.metadata().unwrap().len()
            );
            fs::remove_file(p).unwrap();
        }
        fs::remove_dir(&self.0).unwrap();
    }
}
#[test]
fn declared_no_fire_fill_and_qa_mass_ledger() {
    let _guard = NETCDF.lock().unwrap();
    let f = Fixture::new();
    f.mesh();
    f.source(&[10., -1., 30.], -1.);
    run(f.request(Fixture::row())).unwrap();
    let r: Value = serde_json::from_slice(&fs::read(f.path("receipt.json")).unwrap()).unwrap();
    let a = &r["species"][0]["accounting"];
    assert_eq!(a["source_mass_kg"], 40.);
    assert_eq!(a["mapped_mass_kg"], 10.);
    assert_eq!(a["masked_mass_kg"], 30.);
    assert_eq!(a["mass_residual_kg"], 0.);
    let out = File::open(&f.path("output.nc")).unwrap();
    let v = out.read("arbitrary_species", 0).unwrap();
    assert!((v.data[0] - 0.1).abs() < 1e-8);
    assert_eq!(v.data[1], 0.);
}
#[test]
fn missing_is_not_clean_air() {
    let _guard = NETCDF.lock().unwrap();
    let f = Fixture::new();
    f.mesh();
    f.source(&[10., -9999., 30.], -9999.);
    let error = run(f.request(Fixture::row())).unwrap_err();
    assert!(error.contains("missing cells"));
    assert!(!std::path::Path::new(&f.path("output.nc")).exists());
}
#[test]
fn time_and_units_are_binding() {
    let _guard = NETCDF.lock().unwrap();
    let f = Fixture::new();
    f.mesh();
    f.source(&[10., -1., 30.], -1.);
    let mut req = f.request(Fixture::row());
    req.valid_time = 1;
    assert!(run(req).unwrap_err().contains("valid time"));
    let mut row = Fixture::row();
    row["variables"][0]["units"] = json!("kg m-2 s-1");
    assert!(run(f.request(row)).unwrap_err().contains("units"));
}
#[test]
fn regular_spherical_grid_covers_sphere() {
    let _guard = NETCDF.lock().unwrap();
    let f = Fixture::new();
    let nc = File::create(&f.path("source.nc")).unwrap();
    let y = nc.dim("latitude", 3).unwrap();
    let x = nc.dim("longitude", 4).unwrap();
    let a = nc.define("lat", &[y], "degrees_north").unwrap();
    let b = nc.define("lon", &[x], "degrees_east").unwrap();
    nc.end().unwrap();
    nc.put(a, &[-90., 0., 90.]).unwrap();
    nc.put(b, &[-135., -45., 45., 135.]).unwrap();
    nc.close().unwrap();
    let nc = File::open(&f.path("source.nc")).unwrap();
    let a = source_areas(
        &nc,
        &json!({"grid":{"latitude":"lat","longitude":"lon","area_method":"regular-spherical","angle":"degrees"}}),
        0,
        &[3, 4],
    )
    .unwrap();
    let globe = 4. * std::f64::consts::PI * geo::R.powi(2);
    assert!((a.iter().sum::<f64>() / globe - 1.).abs() < 1e-14);
    assert!(a.iter().all(|v| *v > 0.));
}
#[test]
fn log_pressure_interpolation() {
    let p = [100., 1000.];
    let q = [1., 3.];
    assert!((interpolate(&p, &q, 1000f64.sqrt() * 10.) - 2.).abs() < 1e-12);
    assert_eq!(interpolate(&p, &q, 10.), 1.);
    assert_eq!(interpolate(&p, &q, 2000.), 3.);
}
fn volume_fixture(f: &Fixture, humidity_time: f32) {
    for (filename, humid) in [("source.nc", false), ("humidity.nc", true)] {
        let nc = File::create(&f.path(filename)).unwrap();
        let t = nc.dim("time", 1).unwrap();
        let z = nc.dim("lev", 2).unwrap();
        let y = nc.dim("lat", 1).unwrap();
        let x = nc.dim("lon", 2).unwrap();
        let tv = nc
            .define("time", &[t], "seconds since 1970-01-01 00:00:00")
            .unwrap();
        let lat = nc.define("lat", &[y], "degrees_north").unwrap();
        let lon = nc.define("lon", &[x], "degrees_east").unwrap();
        let levels = nc.define("lev", &[z], "1").unwrap();
        let data = nc
            .define(if humid { "QV" } else { "AER" }, &[t, z, y, x], "kg kg-1")
            .unwrap();
        let dp = if humid {
            None
        } else {
            Some(nc.define("DELP", &[t, z, y, x], "Pa").unwrap())
        };
        nc.end().unwrap();
        nc.put(tv, &[if humid { humidity_time } else { 0. }])
            .unwrap();
        nc.put(levels, &[1., 2.]).unwrap();
        nc.put(lat, &[0.]).unwrap();
        nc.put(lon, &[0., 1.]).unwrap();
        nc.put(
            data,
            if humid {
                &[0.2, 0.2, 0.2, 0.2]
            } else {
                &[1e-9, 2e-9, 3e-9, 4e-9]
            },
        )
        .unwrap();
        if let Some(dp) = dp {
            nc.put(dp, &[198., 198., 1800., 1800.]).unwrap();
        }
        nc.close().unwrap();
    }
    let nc = File::create(&f.path("pressure.nc")).unwrap();
    let c = nc.dim("nCells", 2).unwrap();
    let z = nc.dim("nVertLevels", 2).unwrap();
    let p = nc.define("pressure", &[c, z], "Pa").unwrap();
    let lat = nc.define("latCell", &[c], "radians").unwrap();
    let lon = nc.define("lonCell", &[c], "radians").unwrap();
    nc.putattr(-1, "valid_time_unix", "0").unwrap();
    nc.end().unwrap();
    nc.put(lat, &[0., 0.]).unwrap();
    nc.put(lon, &[0., 0.017453292]).unwrap();
    nc.put(p, &[100., 1099., 1099., 100.]).unwrap();
    nc.close().unwrap();
}
fn composition_row() -> Value {
    json!({"id":"arbitrary-aerosol","kind":"composition","source_basis":"moist_air","humidity":"QV","grid":{"time":"time","latitude":"lat","longitude":"lon","latitude_units":"degrees_north","longitude_units":"degrees_east","angle":"degrees"},"vertical":{"thickness":"DELP","top_pa":1,"coordinate":"lev"},"variables":[{"species":"arbitrary_species","components":[{"selector":"AER","units":"kg kg-1","factor":2}]}]})
}
fn volume_request(f: &Fixture, row: Value) -> Request {
    let mut req = f.request(row);
    req.pressure_file = Some(f.path("pressure.nc"));
    req.pressure_variables = vec!["pressure".into()];
    req.humidity_input = Some(f.path("humidity.nc"));
    req
}
#[test]
fn composition_humidity_pressure_order_and_mass_units() {
    let _guard = NETCDF.lock().unwrap();
    let f = Fixture::new();
    f.mesh();
    volume_fixture(&f, 0.);
    run(volume_request(&f, composition_row())).unwrap();
    let nc = File::open(&f.path("output.nc")).unwrap();
    let v = nc.read("arbitrary_species", 0).unwrap();
    assert_eq!(v.shape, vec![2, 2]);
    for (&a, b) in v.data.iter().zip([2.5e-9, 10e-9, 7.5e-9, 5e-9]) {
        assert!((a / b - 1.).abs() < 2e-7, "{a} {b}")
    }
}
#[test]
fn composition_humidity_must_match_valid_time() {
    let _guard = NETCDF.lock().unwrap();
    let f = Fixture::new();
    f.mesh();
    volume_fixture(&f, 1.);
    assert!(run(volume_request(&f, composition_row()))
        .unwrap_err()
        .contains("valid time"));
}
#[test]
fn composition_negative_factor_refused() {
    let _guard = NETCDF.lock().unwrap();
    let f = Fixture::new();
    f.mesh();
    volume_fixture(&f, 0.);
    let mut row = composition_row();
    row["variables"][0]["components"][0]["factor"] = json!(-1);
    assert!(run(volume_request(&f, row))
        .unwrap_err()
        .contains("factor cannot be negative"));
}
#[test]
fn map_is_native_hex_flux_with_explicit_cf_coordinates() {
    let _guard = NETCDF.lock().unwrap();
    let f = Fixture::new();
    f.mesh();
    f.source(&[10., -1., 30.], -1.);
    let mut req = f.request(Fixture::row());
    req.map_output = Some(f.path("map.nc"));
    req.map_bounds = [0., 2., -1., 1.];
    req.map_spacing_deg = 1.;
    run(req).unwrap();
    let nc = File::open(&f.path("map.nc")).unwrap();
    let field = nc.read("arbitrary_species", 0).unwrap();
    assert_eq!(field.shape, vec![3, 2]);
    assert_eq!(
        nc.attr("arbitrary_species", "units"),
        Some("kg m-2 s-1".into())
    );
    assert_eq!(nc.attr("lat", "standard_name"), Some("latitude".into()));
    assert!(field.data.iter().all(|v| v.is_finite() && *v >= 0.));
}

#[test]
fn unit_sphere_mesh_uses_earth_area() {
    let _guard = NETCDF.lock().unwrap();
    let f = Fixture::new();
    let nc = File::create(&f.path("mesh.nc")).unwrap();
    let c = nc.dim("nCells", 2).unwrap();
    let lat = nc.define("latCell", &[c], "radians").unwrap();
    let lon = nc.define("lonCell", &[c], "radians").unwrap();
    let a = nc.define("areaCell", &[c], "m^2").unwrap();
    nc.numericattr(-1, "sphere_radius", 1.).unwrap();
    nc.putattr(-1, "on_a_sphere", "YES").unwrap();
    nc.end().unwrap();
    nc.put(lat, &[0., 0.]).unwrap();
    nc.put(lon, &[0., 0.017453292]).unwrap();
    nc.put(a, &[0.0002, 0.0003]).unwrap();
    nc.close().unwrap();
    f.source(&[10., -1., 30.], -1.);
    run(f.request(Fixture::row())).unwrap();
    let out = File::open(&f.path("output.nc")).unwrap();
    let flux = out.read("arbitrary_species", 0).unwrap().data[0];
    let expected = 10. / (10. * 0.0002_f32 as f64 * geo::R.powi(2));
    assert!((flux / expected - 1.).abs() < 1e-7);
}
#[test]
fn unwritten_cf_time_uses_declared_coverage_attribute_only() {
    let _guard = NETCDF.lock().unwrap();
    let f = Fixture::new();
    let nc = File::create(&f.path("source.nc")).unwrap();
    let t = nc.dim("time", 1).unwrap();
    nc.define("time", &[t], "hours since 1970-01-01 00:00:00")
        .unwrap();
    nc.putattr(-1, "time_coverage_start", "1970-01-01T00:00:00Z")
        .unwrap();
    nc.end().unwrap();
    nc.close().unwrap();
    let nc = File::open(&f.path("source.nc")).unwrap();
    assert!(nc.read("time", 0).unwrap().data[0].is_nan());
    let row = json!({"grid":{"time":"time","time_attribute":"time_coverage_start"}});
    check_time(&nc, &row, 0, 0).unwrap();
    assert!(check_time(&nc, &row, 0, 1)
        .unwrap_err()
        .contains("valid time"));
    assert!(check_time(&nc, &json!({"grid":{"time":"time"}}), 0, 0).is_err());
}

#[test]
fn target_pressure_grid_and_time_are_authoritative() {
    let _guard = NETCDF.lock().unwrap();
    let f = Fixture::new();
    f.mesh();
    volume_fixture(&f, 0.);
    let nc = File::create(&f.path("wrong-pressure.nc")).unwrap();
    let c = nc.dim("nCells", 2).unwrap();
    let z = nc.dim("nVertLevels", 2).unwrap();
    let p = nc.define("pressure", &[z, c], "Pa").unwrap();
    let lat = nc.define("latCell", &[c], "radians").unwrap();
    let lon = nc.define("lonCell", &[c], "radians").unwrap();
    nc.putattr(-1, "valid_time_unix", "1").unwrap();
    nc.end().unwrap();
    nc.put(p, &[100., 100., 1000., 1000.]).unwrap();
    nc.put(lat, &[0., 0.]).unwrap();
    nc.put(lon, &[0., 0.017453292]).unwrap();
    nc.close().unwrap();
    let mut req = volume_request(&f, composition_row());
    req.pressure_file = Some(f.path("wrong-pressure.nc"));
    assert!(run(req).unwrap_err().contains("target pressure valid time"));
    let nc = File::open(&f.path("wrong-pressure.nc")).unwrap();
    let other = File::create(&f.path("other-mesh.nc")).unwrap();
    let cells = other.dim("nCells", 2).unwrap();
    let lat = other.define("latCell", &[cells], "radians").unwrap();
    let lon = other.define("lonCell", &[cells], "radians").unwrap();
    other.end().unwrap();
    other.put(lat, &[0., 0.]).unwrap();
    other.put(lon, &[1., 2.]).unwrap();
    other.close().unwrap();
    let points = vec![geo::xyz(0., 1.), geo::xyz(0., 2.)];
    let mut req = volume_request(&f, composition_row());
    req.pressure_file = Some(f.path("wrong-pressure.nc"));
    req.mesh = f.path("other-mesh.nc");
    assert!(
        composition(&req, &File::open(&f.path("source.nc")).unwrap(), &points)
            .err()
            .unwrap()
            .contains("pressure mesh coordinates")
    );
    drop(nc);
}

#[test]
fn surface_source_cannot_claim_another_mesh() {
    let _guard = NETCDF.lock().unwrap();
    let fixture = Fixture::new();
    fixture.mesh();
    let f = File::create(&fixture.path("source.nc")).unwrap();
    let t = f.dim("time", 1).unwrap();
    let c = f.dim("nCells", 2).unwrap();
    let time = f
        .define("time", &[t], "seconds since 1970-01-01 00:00:00")
        .unwrap();
    let lat = f.define("latCell", &[c], "radians").unwrap();
    let lon = f.define("lonCell", &[c], "radians").unwrap();
    let land = f.define("land", &[c], "1").unwrap();
    let u = f.define("u", &[c], "m s-1").unwrap();
    let v = f.define("v", &[c], "m s-1").unwrap();
    let h = f.define("h", &[c], "m").unwrap();
    f.end().unwrap();
    for (id, data) in [
        (time, vec![0.]),
        (lat, vec![0., 0.]),
        (lon, vec![0., 0.017453292]),
        (land, vec![1., 2.]),
        (u, vec![10., 10.]),
        (v, vec![0., 0.]),
        (h, vec![0., 0.]),
    ] {
        f.put(id, &data).unwrap();
    }
    f.close().unwrap();
    let row = json!({"id":"arbitrary-surface", "kind":"surface", "operator":"gocart-seasalt",
        "grid":{"time":"time"}, "fields":{
            "land":{"selector":"land","units":"1"}, "u10":{"selector":"u","units":"m s-1"},
            "v10":{"selector":"v","units":"m s-1"}, "surface_height":{"selector":"h","units":"m"}},
        "variables":[{"species":"salt", "radius_lo_um":0.1,"radius_hi_um":0.5,"density_kg_m3":2200}]});
    run(fixture.request(row.clone())).unwrap();
    let f = File::open(&fixture.path("output.nc")).unwrap();
    let values = f.read("salt", 0).unwrap().data;
    assert_eq!(values[0], 0.);
    assert!(values[1] > 0.);
    let receipt: Value =
        serde_json::from_slice(&fs::read(fixture.path("receipt.json")).unwrap()).unwrap();
    let rate = receipt["species"][0]["accounting"]["stored_emission_rate_kg_s"]
        .as_f64()
        .unwrap();
    assert!((rate - values[1] * 20.).abs() < 1e-20);

    drop(f);
    let f = File::open(&fixture.path("source.nc")).unwrap();
    let wrong = vec![geo::xyz(0., 1.), geo::xyz(0., 2.)];
    let other = File::create(&fixture.path("other-mesh.nc")).unwrap();
    let c = other.dim("nCells", 2).unwrap();
    let lat = other.define("latCell", &[c], "radians").unwrap();
    let lon = other.define("lonCell", &[c], "radians").unwrap();
    other.end().unwrap();
    other.put(lat, &[0., 0.]).unwrap();
    other.put(lon, &[1., 2.]).unwrap();
    other.close().unwrap();
    let mut request = fixture.request(row);
    request.mesh = fixture.path("other-mesh.nc");
    assert!(surface(&request, &f, &wrong)
        .err()
        .unwrap()
        .contains("surface source mesh coordinates"));
}

#[test]
fn native_physical_dry_eos_drives_composition_without_pressure_fields() {
    let _guard = NETCDF.lock().unwrap();
    let fixture = Fixture::new();
    fixture.mesh();
    volume_fixture(&fixture, 0.);
    let f = File::create(&fixture.path("eos.nc")).unwrap();
    let c = f.dim("nCells", 2).unwrap();
    let z = f.dim("nVertLevels", 2).unwrap();
    let lat = f.define("latCell", &[c], "radians").unwrap();
    let lon = f.define("lonCell", &[c], "radians").unwrap();
    let rho = f.define("rho", &[c, z], "kg m^{-3}").unwrap();
    let theta = f.define("theta", &[c, z], "K").unwrap();
    let qv = f.define("qv", &[c, z], "kg kg^{-1}").unwrap();
    f.putattr(-1, "valid_time_unix", "0").unwrap();
    f.end().unwrap();
    f.put(lat, &[0., 0.]).unwrap();
    f.put(lon, &[0., 0.017453292]).unwrap();
    // Invert EOS for an independent known pressure column, including moisture.
    let rvord = (461.6_f32 / 287.0_f32) as f64;
    let native_qv = [0.01_f32, -1e-7, -1e-5, 0.01];
    let density: Vec<f32> = [100_f64, 1099., 1099., 100.]
        .iter()
        .zip(native_qv)
        .map(|(p, q)| {
            ((p / 100000.).powf((1004.5 - 287.) / 1004.5) * 100000.
                / (287. * 300. * (1. + rvord * f64::from(q)))) as f32
        })
        .collect();
    f.put(rho, &density).unwrap();
    f.put(theta, &[300.; 4]).unwrap();
    f.put(qv, &native_qv).unwrap();
    f.close().unwrap();
    let mut row = composition_row();
    row["target_pressure"] = json!({"kind":"dry_moist_eos",
        "density":{"selector":"rho","units":"kg m^{-3}"},
        "potential_temperature":{"selector":"theta","units":"K"},
        "water_vapor":{"selector":"qv","units":"kg kg^{-1}"},
        "dry_gas_constant":287.,"vapor_gas_constant":461.6,
        "dry_heat_capacity":1004.5,"reference_pressure_pa":100000.});
    let mut req = volume_request(&fixture, row);
    req.pressure_file = Some(fixture.path("eos.nc"));
    req.pressure_variables.clear();
    run(req).unwrap();
    let out = File::open(&fixture.path("output.nc")).unwrap();
    for (&a, b) in out
        .read("arbitrary_species", 0)
        .unwrap()
        .data
        .iter()
        .zip([2.5e-9, 10e-9, 7.5e-9, 5e-9])
    {
        assert!((a / b - 1.).abs() < 1e-6, "{a} {b}");
    }
    let receipt: Value =
        serde_json::from_slice(&fs::read(fixture.path("receipt.json")).unwrap()).unwrap();
    assert_eq!(
        receipt["species"][0]["accounting"]["target_pressure"]["operator"]["kind"],
        "dry_moist_eos"
    );
    let accounting = &receipt["species"][0]["accounting"]["target_pressure"]["operator"]
        ["initialized_water_vapor"];
    assert_eq!(accounting["negative_values"], 2);
    assert_eq!(accounting["minimum_kg_kg_dry"], f64::from(native_qv[2]));

    // Finite moisture cannot admit an undefined native EOS.
    let invalid = File::create(&fixture.path("invalid-eos.nc")).unwrap();
    let c = invalid.dim("nCells", 2).unwrap();
    let z = invalid.dim("nVertLevels", 2).unwrap();
    let rho = invalid.define("rho", &[c, z], "kg m^{-3}").unwrap();
    let theta = invalid.define("theta", &[c, z], "K").unwrap();
    let qv = invalid.define("qv", &[c, z], "kg kg^{-1}").unwrap();
    invalid.end().unwrap();
    invalid.put(rho, &density).unwrap();
    invalid.put(theta, &[300.; 4]).unwrap();
    invalid.put(qv, &[-1.; 4]).unwrap();
    invalid.close().unwrap();
    let invalid = File::open(&fixture.path("invalid-eos.nc")).unwrap();
    let request = volume_request(&fixture, composition_row());
    let mut request = request;
    request.row["target_pressure"] =
        receipt["species"][0]["accounting"]["target_pressure"]["operator"].clone();
    request.pressure_variables.clear();
    assert!(target_pressure(&request, &invalid, 2)
        .unwrap_err()
        .contains("EOS argument must be finite and positive"));
}

#[test]
fn exact_text_clock_overrides_rounded_float_clock_and_selects_record() {
    let _guard = NETCDF.lock().unwrap();
    let fixture = Fixture::new();
    let f = File::create(&fixture.path("clock.nc")).unwrap();
    let t = f.dim("Time", 2).unwrap();
    let width = f.dim("StrLen", 19).unwrap();
    let numeric = f
        .define("Time", &[t], "seconds since 2000-01-01 00:00:00")
        .unwrap();
    let exact = f.define_text("xtime", &[t, width]).unwrap();
    f.end().unwrap();
    let first = chrono::NaiveDateTime::parse_from_str("2026-10-19_00:00:01", "%Y-%m-%d_%H:%M:%S")
        .unwrap()
        .and_utc()
        .timestamp();
    let origin = 946684800;
    f.put(numeric, &[(first - origin) as f32, 0.]).unwrap();
    f.put_text(exact, "2026-10-19_00:00:012026-10-19_01:00:01")
        .unwrap();
    f.close().unwrap();
    let f = File::open(&fixture.path("clock.nc")).unwrap();
    let numeric_row = json!({"grid":{"time":"Time"}});
    assert!(check_time(&f, &numeric_row, 0, first).is_err());
    assert!(check_time(&f, &numeric_row, 1, first + 3600).is_err());
    let row = json!({"grid":{"time_text":"xtime", "time_format":"%Y-%m-%d_%H:%M:%S"}});
    check_time(&f, &row, 0, first).unwrap();
    check_time(&f, &row, 1, first + 3600).unwrap();
    assert!(check_time(&f, &row, 1, first).is_err());
    assert!(check_time(&f, &row, 2, first).is_err());
}
