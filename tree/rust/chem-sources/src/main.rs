// Source preparation only. Transport and removal belong to chem_transport.
mod geo;
mod nc;
use nc::{Field, File};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, time::Instant};

struct Request {
    row: Value,
    input: String,
    mesh: String,
    output: String,
    receipt: String,
    valid_time: i64,
    time_index: usize,
    max_distance_m: f64,
    pressure_file: Option<String>,
    humidity_input: Option<String>,
    pressure_variables: Vec<String>,
    expected_species: Vec<String>,
    mesh_sha256: String,
    map_output: Option<String>,
    map_spacing_deg: f64,
    map_bounds: [f64; 4],
}
impl<'de> Deserialize<'de> for Request {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let mut m = Value::deserialize(deserializer)?
            .as_object()
            .cloned()
            .ok_or_else(|| D::Error::custom("request must be an object"))?;
        macro_rules! field {
            ($key:literal) => {
                serde_json::from_value(
                    m.remove($key)
                        .ok_or_else(|| D::Error::custom(concat!("missing request ", $key)))?,
                )
                .map_err(D::Error::custom)?
            };
            ($key:literal, $default:expr) => {
                match m.remove($key) {
                    Some(v) => serde_json::from_value(v).map_err(D::Error::custom)?,
                    None => $default,
                }
            };
        }
        let r = Self {
            row: field!("row"),
            input: field!("input"),
            mesh: field!("mesh"),
            output: field!("output"),
            receipt: field!("receipt"),
            valid_time: field!("valid_time"),
            time_index: field!("time_index", 0),
            max_distance_m: field!("max_distance_m", distance()),
            pressure_file: field!("pressure_file", None),
            humidity_input: field!("humidity_input", None),
            pressure_variables: field!("pressure_variables", Vec::new()),
            expected_species: field!("expected_species", Vec::new()),
            mesh_sha256: field!("mesh_sha256", String::new()),
            map_output: field!("map_output", None),
            map_spacing_deg: field!("map_spacing_deg", map_step()),
            map_bounds: field!("map_bounds", map_bounds()),
        };
        if !m.is_empty() {
            return Err(D::Error::custom(format!(
                "unknown request fields: {:?}",
                m.keys()
            )));
        }
        Ok(r)
    }
}
fn map_step() -> f64 {
    1.
}
fn map_bounds() -> [f64; 4] {
    [-180., 180., -90., 90.]
}
fn distance() -> f64 {
    200_000.
}
fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str, String> {
    v[key]
        .as_str()
        .ok_or_else(|| format!("missing string {key}"))
}
fn number(v: &Value, key: &str) -> Result<f64, String> {
    v[key]
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| format!("missing finite number {key}"))
}
fn require(ok: bool, message: &str) -> Result<(), String> {
    if ok {
        Ok(())
    } else {
        Err(message.into())
    }
}
fn units(f: &File, name: &str, wanted: &str) -> Result<(), String> {
    let got = f.attr(name, "units").unwrap_or_default();
    require(
        got.trim() == wanted,
        &format!("{name}: units {got:?} differ from declared {wanted:?}"),
    )
}
fn finite(values: &[f64], name: &str, positive: bool) -> Result<(), String> {
    require(values.iter().all(|v|v.is_finite()&&(!positive||*v>0.)),&format!("{name}: missing, non-finite or non-positive input; inventing a value would change source mass"))
}
fn read(f: &File, name: &str, time: usize) -> Result<Vec<f64>, String> {
    Ok(f.read(name, time)?.data)
}
fn check_time(f: &File, row: &Value, index: usize, expected: i64) -> Result<(), String> {
    if let Some(attribute) = row["grid"]["time_attribute"].as_str() {
        require(
            index == 0,
            "global coverage timestamp describes only one record",
        )?;
        let value = f
            .global_attr(attribute)
            .ok_or("declared global time attribute missing")?;
        let instant = chrono::DateTime::parse_from_rfc3339(&value).map_err(|e| e.to_string())?;
        return require(
            instant.timestamp() == expected && instant.timestamp_subsec_nanos() == 0,
            "source valid time differs from requested instant (global coverage attribute)",
        );
    }
    if let Some(name) = row["grid"]["time_text"].as_str() {
        let format = text(&row["grid"], "time_format")?;
        let value = f.record_text_at(name, index)?;
        let instant = chrono::NaiveDateTime::parse_from_str(&value, format)
            .map_err(|e| e.to_string())?
            .and_utc();
        return require(
            instant.timestamp_subsec_nanos() == 0
                && instant.timestamp() + row["grid"]["valid_time_offset_s"].as_i64().unwrap_or(0)
                    == expected,
            "source valid time differs from requested instant (text coordinate)",
        );
    }
    let name = text(&row["grid"], "time")?;
    let v = read(f, name, index)?;
    // Reading a coordinate with the Time dimension returns just the requested record.
    require(
        v.len() == 1,
        "time coordinate must select exactly one instant",
    )?;
    let u = f
        .attr(name, "units")
        .ok_or("time coordinate has no CF units")?;
    let (scale, origin) = u
        .split_once(" since ")
        .ok_or("time coordinate needs CF units since an origin")?;
    let scale = match scale {
        "seconds" => 1.,
        "minutes" => 60.,
        "hours" => 3600.,
        "days" => 86400.,
        _ => return Err(format!("unsupported CF time unit {scale}")),
    };
    let origin = origin.trim().trim_end_matches('Z').replace('T', " ");
    let parsed = chrono::NaiveDateTime::parse_from_str(&origin, "%Y-%m-%d %H:%M:%S")
        .or_else(|_| {
            chrono::NaiveDateTime::parse_from_str(&(origin + " 00:00:00"), "%Y-%m-%d %H:%M:%S")
        })
        .map_err(|e| e.to_string())?
        .and_utc()
        .timestamp();
    require(
        ((parsed as f64
            + scale * v[0]
            + row["grid"]["valid_time_offset_s"].as_f64().unwrap_or(0.))
            - expected as f64)
            .abs()
            < 0.5,
        "source valid time differs from requested instant; another hour cannot supply this source",
    )
}
fn coordinates(f: &File, row: &Value, time: usize) -> Result<(Vec<[f64; 3]>, Vec<usize>), String> {
    let latname = text(&row["grid"], "latitude")?;
    let lonname = text(&row["grid"], "longitude")?;
    let a = f.read(latname, time)?;
    let b = f.read(lonname, time)?;
    units(f, latname, text(&row["grid"], "latitude_units")?)?;
    units(f, lonname, text(&row["grid"], "longitude_units")?)?;
    finite(&a.data, "latitude", false)?;
    finite(&b.data, "longitude", false)?;
    let angle = text(&row["grid"], "angle")?;
    require(
        angle == "degrees" || angle == "radians",
        "coordinate angle must be degrees or radians",
    )?;
    let factor = if angle == "radians" {
        1.
    } else {
        std::f64::consts::PI / 180.
    };
    let mut out = Vec::new();
    let shape;
    if a.shape.len() == 1 && b.shape.len() == 1 && a.dims != b.dims {
        shape = vec![a.data.len(), b.data.len()];
        for &lat in &a.data {
            for &lon in &b.data {
                require(
                    (lat * factor).abs() <= std::f64::consts::FRAC_PI_2 + 1e-8,
                    "latitude out of range",
                )?;
                out.push(geo::xyz(lat * factor, lon * factor));
            }
        }
    } else {
        require(a.shape == b.shape, "latitude and longitude shape differ")?;
        shape = a.shape;
        for (&lat, &lon) in a.data.iter().zip(&b.data) {
            require(
                (lat * factor).abs() <= std::f64::consts::FRAC_PI_2 + 1e-8,
                "latitude out of range",
            )?;
            out.push(geo::xyz(lat * factor, lon * factor));
        }
    }
    Ok((out, shape))
}
fn source_areas(f: &File, row: &Value, time: usize, shape: &[usize]) -> Result<Vec<f64>, String> {
    if let Some(name) = row["grid"]["area"].as_str() {
        units(f, name, text(&row["grid"], "area_units")?)?;
        let factor = number(&row["grid"], "area_factor_m2")?;
        require(factor > 0., "area factor must be positive")?;
        return Ok(read(f, name, time)?
            .into_iter()
            .map(|v| v * factor)
            .collect());
    }
    require(
        row["grid"]["area_method"] == "regular-spherical",
        "flux requires declared source area or regular-spherical area method",
    )?;
    let lat = read(f, text(&row["grid"], "latitude")?, time)?;
    let lon = read(f, text(&row["grid"], "longitude")?, time)?;
    require(
        shape == [lat.len(), lon.len()] && lat.len() > 1 && lon.len() > 1,
        "spherical area requires a regular latitude/longitude grid",
    )?;
    let radians = std::f64::consts::PI / 180.;
    require(
        row["grid"]["angle"] == "degrees",
        "regular-spherical area method requires degree coordinates",
    )?;
    let dy = (lat[1] - lat[0]).abs();
    let dx = (lon[1] - lon[0]).abs();
    require(
        dx * lon.len() as f64 <= 360.00001,
        "source longitude grid duplicates a cyclic endpoint",
    )?;
    require(
        dy > 0.
            && dx > 0.
            && lat
                .windows(2)
                .all(|p| ((p[1] - p[0]).abs() - dy).abs() < 1e-4)
            && lon
                .windows(2)
                .all(|p| ((p[1] - p[0]).abs() - dx).abs() < 1e-4),
        "spherical area requires uniform coordinate spacing",
    )?;
    // CF cell centres at the poles are clipped to the physical latitude bounds.
    Ok(lat
        .iter()
        .flat_map(|y| {
            let lo = (y - dy / 2.).max(-90.) * radians;
            let hi = (y + dy / 2.).min(90.) * radians;
            let a = geo::R.powi(2) * dx * radians * (hi.sin() - lo.sin());
            std::iter::repeat_n(a, lon.len())
        })
        .collect())
}
fn mesh_area_factor(f: &File) -> Result<f64, String> {
    let unit = f
        .attr("areaCell", "units")
        .ok_or("mesh areaCell lacks units")?;
    require(
        matches!(unit.trim(), "m2" | "m^2" | "m**2"),
        "mesh area units must be square metres",
    )?;
    match f.global_number("sphere_radius") {
        Some(1.) => {
            require(
                f.global_attr("on_a_sphere").as_deref() == Some("YES"),
                "unit-radius mesh requires on_a_sphere=YES",
            )?;
            Ok(geo::R.powi(2))
        }
        Some(r) => {
            require(
                r.is_finite() && (6_000_000. ..7_000_000.).contains(&r),
                "mesh sphere radius must be one or a terrestrial radius in metres",
            )?;
            Ok(1.)
        }
        None => Ok(1.), // Explicit m2 areas on a mesh without a radius attribute.
    }
}
fn mesh(f: &File) -> Result<(Vec<[f64; 3]>, Vec<f64>), String> {
    let lat = read(f, "latCell", 0)?;
    let lon = read(f, "lonCell", 0)?;
    let mut area = read(f, "areaCell", 0)?;
    let factor = mesh_area_factor(f)?;
    area.iter_mut().for_each(|a| *a *= factor);
    require(
        lat.len() == lon.len() && lat.len() == area.len() && !area.is_empty(),
        "mesh coordinates and areas must fill nCells",
    )?;
    finite(&lat, "mesh latitude", false)?;
    finite(&lon, "mesh longitude", false)?;
    finite(&area, "mesh area", true)?;
    // MPAS native mesh coordinates are radians.
    require(
        lat.iter()
            .all(|v| v.abs() <= std::f64::consts::FRAC_PI_2 + 1e-8),
        "mesh latitude must be radians",
    )?;
    Ok((
        lat.iter().zip(lon).map(|(&a, b)| geo::xyz(a, b)).collect(),
        area,
    ))
}
struct Output {
    name: String,
    data: Vec<f32>,
    levels: usize,
    units: String,
    stats: Value,
}
fn emission(
    req: &Request,
    f: &File,
    points: &[[f64; 3]],
    area: &[f64],
) -> Result<Vec<Output>, String> {
    check_time(f, &req.row, req.time_index, req.valid_time)?;
    let (src, shape) = coordinates(f, &req.row, req.time_index)?;
    let plan = geo::Index::new(points).plan(&src);
    let n = area.len();
    let interval = number(&req.row, "interval_s")?;
    require(interval > 0., "source interval must be positive")?;
    let mut out = Vec::new();
    for spec in req.row["variables"]
        .as_array()
        .ok_or("variables must be rows")?
    {
        let name = text(spec, "species")?;
        let selector = text(spec, "selector")?;
        let unit = text(spec, "units")?;
        units(f, selector, unit)?;
        let field = f.read_policy(selector, req.time_index, spec["no_fire_sentinel"].as_f64())?;
        require(
            field.shape == shape,
            "emission field does not fill source coordinate grid",
        )?;
        let mut valid = vec![true; src.len()];
        for mask in spec["masks"].as_array().into_iter().flatten() {
            let values = read(f, text(mask, "selector")?, req.time_index)?;
            require(
                values.len() == valid.len(),
                "QA mask shape differs from emission grid",
            )?;
            for (i, v) in values.iter().enumerate() {
                valid[i] &= v.is_finite()
                    && *v >= number(mask, "minimum")?
                    && *v <= number(mask, "maximum")?;
            }
        }
        let flux = unit == "kg m-2 s-1" || unit == "kg s-1 m-2";
        let source_area = if flux {
            Some(source_areas(f, &req.row, req.time_index, &shape)?)
        } else {
            None
        };
        if let Some(ref a) = source_area {
            require(a.len() == src.len(), "source area does not fill grid")?;
            finite(a, "source area", true)?;
        }
        require(
            unit == "kg" || flux,
            "emissions require cell mass kg or flux kg m-2 s-1",
        )?;
        let sentinel = spec["no_fire_sentinel"].as_f64();
        let mut mass = vec![0.; n];
        let (mut total, mut masked, mut outside, mut missing) = (0., 0., 0., 0usize);
        for (i, &raw) in field.data.iter().enumerate() {
            if !raw.is_finite() {
                missing += 1;
                continue;
            }
            let value = if Some(raw) == sentinel { 0. } else { raw };
            require(
                value >= 0.,
                "undeclared negative emission; only a named no-fire sentinel can be zeroed",
            )?;
            let m = if let Some(ref a) = source_area {
                value * a[i] * interval
            } else {
                value
            };
            total += m;
            if !valid[i] {
                masked += m;
                continue;
            }
            let (cell, d) = plan[i];
            if d > req.max_distance_m {
                outside += m;
                continue;
            }
            mass[cell] += m;
        }
        // Missing cells in an emission product are not a declaration of clean air.
        require(
            missing == 0,
            &format!("{selector}: emission grid has {missing} missing cells; a missing value cannot silently become no fire"),
        )?;
        let mapped: f64 = mass.iter().sum();
        let data: Vec<f32> = mass
            .iter()
            .zip(area)
            .map(|(m, a)| (m / (a * interval)) as f32)
            .collect();
        let stored_mass: f64 = data
            .iter()
            .zip(area)
            .map(|(&v, &a)| v as f64 * a * interval)
            .sum();
        require(
            (stored_mass - mapped).abs() <= 1e-6 * mapped.max(1e-30),
            "float32 source output changed mass by more than one part per million",
        )?;
        out.push(Output{name:name.into(),data,levels:1,units:"kg m-2 s-1".into(),stats:json!({"source_mass_kg":total,"mapped_mass_kg":mapped,"stored_mass_kg":stored_mass,"stored_mass_residual_kg":mapped-stored_mass,"masked_mass_kg":masked,"outside_mass_kg":outside,"mass_residual_kg":total-mapped-masked-outside,"interval_s":interval,"mapping":"nearest destination Voronoi cell; extensive sum"})});
    }
    Ok(out)
}
fn level_cell(field: Field, nc: usize) -> Result<(usize, Vec<f64>), String> {
    require(
        field.shape.len() == 2,
        "target pressure must be a cell/level matrix",
    )?;
    if field.dims[0] == "nCells" {
        let nl = field.shape[1];
        require(field.shape[0] == nc, "target pressure cell count differs")?;
        let out = (0..nl * nc)
            .map(|i| field.data[(i % nc) * nl + i / nc])
            .collect();
        Ok((nl, out))
    } else {
        require(field.shape[1] == nc, "target pressure cell count differs")?;
        Ok((field.shape[0], field.data))
    }
}
fn interpolate(p: &[f64], q: &[f64], target: f64) -> f64 {
    // Source p strictly increases from top to bottom. Constant endpoint extension.
    if target <= p[0] {
        return q[0];
    }
    if target >= p[p.len() - 1] {
        return q[q.len() - 1];
    }
    let hi = p.partition_point(|x| *x < target);
    let lo = hi - 1;
    let w = (target.ln() - p[lo].ln()) / (p[hi].ln() - p[lo].ln());
    q[lo] * (1. - w) + q[hi] * w
}
// Prepared native files store coordinates as float32 even when the mesh is float64.
// Compare original angular coordinates at that precision, without a distance waiver.
fn matching_mesh(req: &Request, f: &File, role: &str) -> Result<(), String> {
    let original = File::open(&req.mesh)?;
    for name in ["latCell", "lonCell"] {
        let a = read(&original, name, 0)?;
        let b = read(f, name, 0)?;
        require(
            a.len() == b.len()
                && a.iter().zip(&b).all(|(&x, &y)| {
                    x.is_finite() && y.is_finite() && (x as f32).to_bits() == (y as f32).to_bits()
                }),
            &format!("{role} mesh coordinates differ from requested grid"),
        )?;
    }
    Ok(())
}
fn target_pressure(req: &Request, f: &File, nc: usize) -> Result<(usize, Vec<f64>, Value), String> {
    let mut target = Vec::new();
    let mut nl = 0;
    for variable in &req.pressure_variables {
        units(f, variable, "Pa")?;
        let (levels, p) = level_cell(f.read(variable, 0)?, nc)?;
        if target.is_empty() {
            target = p;
            nl = levels;
        } else {
            require(levels == nl, "target pressure component levels differ")?;
            target.par_iter_mut().zip(p).for_each(|(a, b)| *a += b);
        }
    }
    let operator = if !req.pressure_variables.is_empty() {
        json!({"kind":"pressure_sum", "selectors":req.pressure_variables})
    } else {
        let config = &req.row["target_pressure"];
        require(
            text(config, "kind")? == "dry_moist_eos",
            "unknown target pressure operator",
        )?;
        let mut fields = Vec::new();
        for key in ["density", "potential_temperature", "water_vapor"] {
            let spec = &config[key];
            let name = text(spec, "selector")?;
            units(f, name, text(spec, "units")?)?;
            let (levels, data) = level_cell(f.read(name, 0)?, nc)?;
            if nl == 0 {
                nl = levels;
            }
            require(levels == nl, "EOS field levels differ")?;
            finite(&data, key, key != "water_vapor")?;
            fields.push(data);
        }
        let rgas = number(config, "dry_gas_constant")?;
        let rv = number(config, "vapor_gas_constant")?;
        let cp = number(config, "dry_heat_capacity")?;
        let p0 = number(config, "reference_pressure_pa")?;
        require(
            rgas > 0. && rv > 0. && cp > rgas && p0 > 0.,
            "invalid EOS constants",
        )?;
        // Match the native compile-time REAL(RKIND) division; rho is physical
        // dry density, theta is dry theta, and qv is kg vapor per kg dry air.
        let rvord = ((rv as f32) / (rgas as f32)) as f64;
        target = (0..fields[0].len())
            .into_par_iter()
            .map(|i| {
                let argument =
                    rgas / p0 * fields[0][i] * fields[1][i] * (1. + rvord * fields[2][i]);
                // Native initialized qv may be slightly negative before the
                // physics clamp. Preserve that pressure authority unchanged;
                // an undefined EOS, rather than the sign of qv, is the refusal.
                require(
                    argument.is_finite() && argument > 0.,
                    "initialized EOS argument must be finite and positive",
                )?;
                Ok(p0 * argument.powf(cp / (cp - rgas)))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut operator = config.clone();
        operator["initialized_water_vapor"] = json!({
            "policy":"preserve_finite_native_values_positive_eos_argument",
            "negative_values":fields[2].par_iter().filter(|&&q| q < 0.).count(),
            "minimum_kg_kg_dry":fields[2].par_iter().copied().reduce(|| f64::INFINITY, f64::min),
        });
        operator
    };
    require(
        nl > 0,
        "composition needs target pressure variables or a declared EOS",
    )?;
    finite(&target, "target pressure", true)?;
    let range = target
        .iter()
        .fold([f64::INFINITY, f64::NEG_INFINITY], |r, &p| {
            [r[0].min(p), r[1].max(p)]
        });
    Ok((nl, target, json!({"operator":operator,"range_pa":range})))
}
fn composition(req: &Request, f: &File, points: &[[f64; 3]]) -> Result<Vec<Output>, String> {
    check_time(f, &req.row, req.time_index, req.valid_time)?;
    let (src, shape) = coordinates(f, &req.row, req.time_index)?;
    let ns = src.len();
    let nc = points.len();
    let plan = geo::Index::new(&src).plan(points);
    require(
        plan.iter().all(|v| v.1 <= req.max_distance_m),
        "composition source does not cover every target cell",
    )?;
    let pressure = File::open(
        req.pressure_file
            .as_ref()
            .ok_or("composition needs pressure_file")?,
    )?;
    // Pressure is an initialization authority, not an arbitrary same-size matrix.
    matching_mesh(req, &pressure, "target pressure")?;
    let pressure_time = if let Some(value) = pressure.global_attr("valid_time_unix") {
        value.parse::<i64>().map_err(|e| e.to_string())?
    } else {
        chrono::NaiveDateTime::parse_from_str(&pressure.record_text("xtime")?, "%Y-%m-%d_%H:%M:%S")
            .map_err(|e| e.to_string())?
            .and_utc()
            .timestamp()
    };
    require(
        pressure_time == req.valid_time,
        "target pressure valid time differs from composition",
    )?;
    let (nl, target, pressure_provenance) = target_pressure(req, &pressure, nc)?;
    let dpname = text(&req.row["vertical"], "thickness")?;
    units(f, dpname, "Pa")?;
    let dp = f.read(dpname, req.time_index)?;
    require(
        dp.shape.len() == shape.len() + 1 && dp.shape[1..] == shape,
        "pressure thickness must have level followed by source grid",
    )?;
    let sl = dp.shape[0];
    let level_name = text(&req.row["vertical"], "coordinate")?;
    let source_levels = read(f, level_name, req.time_index)?;
    require(
        source_levels.len() == sl
            && source_levels
                .iter()
                .enumerate()
                .all(|(i, v)| (*v - (i + 1) as f64).abs() < 1e-6),
        "composition levels must explicitly run top-to-bottom 1..N",
    )?;
    finite(&dp.data, "pressure thickness", true)?;
    let top = number(&req.row["vertical"], "top_pa")?;
    require(top > 0., "source top pressure must be positive")?;
    require(sl > 1, "composition requires at least two vertical levels")?;
    let basis = text(&req.row, "source_basis")?;
    require(
        basis == "dry_air" || basis == "moist_air",
        "source basis must explicitly be dry_air or moist_air",
    )?;
    require(
        (basis == "moist_air") == req.row["humidity"].is_string(),
        "moist-air composition needs specific humidity; dry-air source must not be corrected twice",
    )?;
    let humidity = if let Some(q) = req.row["humidity"].as_str() {
        let auxiliary =
            File::open(req.humidity_input.as_ref().ok_or(
                "moist-air source needs humidity_input with same-time specific humidity",
            )?)?;
        check_time(&auxiliary, &req.row, req.time_index, req.valid_time)?;
        let (coordinates, aux_shape) = coordinates(&auxiliary, &req.row, req.time_index)?;
        require(
            aux_shape == shape && coordinates == src,
            "specific humidity grid differs from composition grid",
        )?;
        units(&auxiliary, q, "kg kg-1")?;
        let field = auxiliary.read(q, req.time_index)?;
        require(
            read(&auxiliary, level_name, req.time_index)? == source_levels,
            "specific humidity level coordinate differs from composition",
        )?;
        require(
            field.shape == dp.shape,
            "specific humidity vertical layout differs from composition",
        )?;
        Some(field.data)
    } else {
        None
    };
    if let Some(ref q) = humidity {
        require(
            q.len() == dp.data.len() && q.iter().all(|x| x.is_finite() && *x >= 0. && *x < 1.),
            "humidity must fill source levels and lie in [0,1)",
        )?;
    }
    let mut out = Vec::new();
    for spec in req.row["variables"]
        .as_array()
        .ok_or("variables must be rows")?
    {
        let mut q = vec![0.; sl * ns];
        for component in spec["components"]
            .as_array()
            .ok_or("composition components must be a list")?
        {
            let name = text(component, "selector")?;
            units(f, name, text(component, "units")?)?;
            let v = f.read(name, req.time_index)?;
            require(
                v.shape == dp.shape,
                "aerosol component shape differs from pressure",
            )?;
            finite(&v.data, name, false)?;
            require(
                v.data.iter().all(|x| *x >= 0.),
                "negative composition mixing ratio",
            )?;
            let factor = number(component, "factor")?;
            require(factor >= 0., "composition factor cannot be negative")?;
            q.par_iter_mut()
                .zip(v.data)
                .for_each(|(a, b)| *a += b * factor);
        }
        let columns: Vec<Vec<f32>> = (0..nc)
            .into_par_iter()
            .map(|c| {
                let s = plan[c].0;
                let mut p = Vec::with_capacity(sl);
                let mut values = Vec::with_capacity(sl);
                let mut half = top;
                for k in 0..sl {
                    let delta = dp.data[k * ns + s];
                    p.push(half + delta * 0.5);
                    half += delta;
                    let v = q[k * ns + s];
                    values.push(if let Some(ref h) = humidity {
                        v / (1. - h[k * ns + s])
                    } else {
                        v
                    });
                }
                (0..nl)
                    .map(|k| interpolate(&p, &values, target[k * nc + c]) as f32)
                    .collect()
            })
            .collect();
        let data = (0..nl * nc).map(|i| columns[i % nc][i / nc]).collect();
        out.push(Output{name:text(spec,"species")?.into(),data,levels:nl,units:"kg kg-1".into(),stats:json!({"basis":"dry_air","horizontal":"spherical nearest source cell","vertical":"linear in log pressure, constant endpoint extension","source_levels":sl,"target_levels":nl,"target_pressure":pressure_provenance})});
    }
    Ok(out)
}
mod surface;
fn surface(req: &Request, f: &File, points: &[[f64; 3]]) -> Result<Vec<Output>, String> {
    check_time(f, &req.row, req.time_index, req.valid_time)?;
    let nc = points.len();
    matching_mesh(req, f, "surface source")?;
    let mut fields = BTreeMap::new();
    for (key, value) in req.row["fields"]
        .as_object()
        .ok_or("surface source needs field declarations")?
    {
        let name = text(value, "selector")?;
        units(f, name, text(value, "units")?)?;
        let data = read(f, name, req.time_index)?;
        require(data.len() == nc, "surface field must fill nCells")?;
        finite(&data, key, false)?;
        fields.insert(key.as_str(), data);
    }
    let get = |name: &str, c: usize| -> Result<f64, String> {
        fields
            .get(name)
            .map(|v| v[c])
            .ok_or_else(|| format!("surface source missing {name}"))
    };
    let mut out = Vec::new();
    for row in req.row["variables"]
        .as_array()
        .ok_or("variables must be rows")?
    {
        let data: Result<Vec<f32>, String> = (0..nc)
            .into_par_iter()
            .map(|c| {
                let land = get("land", c)?;
                require(
                    land == 1. || land == 2.,
                    "surface land flag must be 1 (land) or 2 (water)",
                )?;
                let wind = get("u10", c)?.hypot(get("v10", c)?);
                let source = match text(&req.row, "operator")? {
                    "gocart-dust" => {
                        if land < 1.5 {
                            let soil = get("soil", c)?;
                            require(soil.fract() == 0., "soil type must be integer")?;
                            surface::dust(
                                number(row, "density_kg_m3")?,
                                number(row, "radius_m")?,
                                number(row, "fraction")?,
                                number(row, "coefficient")?,
                                get(text(row, "erodibility_field")?, c)?,
                                soil as usize,
                                get("soil_moisture", c)?,
                                get("air_density", c)?,
                                wind,
                            )?
                        } else {
                            0.
                        }
                    }
                    "gocart-seasalt" => {
                        if land > 1.5 && get("surface_height", c)? < 1e-3 {
                            surface::salt(
                                number(row, "radius_lo_um")?,
                                number(row, "radius_hi_um")?,
                                number(row, "density_kg_m3")?,
                                wind,
                            )?
                        } else {
                            0.
                        }
                    }
                    _ => return Err("unknown surface emission operator".into()),
                };
                require(
                    source.is_finite() && source >= 0.,
                    "surface source is non-finite or negative",
                )?;
                Ok(source as f32)
            })
            .collect();
        out.push(Output{name:text(row,"species")?.into(),data:data?,levels:1,units:"kg m-2 s-1".into(),stats:json!({"operator":req.row["operator"],"wind":"10 m wind","source":"WRF v4.7.1 GOCART dust or Monahan old sea-salt arm"})});
    }
    Ok(out)
}
fn write_map(req: &Request, points: &[[f64; 3]], outputs: &[Output]) -> Result<(), String> {
    let Some(path) = &req.map_output else {
        return Ok(());
    };
    require(
        outputs.iter().all(|o| o.levels == 1),
        "a source map requires surface flux outputs, not an arbitrary composition level",
    )?;
    let [west, east, south, north] = req.map_bounds;
    let step = req.map_spacing_deg;
    require(
        step.is_finite()
            && step > 0.
            && [west, east, south, north].iter().all(|v| v.is_finite())
            && west < east
            && east - west <= 360.
            && south >= -90.
            && north <= 90.
            && south < north,
        "invalid map spacing or geographic bounds",
    )?;
    let nx = ((east - west) / step).ceil() as usize;
    let ny = ((north - south) / step).floor() as usize + 1;
    require(
        nx > 1 && ny > 1 && nx.checked_mul(ny).is_some_and(|n| n <= 10_000_000),
        "source map requires 2..10000000 pixels",
    )?;
    let lat: Vec<f32> = (0..ny).map(|i| (south + i as f64 * step) as f32).collect();
    let lon: Vec<f32> = (0..nx).map(|i| (west + i as f64 * step) as f32).collect();
    let geographic: Vec<_> = lat
        .iter()
        .flat_map(|y| {
            lon.iter()
                .map(move |x| geo::xyz((*y as f64).to_radians(), (*x as f64).to_radians()))
        })
        .collect();
    let plan = geo::Index::new(points).plan(&geographic);
    let f = File::create(path)?;
    let t = f.dim("time", 1)?;
    let y = f.dim("lat", ny)?;
    let x = f.dim("lon", nx)?;
    let tv = f.define(
        "time",
        &[t],
        &format!(
            "seconds since {}",
            chrono::DateTime::from_timestamp(req.valid_time, 0)
                .ok_or("invalid map UTC time")?
                .format("%Y-%m-%d %H:%M:%S")
        ),
    )?;
    f.putattr(tv, "standard_name", "time")?;
    let a = f.define("lat", &[y], "degrees_north")?;
    f.putattr(a, "standard_name", "latitude")?;
    let b = f.define("lon", &[x], "degrees_east")?;
    f.putattr(b, "standard_name", "longitude")?;
    f.putattr(-1,"source",&format!("Hex cell-mean source flux; {}; nearest-cell display sampling, not new source resolution",text(&req.row,"id")?))?;
    let vars: Vec<_> = outputs
        .iter()
        .map(|o| f.define(&o.name, &[t, y, x], &o.units))
        .collect::<Result<_, _>>()?;
    f.end()?;
    f.put(tv, &[0.])?;
    f.put(a, &lat)?;
    f.put(b, &lon)?;
    for (o, v) in outputs.iter().zip(vars) {
        let data: Vec<f32> = plan
            .par_iter()
            .map(|&(c, d)| {
                if d <= req.max_distance_m {
                    o.data[c]
                } else {
                    f32::NAN
                }
            })
            .collect();
        f.put(v, &data)?;
    }
    f.close()
}
fn run(req: Request) -> Result<(), String> {
    let begin = Instant::now();
    require(
        req.max_distance_m.is_finite() && req.max_distance_m > 0.,
        "max distance must be positive",
    )?;
    let meshfile = File::open(&req.mesh)?;
    let area_factor = mesh_area_factor(&meshfile)?;
    let (points, area) = mesh(&meshfile)?;
    drop(meshfile);
    let f = File::open(&req.input)?;
    let mut outputs = match text(&req.row, "kind")? {
        "emission" => emission(&req, &f, &points, &area)?,
        "composition" => composition(&req, &f, &points)?,
        "surface" => surface(&req, &f, &points)?,
        x => return Err(format!("unknown source kind {x}")),
    };
    require(!outputs.is_empty(), "source row yields no species")?;
    require(
        outputs
            .iter()
            .all(|o| o.data.iter().all(|v| v.is_finite() && *v >= 0.)),
        "normalized float32 source contains non-finite or negative values",
    )?;
    let mut names = std::collections::BTreeSet::new();
    for o in &outputs {
        require(
            names.insert(o.name.as_str()),
            "duplicate output species would double count mass",
        )?;
        require(
            o.data.iter().all(|v| v.is_finite() && *v >= 0.),
            "output contains invalid concentration or source flux",
        )?;
    }
    if !req.expected_species.is_empty() {
        let wanted: std::collections::BTreeSet<_> =
            req.expected_species.iter().map(String::as_str).collect();
        require(
            names == wanted,
            "source species differ from requested transport species",
        )?;
    }
    if req.row["kind"] == "surface" {
        for output in &mut outputs {
            output.stats["stored_emission_rate_kg_s"] = json!(output
                .data
                .par_iter()
                .zip(&area)
                .map(|(&flux, &a)| flux as f64 * a)
                .sum::<f64>());
            output.stats["rate_basis"] = json!("instantaneous native cell-mean flux times physical cell area; no interval mass inferred");
        }
    }
    let output = File::create(&req.output)?;
    let cell = output.dim("nCells", area.len())?;
    let levels = outputs.iter().map(|v| v.levels).max().unwrap();
    let level = output.dim("nVertLevels", levels)?;
    output.putattr(-1, "schema", "gpuwm-hex.chem-source-frame/v1")?;
    output.putattr(-1, "source_id", text(&req.row, "id")?)?;
    output.putattr(-1, "valid_time_unix", &req.valid_time.to_string())?;
    output.putattr(-1, "basis", "dry_air")?;
    output.putattr(-1, "mesh_sha256", &req.mesh_sha256)?;
    let mut vars = Vec::new();
    for o in &outputs {
        let dims = if o.levels == 1 && req.row["kind"] != "composition" {
            vec![cell]
        } else {
            vec![level, cell]
        };
        vars.push(output.define(&o.name, &dims, &o.units)?);
    }
    output.end()?;
    for (o, v) in outputs.iter().zip(vars) {
        output.put(v, &o.data)?;
    }
    output.close()?;
    write_map(&req, &points, &outputs)?;
    let stats:Vec<_>=outputs.iter().map(|o|json!({"species":o.name,"units":o.units,"levels":o.levels,"minimum":o.data.iter().copied().fold(f32::INFINITY,f32::min),"maximum":o.data.iter().copied().fold(0.,f32::max),"accounting":o.stats})).collect();
    fs::write(&req.receipt,serde_json::to_vec_pretty(&json!({"schema":"gpuwm-hex.chem-source-receipt/v1","source_id":req.row["id"],"input":req.input,"mesh":req.mesh,"output":req.output,"valid_time_unix":req.valid_time,"cells":area.len(),"mesh_area_factor_m2":area_factor,"mesh_area_sum_m2":area.iter().sum::<f64>(),"earth_radius_m":geo::R,"threads":rayon::current_num_threads(),"wall_s":begin.elapsed().as_secs_f64(),"species":stats})).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    Ok(())
}
fn main() {
    let result = (|| {
        let path = std::env::args()
            .nth(1)
            .ok_or("usage: rw_hex_chem_sources REQUEST.json")?;
        let request: Request = serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        run(request)
    })();
    if let Err(e) = result {
        eprintln!("hex chemistry sources: {e}");
        std::process::exit(1)
    }
}
#[cfg(test)]
mod tests;
