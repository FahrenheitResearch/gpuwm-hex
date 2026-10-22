use crate::{finite, license, load, nc::File, save};
use chrono::{Duration, NaiveDate, NaiveDateTime, Timelike};
use rayon::prelude::*;
use serde_json::json;
use std::{
    collections::{BTreeMap, VecDeque},
    path::Path,
    time::Instant,
};

const TIME: &str = "%Y-%m-%d_%H:%M:%S";
struct Plan {
    binding: String,
    reference: String,
    output: String,
    research_data_license: Option<String>,
}
struct Binding {
    schema: String,
    start: String,
    days: usize,
    cadence_hours: usize,
    grid: String,
    members: Vec<Member>,
    binding_sha256: String,
}
struct Member {
    member: String,
    frames: Vec<Frame>,
}
struct Frame {
    valid_time: String,
    path: String,
    bytes: u64,
    mtime_ns: Option<u128>,
}
struct Reference {
    schema: String,
    source: String,
    license: String,
    training_period: String,
    basis_id: String,
    daily_interval_start_hour: u32,
    method: String,
    raw_fields: BTreeMap<String, String>,
    definitions: Vec<Definition>,
}
struct Definition {
    id: String,
    fields: Vec<String>,
    domain: [f64; 4],
    nx: usize,
    ny: usize,
    standard_deviation: Vec<f64>,
    eofs: Vec<Vec<f64>>,
    pc_standard_deviation: Vec<f64>,
    phase_order: Vec<usize>,
    eof_modes: [usize; 2],
    seasonal: BTreeMap<String, Vec<f64>>,
    prior_anomalies: BTreeMap<String, Vec<f64>>,
    minimum_bin_area_fraction: f64,
    feature_mask: Option<Vec<bool>>,
}
crate::deserialize_fields!(Plan {
    binding,
    reference,
    output,
    research_data_license
});
crate::deserialize_fields!(Binding {
    schema,
    start,
    days,
    cadence_hours,
    grid,
    members,
    binding_sha256
});
crate::deserialize_fields!(Member { member, frames });
crate::deserialize_fields!(Frame {
    valid_time,
    path,
    bytes,
    mtime_ns
});
crate::deserialize_fields!(Reference {
    schema,
    source,
    license,
    training_period,
    basis_id,
    daily_interval_start_hour,
    method,
    raw_fields,
    definitions
});
crate::deserialize_fields!(Definition {
    id,
    fields,
    domain,
    nx,
    ny,
    standard_deviation,
    eofs,
    pc_standard_deviation,
    phase_order,
    eof_modes,
    seasonal,
    prior_anomalies,
    minimum_bin_area_fraction,
    feature_mask
});

pub fn log_pressure(p: &[f64], u: &[f64], target: f64) -> f64 {
    if p.len() != u.len() || p.len() < 2 || p.windows(2).any(|v| v[0] <= v[1]) {
        return f64::NAN;
    }
    for k in 0..p.len() - 1 {
        if p[k] >= target && p[k + 1] <= target {
            let w = (target.ln() - p[k].ln()) / (p[k + 1].ln() - p[k].ln());
            return u[k] + w * (u[k + 1] - u[k]);
        }
    }
    f64::NAN
}

fn native(file: &File, name: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = file.read(name)?;
    if a.shape == [n] {
        a.layout(&["nCells"], &[n])?;
    } else {
        a.layout(&["Time", "nCells"], &[1, n])?;
    }
    Ok(a.values)
}

type Groups = Vec<Vec<(usize, f64)>>;
fn groups(d: &Definition, lat: &[f64], lon: &[f64], area: &[f64]) -> Result<Groups, String> {
    let mut bins = vec![Vec::new(); d.nx * d.ny];
    for i in 0..lat.len() {
        let y = lat[i].to_degrees();
        let x = lon[i].to_degrees().rem_euclid(360.);
        if y >= d.domain[0] && y <= d.domain[1] && x >= d.domain[2] && x < d.domain[3] {
            let iy = (((y - d.domain[0]) / (d.domain[1] - d.domain[0]) * d.ny as f64) as usize)
                .min(d.ny - 1);
            let ix = (((x - d.domain[2]) / (d.domain[3] - d.domain[2]) * d.nx as f64) as usize)
                .min(d.nx - 1);
            bins[iy * d.nx + ix].push((i, area[i]));
        }
    }
    for (i, bin) in bins.iter_mut().enumerate() {
        let total: f64 = bin.iter().map(|v| v.1).sum();
        if total <= 0. {
            return Err(format!(
                "{} empty spatial bin {i}; use a reference supported by the native mesh",
                d.id
            ));
        }
        for v in bin {
            v.1 /= total;
        }
    }
    Ok(bins)
}

fn validate(d: &Definition) -> Result<(), String> {
    let (fields, domain, ny) = match d.id.as_str() {
        "rmm" => (vec!["olr", "u850", "u200"], [-15., 15., 0., 360.], Some(1)),
        "bsiso1" | "bsiso2" => (vec!["olr", "u850"], [-10., 40., 40., 160.], None),
        _ => return Err(format!("unknown diagnostic {}", d.id)),
    };
    if d.fields.iter().map(String::as_str).collect::<Vec<_>>() != fields
        || d.domain != domain
        || ny.map_or(false, |n| d.ny != n)
        || d.nx < 4
        || d.ny == 0
        || d.nx * d.ny > 100_000
        || !(0.8..=1.).contains(&d.minimum_bin_area_fraction)
    {
        return Err(format!("{} invalid fields/domain/spatial grid", d.id));
    }
    let n = d.nx * d.ny * d.fields.len();
    if d.feature_mask.as_ref().map_or(false, |mask| mask.len() != n) {
        return Err(format!("{} invalid fixed observation feature mask", d.id));
    }
    if d.standard_deviation.len() != d.fields.len()
        || d.standard_deviation
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.)
        || d.pc_standard_deviation.len() != 2
        || d.pc_standard_deviation
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.)
        || d.eofs.len() != 2
        || d.eofs.iter().any(|v| v.len() != n)
        || d.phase_order.len() != 8
        || {
            let mut p = d.phase_order.clone();
            p.sort();
            p != (1..=8).collect::<Vec<_>>()
        }
    {
        return Err(format!("{} invalid EOF/scaling/phase dimensions", d.id));
    }
    if d.eof_modes != if d.id == "bsiso2" { [3, 4] } else { [1, 2] } {
        return Err(format!("{} has wrong EOF mode numbers", d.id));
    }
    for eof in &d.eofs {
        finite(eof, "EOF")?;
        let norm: f64 = eof.iter().map(|v| v * v).sum();
        if (norm - 1.).abs() > 1e-3 {
            return Err(format!("{} EOF is not unit norm", d.id));
        }
    }
    if d.eofs[0]
        .iter()
        .zip(&d.eofs[1])
        .map(|(a, b)| a * b)
        .sum::<f64>()
        .abs()
        > 1e-3
    {
        return Err(format!("{} EOFs are not orthogonal", d.id));
    }
    for (date, values) in d.seasonal.iter().chain(&d.prior_anomalies) {
        NaiveDate::parse_from_str(date, "%Y-%m-%d").map_err(|e| e.to_string())?;
        if values.len() != n {
            return Err(format!(
                "{} {date} background has wrong feature dimension",
                d.id
            ));
        }
        finite(values, "reference background")?;
    }
    Ok(())
}

fn features(
    frame: &Frame,
    reference: &Reference,
    lat: &[f64],
    lon: &[f64],
    bins: &[Groups],
) -> Result<(Vec<Vec<f64>>, bool), String> {
    let n = lat.len();
    let metadata = std::fs::metadata(&frame.path).map_err(|e| e.to_string())?;
    if metadata.len() != frame.bytes {
        return Err(format!("bound history changed: {}", frame.path));
    }
    if let Some(expected) = frame.mtime_ns {
        let actual = metadata
            .modified()
            .map_err(|e| e.to_string())?
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_nanos();
        if actual != expected {
            return Err(format!(
                "bound history modification time changed: {}",
                frame.path
            ));
        }
    }
    let file = File::open(&frame.path)?;
    if file.clock()? != frame.valid_time {
        return Err(format!(
            "native xtime differs from bound time: {}",
            frame.path
        ));
    }
    for (name, expected) in [("latCell", lat), ("lonCell", lon)] {
        let actual = native(&file, name, n)?;
        // Forecast snapshots retain float32 coordinates and can use the
        // signed longitude convention while the source mesh uses [0, 2*pi).
        // Compare exact float32 representations, including circular aliases;
        // do not relax the cell-order check with a spatial tolerance.
        if actual
            .iter()
            .zip(expected)
            .any(|(a, b)| !same_coordinate(*a, *b, name == "lonCell"))
        {
            return Err(format!(
                "{name} cell order differs from bound mesh: {}",
                frame.path
            ));
        }
    }
    let field = |id: &str| {
        reference
            .raw_fields
            .get(id)
            .map(String::as_str)
            .ok_or_else(|| format!("reference raw_fields lacks {id}"))
    };
    let olr_array = file.read(field("olr")?)?;
    olr_array.layout(&["Time", "nCells"], &[1, n])?;
    olr_array.units(&["W m-2", "W m^-2", "W m^{-2}", "W/m2"])?;
    let olr = olr_array.values;
    // A globally zero outgoing longwave flux is the allocated cold-start
    // diagnostic, before the first radiation call, not a physical sample.
    let uninitialized_olr = olr.iter().all(|&v| v == 0.);
    let pressure = file.read(field("pressure")?)?;
    pressure.units(&["Pa"])?;
    let wind = file.read(field("u")?)?;
    wind.units(&["m s-1", "m s^-1", "m s^{-1}", "m/s"])?;
    let nz = *pressure.shape.last().ok_or("pressure rank")?;
    pressure.layout(&["Time", "nCells", "nVertLevels"], &[1, n, nz])?;
    wind.layout(&["Time", "nCells", "nVertLevels"], &[1, n, nz])?;
    let u850: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            log_pressure(
                &pressure.values[i * nz..(i + 1) * nz],
                &wind.values[i * nz..(i + 1) * nz],
                85000.,
            )
        })
        .collect();
    let u200: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            log_pressure(
                &pressure.values[i * nz..(i + 1) * nz],
                &wind.values[i * nz..(i + 1) * nz],
                20000.,
            )
        })
        .collect();
    bins.iter()
        .zip(&reference.definitions)
        .map(|(groups, d)| {
            let mut result = Vec::new();
            for (field_index, id) in d.fields.iter().enumerate() {
                let values = match id.as_str() {
                    "olr" => &olr,
                    "u850" => &u850,
                    "u200" => &u200,
                    _ => unreachable!(),
                };
                // Pressure surfaces below terrain are absent, not zero winds.
                // Renormalize only bins meeting the reference's declared support.
                let reduced: Vec<f64> = groups
                    .par_iter()
                    .enumerate()
                    .map(|(bin_index, g)| {
                        if d.feature_mask.as_ref().map_or(false, |mask| !mask[field_index * d.nx * d.ny + bin_index]) {
                            return 0.;
                        }
                        let available: f64 = g
                            .iter()
                            .filter(|&&(i, _)| values[i].is_finite())
                            .map(|v| v.1)
                            .sum();
                        if available + 1e-12 < d.minimum_bin_area_fraction {
                            f64::NAN
                        } else {
                            g.iter()
                                .filter(|&&(i, _)| values[i].is_finite())
                                .map(|&(i, w)| values[i] * w)
                                .sum::<f64>()
                                / available
                        }
                    })
                    .collect();
                finite(&reduced, &format!("{} {} {}", frame.path, d.id, id))?;
                result.extend(reduced);
            }
            Ok(result)
        })
        .collect::<Result<Vec<_>, String>>()
        .map(|values| (values, uninitialized_olr))
}

fn same_coordinate(actual: f64, expected: f64, longitude: bool) -> bool {
    actual.is_finite() && expected.is_finite() &&
        (actual as f32 == expected as f32 || longitude &&
            (actual as f32 == (expected - std::f64::consts::TAU) as f32 ||
             actual as f32 == (expected + std::f64::consts::TAU) as f32))
}

pub fn project(values: &[f64], eofs: &[Vec<f64>], scales: &[f64]) -> [f64; 2] {
    let dot = |k: usize| values.iter().zip(&eofs[k]).map(|(v, e)| v * e).sum::<f64>() / scales[k];
    [dot(0), dot(1)]
}
pub fn phase(pc: [f64; 2], order: &[usize]) -> (f64, usize) {
    let amplitude = pc[0].hypot(pc[1]);
    let sector = (pc[1].atan2(pc[0]).rem_euclid(std::f64::consts::TAU)
        / std::f64::consts::FRAC_PI_4)
        .floor() as usize
        % 8;
    (amplitude, if amplitude < 1. { 0 } else { order[sector] })
}

pub fn run(plan_path: &str) -> Result<(), String> {
    let wall = Instant::now();
    let p: Plan = load(plan_path)?;
    let b: Binding = load(&p.binding)?;
    let r: Reference = load(&p.reference)?;
    if b.schema != "gpuwm-hex.s2s-history-binding/v1"
        || r.schema != "gpuwm-hex.s2s-mjo-reference/v1"
        || r.method != "fixed-observed-eofs"
        || r.source.trim().is_empty()
        || r.training_period.trim().is_empty()
        || r.basis_id.trim().is_empty()
    {
        return Err(
            "unqualified binding/reference schema, EOF method, source or training period".into(),
        );
    }
    license(&r.license, p.research_data_license.as_deref())?;
    if b.members.is_empty()
        || b.days < 1
        || ![6, 24].contains(&b.cadence_hours)
        || r.definitions.is_empty()
    {
        return Err("empty roster/reference or invalid cadence".into());
    }
    let mut ids = std::collections::BTreeSet::new();
    for d in &r.definitions {
        validate(d)?;
        if !ids.insert(&d.id) {
            return Err("duplicate diagnostic definition".into());
        }
    }
    if !ids.contains(&"rmm".to_string()) {
        return Err("reference must contain rmm".into());
    }
    let start = NaiveDateTime::parse_from_str(&b.start, TIME).map_err(|e| e.to_string())?;
    if r.daily_interval_start_hour > 23
        || start.hour() != r.daily_interval_start_hour
        || start.minute() != 0
        || start.second() != 0
    {
        return Err(
            "reference daily interval UTC hour differs from common verification clock".into(),
        );
    }
    let grid = File::open(&b.grid)?;
    let lat = grid.read("latCell")?;
    let n = lat.values.len();
    lat.layout(&["nCells"], &[n])?;
    lat.units(&["rad", "radian", "radians"])?;
    let lon = grid.read("lonCell")?;
    lon.layout(&["nCells"], &[n])?;
    lon.units(&["rad", "radian", "radians"])?;
    let area = grid.read("areaCell")?;
    area.layout(&["nCells"], &[n])?;
    finite(&lat.values, "latitude")?;
    finite(&lon.values, "longitude")?;
    finite(&area.values, "area")?;
    if area.values.iter().any(|&a| a <= 0.) {
        return Err("nonpositive native cell area".into());
    }
    let bins: Vec<Groups> = r
        .definitions
        .iter()
        .map(|d| groups(d, &lat.values, &lon.values, &area.values))
        .collect::<Result<_, _>>()?;
    drop(grid);
    let mut series = Vec::new();
    let mut cold_start_intervals = Vec::new();
    for member in &b.members {
        if member.frames.len() != b.days * 24 / b.cadence_hours + 1 {
            return Err(format!("{} incomplete scheduled frames", member.member));
        }
        for (at, f) in member.frames.iter().enumerate() {
            if f.valid_time
                != (start + Duration::hours((at * b.cadence_hours) as i64))
                    .format(TIME)
                    .to_string()
            {
                return Err("frame valid time differs from common verification clock".into());
            }
        }
        let mut contexts: Vec<VecDeque<Vec<f64>>> =
            r.definitions
                .iter()
                .map(|d| {
                    (0..120)
                        .map(|k| {
                            let date = (start.date() - Duration::days(120 - k)).to_string();
                            d.prior_anomalies.get(&date).cloned().ok_or_else(|| {
                                format!("{} missing prior daily anomaly {date}", d.id)
                            })
                        })
                        .collect::<Result<_, _>>()
                })
                .collect::<Result<_, _>>()?;
        let (mut previous, cold_start_olr) = features(&member.frames[0], &r, &lat.values, &lon.values, &bins)?;
        if cold_start_olr {
            cold_start_intervals.push(json!({"member":member.member,"start":b.start,
                "hours":b.cadence_hours,"field":"olr",
                "reason":"globally zero initial radiation allocation; no radiation has run",
                "method":"first valid scheduled endpoint held over first subinterval; no observation substituted"}));
        }
        let mut daily: Vec<Vec<f64>> = previous.iter().map(|v| vec![0.; v.len()]).collect();
        let per_day = 24 / b.cadence_hours;
        for at in 1..member.frames.len() {
            let (next, uninitialized_olr) = features(&member.frames[at], &r, &lat.values, &lon.values, &bins)?;
            if uninitialized_olr {
                return Err(format!("{} globally zero OLR after initialization at {}", member.member, member.frames[at].valid_time));
            }
            if at == 1 && cold_start_olr {
                // Only this unavailable endpoint uses a one-sided quadrature.
                // Winds and all subsequent OLR intervals retain trapezoids.
                for (k, d) in r.definitions.iter().enumerate() {
                    let spatial = d.nx * d.ny;
                    for (field, name) in d.fields.iter().enumerate() {
                        if name == "olr" {
                            let range = field * spatial..(field + 1) * spatial;
                            previous[k][range.clone()].copy_from_slice(&next[k][range]);
                        }
                    }
                }
            }
            for k in 0..daily.len() {
                for j in 0..daily[k].len() {
                    daily[k][j] += (previous[k][j] + next[k][j]) * 0.5 / per_day as f64;
                }
            }
            previous = next;
            if at % per_day != 0 {
                continue;
            }
            let end = start + Duration::days((at / per_day) as i64);
            // Daily means are labelled by the interval's starting date.
            let date = (end - Duration::days(1)).date().to_string();
            for (k, d) in r.definitions.iter().enumerate() {
                let seasonal = d
                    .seasonal
                    .get(&date)
                    .ok_or_else(|| format!("{} missing seasonal background {date}", d.id))?;
                let anomaly: Vec<f64> = daily[k].iter().zip(seasonal).map(|(a, c)| a - c).collect();
                let corrected: Vec<f64> = (0..anomaly.len())
                    .into_par_iter()
                    .map(|j| anomaly[j] - contexts[k].iter().map(|v| v[j]).sum::<f64>() / 120.)
                    .collect();
                let spatial = d.nx * d.ny;
                let normalized: Vec<f64> = corrected
                    .iter()
                    .enumerate()
                    .map(|(j, v)| v / d.standard_deviation[j / spatial])
                    .collect();
                let pc = project(&normalized, &d.eofs, &d.pc_standard_deviation);
                finite(&pc, "projected PC")?;
                let (amplitude, phase) = phase(pc, &d.phase_order);
                // Longitude-time anomalies and latitude-time anomalies share
                // exactly the fields used by the fixed EOF projection.
                let bin_areas: Vec<f64> = bins[k]
                    .iter()
                    .map(|g| g.iter().map(|(i, _)| area.values[*i]).sum())
                    .collect();
                let hov: Vec<Vec<f64>> = corrected
                    .chunks(spatial)
                    .map(|f| {
                        (0..d.nx)
                            .map(|x| {
                                let denom: f64 = (0..d.ny).map(|y| bin_areas[y * d.nx + x]).sum();
                                (0..d.ny)
                                    .map(|y| f[y * d.nx + x] * bin_areas[y * d.nx + x])
                                    .sum::<f64>()
                                    / denom
                            })
                            .collect()
                    })
                    .collect();
                let lat_hov: Vec<Vec<f64>> = corrected
                    .chunks(spatial)
                    .map(|f| {
                        (0..d.ny)
                            .map(|y| {
                                let denom: f64 = bin_areas[y * d.nx..(y + 1) * d.nx].iter().sum();
                                (0..d.nx)
                                    .map(|x| f[y * d.nx + x] * bin_areas[y * d.nx + x])
                                    .sum::<f64>()
                                    / denom
                            })
                            .collect()
                    })
                    .collect();
                series.push(json!({"member":member.member,"diagnostic":d.id,"date":date,"interval_end":end.format(TIME).to_string(),
                    "pc":pc,"amplitude":amplitude,"phase":phase,"phase_order":d.phase_order,"fields":d.fields,"domain":d.domain,
                    "longitude_bins":d.nx,"latitude_bins":d.ny,"hovmoller":hov,"latitude_hovmoller":lat_hov}));
                contexts[k].pop_front();
                contexts[k].push_back(anomaly);
                daily[k].fill(0.);
            }
        }
    }
    let result = json!({"schema":"gpuwm-hex.s2s-mjo/v1","status":"complete","reference":p.reference,
        "reference_source":r.source,"license":r.license,"training_period":r.training_period,"basis_id":r.basis_id,
        "binding_sha256":b.binding_sha256,"method":"fixed observed EOF projection; seasonal background then preceding 120 daily anomalies removed",
        "qualification":"RMM-style / BSISO fixed-basis indices; equivalence to an operational index requires its qualified basis and preprocessing",
        "daily_interval_start":b.start,"sampling":"daily endpoint trapezoids of scheduled native fields; daily intervals labelled by start date; unavailable zero initial OLR uses first valid endpoint for first subinterval",
        "cold_start_olr_intervals":cold_start_intervals,
        "spatial_support":"area-weighted bins, finite pressure-level support renormalized only above the qualified reference minimum (default 90%, minimum 80%)",
        "history_fields":r.raw_fields,"series":series,"seconds":wall.elapsed().as_secs_f64(),
        "parallelism":{"backend":"Rust / Rayon","threads":rayon::current_num_threads()},"peak_vram_mib":0});
    save(Path::new(&p.output).join("receipt.json"), &result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_history_longitude_preserves_cell_identity() {
        let mesh = 4.486317441406737;
        let history = (mesh - std::f64::consts::TAU) as f32 as f64;
        assert!(same_coordinate(history, mesh, true));
        assert!(!same_coordinate(history, mesh, false));
        assert!(!same_coordinate(history + 0.001, mesh, true));
        assert!(!same_coordinate(f64::NAN, mesh, true));
    }
    #[test]
    fn interpolation_and_below_ground() {
        let p: [f64; 4] = [100000., 85000., 50000., 20000.];
        let u: Vec<_> = p.iter().map(|&p| p.ln()).collect();
        assert!((log_pressure(&p, &u, 70000.) - 70000f64.ln()).abs() < 1e-12);
        assert!(log_pressure(&p[1..], &u[1..], 90000.).is_nan());
        assert!(log_pressure(&[85000., 90000.], &[1., 2.], 87000.).is_nan());
    }
    #[test]
    fn fixed_projection_and_phase_order() {
        assert_eq!(
            project(&[2., 3.], &[vec![1., 0.], vec![0., 1.]], &[2., 3.]),
            [1., 1.]
        );
        assert_eq!(phase([2., 0.], &[5, 6, 7, 8, 1, 2, 3, 4]), (2., 5));
        assert_eq!(phase([0.2, 0.], &[5, 6, 7, 8, 1, 2, 3, 4]), (0.2, 0));
    }
}
