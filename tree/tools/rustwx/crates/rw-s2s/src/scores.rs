use crate::{
    finite, license, load,
    nc::{Array, File},
    save,
};
use chrono::{Duration, NaiveDateTime};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Instant,
};

const FIELDS: [(&str, &str); 4] = [
    ("temperature2", "K"),
    ("precipitation_rate", "mm day-1"),
    ("z500", "m"),
    ("mslp", "hPa"),
];
const TIME: &str = "%Y-%m-%d_%H:%M:%S";
struct Plan {
    observation_manifest: String,
    products: String,
    mjo: String,
    output: String,
    days: usize,
    start: String,
    members: Vec<String>,
    research_data_license: Option<String>,
}
struct Observations {
    schema: String,
    source: String,
    license: String,
    provenance: String,
    weeks: Vec<Week>,
    rmm: BTreeMap<String, [f64; 2]>,
    minimum_area_fraction: f64,
    rmm_basis_id: String,
    rmm_sampling: String,
    rmm_interval_start: String,
}
struct Week {
    week: usize,
    start: String,
    end: String,
    observations: String,
    climatology: String,
}
crate::deserialize_fields!(Plan {
    observation_manifest,
    products,
    mjo,
    output,
    days,
    start,
    members,
    research_data_license
});
crate::deserialize_fields!(Observations {
    schema,
    source,
    license,
    provenance,
    weeks,
    rmm,
    minimum_area_fraction,
    rmm_basis_id,
    rmm_sampling,
    rmm_interval_start
});
crate::deserialize_fields!(Week {
    week,
    start,
    end,
    observations,
    climatology
});

/// Uncentred spatial ACC of anomalies about one independent climatology.
/// Constant nonzero anomalies are correlated; a zero anomaly norm is undefined.
pub fn metrics(f: &[f64], o: &[f64], c: &[f64], w: &[f64]) -> Value {
    let sums = (0..f.len())
        .into_par_iter()
        .map(|i| {
            if f[i].is_finite()
                && o[i].is_finite()
                && c[i].is_finite()
                && w[i].is_finite()
                && w[i] > 0.
            {
                let (a, b, d) = (f[i] - c[i], o[i] - c[i], f[i] - o[i]);
                [
                    w[i],
                    w[i] * d,
                    w[i] * d * d,
                    w[i] * a * b,
                    w[i] * a * a,
                    w[i] * b * b,
                    1.,
                ]
            } else {
                [0.; 7]
            }
        })
        .reduce(
            || [0.; 7],
            |mut a, b| {
                for i in 0..7 {
                    a[i] += b[i];
                }
                a
            },
        );
    let all: f64 = w.iter().filter(|x| x.is_finite() && **x > 0.).sum();
    let den = (sums[4] * sums[5]).sqrt();
    json!({"finite_cells":sums[6] as usize,"area_fraction":if all>0.{sums[0]/all}else{0.},
        "bias":if sums[0]>0.{Some(sums[1]/sums[0])}else{None},
        "rmse":if sums[0]>0.{Some((sums[2]/sums[0]).sqrt())}else{None},
        "acc":if den>0.{Some((sums[3]/den).clamp(-1.,1.))}else{None}})
}

pub fn bivariate(f: &[[f64; 2]], o: &[[f64; 2]]) -> Value {
    let mut xy = 0.;
    let mut xx = 0.;
    let mut yy = 0.;
    let mut error = 0.;
    let mut count = 0;
    for (a, b) in f.iter().zip(o) {
        if a.iter().chain(b).any(|v| !v.is_finite()) {
            continue;
        }
        for k in 0..2 {
            xy += a[k] * b[k];
            xx += a[k] * a[k];
            yy += b[k] * b[k];
            error += (a[k] - b[k]).powi(2);
        }
        count += 1;
    }
    let den = (xx * yy).sqrt();
    json!({"paired_days":count,"bivariate_correlation":if den>0. {Some((xy/den).clamp(-1.,1.))}else{None},
        "bivariate_rmse":if count>0 {Some((error/count as f64).sqrt())}else{None}})
}

fn plane(file: &File, name: &str, shape: &[usize], units: &str) -> Result<Array, String> {
    let a = file.read(name)?;
    a.layout(&["south_north", "west_east"], shape)?;
    a.units(&[units])?;
    Ok(a)
}
fn equal(a: &[f64], b: &[f64], label: &str, tolerance: f64) -> Result<(), String> {
    if a.len() != b.len()
        || a.iter()
            .zip(b)
            .any(|(a, b)| !a.is_finite() || !b.is_finite() || (a - b).abs() > tolerance)
    {
        Err(format!("{label} differs from forecast grid"))
    } else {
        Ok(())
    }
}
fn located(root: &Path, path: &str) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.into()
    } else {
        root.join(p)
    }
}

pub fn run(plan_path: &str) -> Result<(), String> {
    let wall = Instant::now();
    let p: Plan = load(plan_path)?;
    let o: Observations = load(&p.observation_manifest)?;
    if o.schema != "gpuwm-hex.s2s-observations/v1"
        || o.source != "ERA5"
        || o.provenance.trim().is_empty()
        || !(0.95..=1.).contains(&o.minimum_area_fraction)
        || p.members.is_empty()
        || p.days < 7
    {
        return Err("unqualified ERA5 observation manifest, area coverage or roster".into());
    }
    license(&o.license, p.research_data_license.as_deref())?;
    let root = Path::new(&p.observation_manifest)
        .parent()
        .ok_or("observation path parent")?;
    let start = NaiveDateTime::parse_from_str(&p.start, TIME).map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    for week in 1..=p.days / 7 {
        let matching: Vec<_> = o.weeks.iter().filter(|w| w.week == week).collect();
        if matching.len() != 1 {
            return Err(format!("ERA5 week-{week} missing or duplicated"));
        }
        let spec = matching[0];
        let begin = (start + Duration::days(((week - 1) * 7) as i64))
            .format(TIME)
            .to_string();
        let end = (start + Duration::days((week * 7) as i64))
            .format(TIME)
            .to_string();
        if spec.start != begin || spec.end != end {
            return Err(format!("ERA5 week-{week} has different valid interval"));
        }
        // libnetcdf is serialized, then all field reductions use Rayon.
        let forecast = File::open(Path::new(&p.products).join(format!("week-{week}.nc")))?;
        if forecast.attr(-1, "s2s_window")? != format!("week-{week}")
            || forecast.attr(-1, "valid_time")? != end
        {
            return Err("forecast week or valid time differs from requested interval".into());
        }
        let ids: Vec<String> = serde_json::from_str(&forecast.attr(-1, "s2s_member_ids")?)
            .map_err(|e| e.to_string())?;
        if ids != p.members {
            return Err("weekly product roster differs from bound S2S members".into());
        }
        let lat = forecast.read("XLAT")?;
        let lon = forecast.read("XLONG")?;
        if lat.shape.len() != 2 {
            return Err("weekly coordinates must be two dimensional".into());
        }
        lat.layout(&["south_north", "west_east"], &lat.shape)?;
        lon.layout(&["south_north", "west_east"], &lat.shape)?;
        lat.units(&["degrees_north"])?;
        lon.units(&["degrees_east"])?;
        finite(&lat.values, "latitude")?;
        finite(&lon.values, "longitude")?;
        if lat.values.iter().any(|v| v.abs() > 90.) {
            return Err("latitude outside degree range".into());
        }
        let weights: Vec<f64> = lat
            .values
            .iter()
            .map(|v| v.to_radians().cos().max(0.))
            .collect();
        // Coordinate rows must describe a regular complete globe. A clipped or
        // duplicate seam would bias an area-weighted global verification.
        let (ny, nx) = (lat.shape[0], lat.shape[1]);
        if nx < 4 || ny < 3 {
            return Err("weekly global grid is too small".into());
        }
        let dx = 360. / nx as f64;
        let dy = (lat.values[(ny - 1) * nx] - lat.values[0]) / (ny - 1) as f64;
        for y in 0..ny {
            for x in 0..nx {
                if (lat.values[y * nx + x] - (lat.values[0] + y as f64 * dy)).abs() > 1e-5
                    || (lon.values[y * nx] - lon.values[0]).abs() > 1e-5
                    || (lon.values[y * nx + x] - (lon.values[y * nx] + x as f64 * dx)).abs() > 1e-5
                {
                    return Err("verification grid is not regular global longitude/latitude".into());
                }
            }
        }
        if (lat.values[0].abs() - 90.).abs() > 1e-5
            || (lat.values[(ny - 1) * nx].abs() - 90.).abs() > 1e-5
            || lat.values[0] * lat.values[(ny - 1) * nx] >= 0.
        {
            return Err("verification grid does not cover both poles".into());
        }
        let terrain = plane(&forecast, "model_orography", &lat.shape, "m")?;
        let mut member_fields = Vec::new();
        for (field, units) in FIELDS {
            let a = forecast.read(&format!("{field}_members"))?;
            a.layout(
                &[&format!("{field}_member"), "south_north", "west_east"],
                &[ids.len(), ny, nx],
            )?;
            a.units(&[units])?;
            member_fields.push(a.values);
        }
        drop(forecast);
        let mut references = Vec::new();
        for (path, source) in [
            (&spec.observations, "observations"),
            (&spec.climatology, "climatology"),
        ] {
            let file = File::open(located(root, path))?;
            if file.attr(-1, "s2s_reference_source")? != "ERA5"
                || file.attr(-1, "s2s_reference_kind")? != source
                || file.attr(-1, "s2s_window_start")? != begin
                || file.attr(-1, "s2s_window_end")? != end
            {
                return Err(format!(
                    "week-{week} {source} has different source/kind/interval"
                ));
            }
            for (name, expected, units) in [
                ("XLAT", &lat.values, "degrees_north"),
                ("XLONG", &lon.values, "degrees_east"),
                ("model_orography", &terrain.values, "m"),
            ] {
                let a = plane(&file, name, &lat.shape, units)?;
                equal(
                    &a.values,
                    expected,
                    name,
                    if name == "model_orography" {
                        0.01
                    } else {
                        1e-5
                    },
                )?;
            }
            let fields = FIELDS
                .iter()
                .map(|(field, units)| plane(&file, field, &lat.shape, units).map(|a| a.values))
                .collect::<Result<Vec<_>, _>>()?;
            references.push(fields);
        }
        let n = ny * nx;
        for (field_at, (field, units)) in FIELDS.iter().enumerate() {
            let members = &member_fields[field_at];
            let mean: Vec<f64> = (0..n)
                .into_par_iter()
                .map(|i| (0..ids.len()).map(|m| members[m * n + i]).sum::<f64>() / ids.len() as f64)
                .collect();
            for member in 0..=ids.len() {
                let f = if member == ids.len() {
                    &mean[..]
                } else {
                    &members[member * n..(member + 1) * n]
                };
                let score = metrics(
                    f,
                    &references[0][field_at],
                    &references[1][field_at],
                    &weights,
                );
                if score["area_fraction"].as_f64().unwrap_or(0.) < o.minimum_area_fraction {
                    return Err(format!(
                        "week-{week} {field} insufficient paired area: {score}"
                    ));
                }
                rows.push(json!({"week":week,"start":begin,"end":end,"member":if member==ids.len(){"ensemble-mean"}else{&ids[member]},
                    "field":field,"units":units,"scores":score,"observations":spec.observations,"climatology":spec.climatology}));
            }
        }
    }
    let mjo: Value = load(&p.mjo)?;
    if mjo["schema"] != "gpuwm-hex.s2s-mjo/v1" || mjo["status"] != "complete" {
        return Err("qualified completed MJO receipt is required for RMM scores".into());
    }
    if o.rmm_basis_id.is_empty()
        || mjo["basis_id"] != o.rmm_basis_id
        || o.rmm_sampling != "daily-interval-start"
        || o.rmm_interval_start != p.start
        || mjo["daily_interval_start"] != p.start
    {
        return Err("observed RMM basis or daily sampling differs from forecast index".into());
    }
    let series = mjo["series"].as_array().ok_or("MJO series missing")?;
    let mut rmm_rows = Vec::new();
    let mut dates = Vec::new();
    let mut ensemble = Vec::new();
    for day in 0..p.days {
        let date = (start + Duration::days(day as i64)).date().to_string();
        let observed = o
            .rmm
            .get(&date)
            .ok_or_else(|| format!("missing observed RMM pair {date}"))?;
        finite(observed, "observed RMM")?;
        let mut pairs = Vec::new();
        for id in &p.members {
            let matching: Vec<_> = series
                .iter()
                .filter(|s| {
                    s["diagnostic"] == "rmm" && s["member"] == id.as_str() && s["date"] == date
                })
                .collect();
            if matching.len() != 1 {
                return Err(format!("missing/duplicate forecast RMM pair {id} {date}"));
            }
            let pair: [f64; 2] =
                serde_json::from_value(matching[0]["pc"].clone()).map_err(|e| e.to_string())?;
            finite(&pair, "forecast RMM")?;
            pairs.push(pair);
        }
        ensemble.push([
            pairs.iter().map(|p| p[0]).sum::<f64>() / pairs.len() as f64,
            pairs.iter().map(|p| p[1]).sum::<f64>() / pairs.len() as f64,
        ]);
        dates.push((date, pairs, *observed));
    }
    for member in 0..=p.members.len() {
        for week in 0..=p.days / 7 {
            let (first, last) = if week == 0 {
                (0, p.days)
            } else {
                ((week - 1) * 7, week * 7)
            };
            let f: Vec<_> = (first..last)
                .map(|day| {
                    if member == p.members.len() {
                        ensemble[day]
                    } else {
                        dates[day].1[member]
                    }
                })
                .collect();
            let observed: Vec<_> = dates[first..last].iter().map(|d| d.2).collect();
            rmm_rows.push(json!({"member":if member==p.members.len(){"ensemble-mean"}else{&p.members[member]},
                "window":if week==0{"all".to_string()}else{format!("week-{week}")},"scores":bivariate(&f,&observed)}));
        }
    }
    save(
        Path::new(&p.output).join("receipt.json"),
        &json!({"schema":"gpuwm-hex.s2s-scores/v1","status":"complete",
        "source":"ERA5","license":o.license,"provenance":o.provenance,"observation_manifest":p.observation_manifest,
        "weekly":rows,"rmm":rmm_rows,"coverage_days":p.days,"complete_weeks":p.days/7,"unscored_tail_days":p.days%7,
        "acc":"cosine-latitude-weighted uncentred anomaly pattern correlation about an independent ERA5 normal",
        "rmm_correlation":"sum(F1*O1+F2*O2)/sqrt(sum(F1^2+F2^2)*sum(O1^2+O2^2)); zero norm is null",
        "seconds":wall.elapsed().as_secs_f64(),"peak_vram_mib":0,"parallelism":{"backend":"Rust / Rayon","threads":rayon::current_num_threads()}}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn weighted_error_and_independent_climatology() {
        let m = metrics(
            &[2., 4., f64::NAN],
            &[1., 1., 3.],
            &[0., 0., 0.],
            &[1., 3., 10.],
        );
        assert_eq!(m["finite_cells"], 2);
        assert_eq!(m["bias"], 2.5);
        assert!((m["rmse"].as_f64().unwrap() - 7f64.sqrt()).abs() < 1e-12);
        assert_eq!(
            metrics(&[1., 1.], &[1., 1.], &[1., 1.], &[1., 1.])["acc"],
            Value::Null
        );
    }
    #[test]
    fn rmm_is_uncentred_two_component_correlation() {
        let f = [[1., 0.], [0., 1.], [-1., 0.], [0., -1.]];
        assert_eq!(bivariate(&f, &f)["bivariate_correlation"], 1.);
        assert_eq!(
            bivariate(&f, &[[0., 1.], [-1., 0.], [0., -1.], [1., 0.]])["bivariate_correlation"],
            0.
        );
        assert_eq!(
            bivariate(&[[0., 0.]], &[[0., 0.]])["bivariate_correlation"],
            Value::Null
        );
    }
}
