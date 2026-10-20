//! Verification from Rust-decoded native fields and normalized observation contracts.
use std::{collections::BTreeMap, env, fs, path::{Path, PathBuf}};
use serde_json::{json, Value};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
struct Field { values: Vec<f64>, shape: Vec<usize>, dims: Vec<String> }
impl Field {
    fn get(&self, cell: usize, level: usize) -> Result<f64> {
        let mut index: usize = 0;
        for (d, &n) in self.dims.iter().zip(&self.shape) {
            let i = match d.as_str() { "nCells" => cell, "nVertLevels" => level, "Time" => 0,
                _ if n == 1 => 0, _ => return Err(format!("unsupported dimension {d}").into()) };
            if i >= n { return Err("field index outside declared shape".into()); }
            index = index.checked_mul(n).and_then(|v: usize| v.checked_add(i)).ok_or("field index overflow")?;
        }
        Ok(self.values[index])
    }
}
fn dump(root: &Path) -> Result<BTreeMap<String, Field>> {
    let meta: Value = serde_json::from_slice(&fs::read(root.join("metadata.json"))?)?;
    if meta["schema"] != "gpuwm-rw-netcdf-dump-v1" { return Err("unsupported decoded-field contract".into()); }
    let mut fields = BTreeMap::new();
    for v in meta["variables"].as_array().ok_or("dump has no variables")? {
        if v["dtype"] != "<f8" { return Err("dump must carry little-endian f64".into()); }
        let shape: Vec<usize> = v["shape"].as_array().ok_or("no shape")?.iter()
            .map(|x| x.as_u64().and_then(|n| usize::try_from(n).ok()).ok_or("invalid dimension length"))
            .collect::<std::result::Result<_,_>>()?;
        let filename = Path::new(v["filename"].as_str().ok_or("no filename")?);
        if filename.components().count() != 1 || !matches!(filename.components().next(), Some(std::path::Component::Normal(_))) {
            return Err("decoded payload must be a file in its own dump directory".into());
        }
        let bytes = fs::read(root.join(filename))?;
        let count = shape.iter().try_fold(1usize, |n,&d| n.checked_mul(d)).ok_or("shape overflow")?;
        if bytes.len() != count.checked_mul(8).ok_or("dump size overflow")? { return Err("dump byte count mismatch".into()); }
        let values = bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();
        let dims: Vec<String> = v["dimensions"].as_array().ok_or("no dimensions")?.iter()
            .map(|x| x.as_str().map(str::to_string).ok_or("invalid dimension name"))
            .collect::<std::result::Result<_,_>>()?;
        if dims.len() != shape.len() { return Err("dimension names and shape disagree".into()); }
        fields.insert(v["name"].as_str().ok_or("no field name")?.to_string(), Field { values, shape, dims });
    }
    Ok(fields)
}
fn required<'a>(fields: &'a BTreeMap<String, Field>, name: &str) -> Result<&'a Field> {
    fields.get(name).ok_or_else(|| format!("native field {name} is absent").into())
}
fn timestamp(text: &str) -> Result<i64> {
    let text = if text.len() == 19 { format!("{text}Z") } else { text.to_string() };
    Ok(chrono::DateTime::parse_from_rfc3339(&text)?.timestamp())
}
#[derive(Default)]
struct Stat { n: usize, error: f64, square: f64 }
fn add(stats: &mut BTreeMap<String, Stat>, key: String, model: f64, obs: f64) {
    if !model.is_finite() || !obs.is_finite() { return; }
    let s = stats.entry(key).or_default(); let e = model - obs;
    s.n += 1; s.error += e; s.square += e * e;
}
fn xyz(lat: f64, lon: f64) -> [f64; 3] { [lat.cos()*lon.cos(), lat.cos()*lon.sin(), lat.sin()] }
fn nearest(coords: &[[f64; 3]], lat: f64, lon: f64) -> (usize, f64) {
    let target = xyz(lat.to_radians(), lon.to_radians());
    let mut best = (0, -2.0_f64);
    for (i, p) in coords.iter().enumerate() {
        let dot = p[0]*target[0] + p[1]*target[1] + p[2]*target[2];
        if dot > best.1 { best = (i, dot); }
    }
    (best.0, best.1.clamp(-1.0, 1.0).acos() * 6371.229)
}
fn profile(fields: &BTreeMap<String, Field>, cell: usize, target: f64, variable: &str) -> Result<Option<f64>> {
    let pressure = required(fields, "pressure")?;
    let axis = pressure.dims.iter().position(|d| d == "nVertLevels").ok_or("pressure has no vertical axis")?;
    let n = pressure.shape[axis];
    let values = required(fields, match variable { "temperature_k" => "theta", "wind_u_m_s" => "u_zonal", "wind_v_m_s" => "v_meridional", _ => return Ok(None) })?;
    let mut p = Vec::new(); let mut y = Vec::new();
    for k in 0..n {
        let pk = pressure.get(cell, k)?;
        let raw = values.get(cell, k)?;
        if !pk.is_finite() || pk <= 0.0 || !raw.is_finite() { return Ok(None); }
        p.push(pk);
        y.push(if variable == "temperature_k" { raw * (pk / 100000.0).powf(287.0 / 1004.5) } else { raw });
    }
    if target > p[0] || target < p[n-1] || p.windows(2).any(|v| v[0] <= v[1]) { return Ok(None); }
    for k in 0..n-1 {
        if p[k] >= target && target >= p[k+1] {
            let weight = (target.ln() - p[k].ln()) / (p[k+1].ln() - p[k].ln());
            return Ok(Some(y[k] + weight * (y[k+1] - y[k])));
        }
    }
    Ok(None)
}
fn run() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 8 { return Err("usage: gpuwm-hex-native-score GEOMETRY_DUMP HISTORY_DUMP STATIONS.json SURFACE.json SONDES.csv|- VALID_TIME OUT.json".into()); }
    let geo = dump(Path::new(&args[1]))?; let fields = dump(Path::new(&args[2]))?;
    let lat = required(&geo, "latCell")?; let lon = required(&geo, "lonCell")?;
    if lat.values.len() != lon.values.len() { return Err("coordinate count mismatch".into()); }
    let coords: Vec<_> = lat.values.iter().zip(&lon.values).map(|(&a,&b)| xyz(a,b)).collect();
    let valid = timestamp(&args[6])?;
    let stations: Value = serde_json::from_slice(&fs::read(&args[3])?)?;
    let surface: Value = serde_json::from_slice(&fs::read(&args[4])?)?;
    let mut stats = BTreeMap::new(); let mut pairs = Vec::new();
    for report in surface["reports"].as_array().ok_or("surface record has no reports")? {
        let id = report["station_id"].as_str().ok_or("surface station missing id")?;
        let station = stations["stations"].as_array().ok_or("no stations")?.iter().find(|s| s["station_id"] == id).ok_or("station not frozen")?;
        let delta = timestamp(report["observation_time"].as_str().ok_or("no observation time")?)? - valid;
        if delta.abs() > 600 { continue; }
        let (cell, distance) = nearest(&coords, station["latitude"].as_f64().ok_or("no latitude")?, station["longitude"].as_f64().ok_or("no longitude")?);
        for (key, value) in [("temperature_2m", required(&fields,"t2")?.get(cell,0)?),
                             ("wind_speed_10m", required(&fields,"u10")?.get(cell,0)?.hypot(required(&fields,"v10")?.get(cell,0)?))] {
            if let Some(obs) = report["values"][key].as_f64() {
                add(&mut stats, format!("surface.{key}"), value, obs);
                pairs.push(json!({"source":"surface","station":id,"field":key,"cell":cell,"distance_km":distance,"offset_seconds":delta,"model":value,"observation":obs}));
            }
        }
    }
    let mut skipped = BTreeMap::<String,usize>::new();
    if args[5] != "-" {
        let mut reader = csv::Reader::from_path(&args[5])?;
        let headers = reader.headers()?.clone();
        let column = |name: &str| -> Result<usize> { headers.iter().position(|h| h == name).ok_or_else(|| format!("sonde table lacks {name}").into()) };
        let (itime, ilat, ilon, ilevel, ivar, ivalue, iid) = (column("valid_time")?, column("latitude_deg")?, column("longitude_deg")?, column("level_pa")?, column("variable")?, column("value")?, column("station_id")?);
        let mut locations = BTreeMap::new();
        for row in reader.records() {
            let row = row?; let variable = &row[ivar];
            if !["temperature_k","wind_u_m_s","wind_v_m_s"].contains(&variable) { continue; }
            let delta = timestamp(&row[itime])? - valid;
            if delta.abs() > 1800 { *skipped.entry("time_outside_30_minutes".into()).or_default() += 1; continue; }
            let level: f64 = match row[ilevel].parse() { Ok(v) if v > 0.0 => v, _ => { *skipped.entry("no_pressure_level".into()).or_default() += 1; continue; } };
            let id = row[iid].to_string();
            let (cell, distance) = *locations.entry(id.clone()).or_insert_with(|| nearest(&coords,row[ilat].parse().unwrap(),row[ilon].parse().unwrap()));
            if let Some(value) = profile(&fields,cell,level,variable)? {
                let obs: f64 = row[ivalue].parse()?;
                add(&mut stats,format!("sonde.{variable}.all"),value,obs);
                add(&mut stats,format!("sonde.{variable}.{}hPa",level/100.0),value,obs);
                pairs.push(json!({"source":"sonde","station":id,"field":variable,"pressure_pa":level,"cell":cell,"distance_km":distance,"offset_seconds":delta,"model":value,"observation":obs}));
            } else { *skipped.entry("profile_not_bracketed_or_invalid".into()).or_default() += 1; }
        }
    }
    let metrics: BTreeMap<_,_> = stats.into_iter().map(|(k,s)| (k,json!({"n":s.n,"bias_model_minus_observation":s.error/s.n as f64,"rmse":(s.square/s.n as f64).sqrt()}))).collect();
    let out = json!({"schema":"gpuwm-hex.native-observation-score/v1","valid_time":args[6],"operators":{"horizontal":"nearest native cell on unit sphere; no land or elevation correction","vertical":"linear interpolation in log pressure; no vertical extrapolation","temperature":"dry theta*(p/100000)^(287/1004.5)","sonde_time_window_seconds":1800,"surface_time_window_seconds":600},"metrics":metrics,"skipped":skipped,"pairs":pairs});
    fs::write(PathBuf::from(&args[7]),serde_json::to_vec_pretty(&out)?)?;
    println!("SCORED\t{}\t{}",out["pairs"].as_array().unwrap().len(),args[7]);
    Ok(())
}
fn main() { if let Err(e) = run() { eprintln!("gpuwm-hex-native-score: {e}"); std::process::exit(1); } }
