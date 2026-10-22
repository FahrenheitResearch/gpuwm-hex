//! Dated hindcast reference: average converted annual energy, never mean weather.
use super::*;
use std::collections::BTreeSet;

#[derive(Deserialize)]
struct Manifest {
    target_start: String,
    requests: Vec<PathBuf>,
    output: PathBuf,
}

pub(super) fn build(path: &Path) -> Result<()> {
    let started = Instant::now();
    let manifest: Manifest = serde_json::from_slice(&fs::read(path)?)?;
    let target = NaiveDateTime::parse_from_str(&manifest.target_start, FORMAT)?;
    if manifest.requests.len() < 2 {
        return Err("energy reference needs at least two distinct historical years".into());
    }
    let annual: Vec<Request> = manifest
        .requests
        .iter()
        .map(|p| Ok(serde_json::from_slice(&fs::read(p)?)?))
        .collect::<Result<_>>()?;
    let first = &annual[0];
    if first.windows.is_empty() {
        return Err("energy reference needs at least one window".into());
    }
    let prods = products(&first.config)?;
    let mut years = BTreeSet::new();
    for req in &annual {
        let start = NaiveDateTime::parse_from_str(&req.start, FORMAT)?;
        if start.year() >= target.year()
            || !years.insert(start.year())
            || start.format("%m-%d_%H:%M:%S").to_string()
                != target.format("%m-%d_%H:%M:%S").to_string()
        {
            return Err("reference years must be distinct, preceding and calendar matched".into());
        }
        if req.config.climatology.is_some()
            || req.config.table_sha256 != first.config.table_sha256
            || req.config.heights_m_agl != first.config.heights_m_agl
            || req.config.rows != first.config.rows
            || req.step_degrees != first.step_degrees
            || req.cadence_hours != 6
            || req.days != first.days
            || req.members.is_empty()
            || req.windows.len() != first.windows.len()
            || req
                .windows
                .iter()
                .zip(&first.windows)
                .any(|(a, b)| a.name != b.name || a.low != b.low || a.high != b.high)
        {
            return Err("reference conversion, grid, sampling or windows differ".into());
        }
    }
    let sample = netcrust::open(first.output.join(format!("{}.nc", first.windows[0].name)))?;
    let coordinate = sample.read_array_f64("XLAT")?;
    let shape = coordinate.shape();
    if shape.len() != 2 {
        return Err("reference coordinates must be two-dimensional".into());
    }
    let (ny, nx) = (shape[0], shape[1]);
    let lat = read(&sample, &["XLAT"], &[ny, nx])?;
    let lon = read(&sample, &["XLONG"], &[ny, nx])?;
    let mut defs = Vec::new();
    let mut payloads: Vec<(String, Vec<f32>)> = Vec::new();
    for (key, data, units) in [
        ("XLAT", &lat, "degrees_north"),
        ("XLONG", &lon, "degrees_east"),
    ] {
        defs.push(
            NcVarDef::new(key, NcType::Float, vec![0, 1])
                .with_attrs(vec![NcAttr::text("units", units)]),
        );
        payloads.push((key.into(), data.iter().map(|&v| v as f32).collect()));
    }
    let mut summaries = Vec::new();
    for window in &first.windows {
        let files = annual
            .iter()
            .map(|req| netcrust::open(req.output.join(format!("{}.nc", window.name))))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for (req, file) in annual.iter().zip(&files) {
            let valid = (NaiveDateTime::parse_from_str(&req.start, FORMAT)?
                + Duration::days(window.high as i64))
            .format(FORMAT)
            .to_string();
            if attr(file, "s2s_energy_table_sha256")? != first.config.table_sha256
                || attr(file, "s2s_energy_heights_m_agl")?
                    != serde_json::to_string(&first.config.heights_m_agl)?
                || attr(file, "s2s_energy_rows")? != serde_json::to_string(&first.config.rows)?
                || attr(file, "s2s_window")? != window.name
                || attr(file, "valid_time")? != valid
                || attr(file, "s2s_sampling")?
                    != "6-hour sampled endpoint trapezoids; conversion before averaging"
                || file.attribute("ensemble_members").and_then(|a| a.as_f64())
                    != Some(req.members.len() as f64)
            {
                return Err("annual aggregate provenance differs from request".into());
            }
            for (key, expected) in [("XLAT", &lat), ("XLONG", &lon)] {
                let values = read(file, &[key], &[ny, nx])?;
                if values
                    .iter()
                    .zip(expected)
                    .any(|(a, b)| !a.is_finite() || !b.is_finite() || (a - b).abs() > 1e-5)
                {
                    return Err("annual reference coordinates differ".into());
                }
            }
        }
        for product in &prods {
            let key = format!("{}_{}", window.name.replace('-', "_"), product.id);
            let fields = files
                .iter()
                .zip(&annual)
                .map(|(file, req)| {
                    let values = read_units(
                        file,
                        &[&format!("{}_mean", product.id)],
                        &[ny, nx],
                        &[&product.units],
                    )?;
                    let counts = read_units(
                        file,
                        &[&format!("{}_finite_count", product.id)],
                        &[ny, nx],
                        &["1"],
                    )?;
                    if values
                        .iter()
                        .zip(counts)
                        .any(|(v, n)| v.is_finite() && n != req.members.len() as f64)
                    {
                        return Err("annual reference has an incomplete member roster".into());
                    }
                    Ok(values)
                })
                .collect::<Result<Vec<_>>>()?;
            let mean: Vec<f32> = (0..ny * nx)
                .into_par_iter()
                .map(|i| (fields.iter().map(|v| v[i]).sum::<f64>() / fields.len() as f64) as f32)
                .collect();
            summaries.push(json!({"field":key,"units":product.units,"cells":mean.len(),"finite":mean.iter().filter(|v|v.is_finite()).count()}));
            defs.push(
                NcVarDef::new(&key, NcType::Float, vec![0, 1])
                    .with_attrs(vec![NcAttr::text("units", &product.units)]),
            );
            payloads.push((key, mean));
        }
    }
    let years: Vec<_> = years.into_iter().collect();
    let attrs = vec![
        NcAttr::text("s2s_energy_table_sha256", &first.config.table_sha256),
        NcAttr::text(
            "s2s_energy_heights_m_agl",
            serde_json::to_string(&first.config.heights_m_agl)?,
        ),
        NcAttr::text(
            "s2s_climatology_start_mmdd",
            target.format("%m-%d").to_string(),
        ),
        NcAttr::int("climatology_year_start", years[0]),
        NcAttr::int("climatology_year_end", *years.last().unwrap()),
        NcAttr::int("climatology_year_count", years.len() as i32),
        NcAttr::text("s2s_reference_years", serde_json::to_string(&years)?),
        NcAttr::text(
            "s2s_reference_type",
            "dated hindcast mean; not an observational climate normal",
        ),
        NcAttr::text(
            "s2s_reference_requests",
            serde_json::to_string(&manifest.requests)?,
        ),
        NcAttr::text(
            "s2s_sampling",
            "6-hour sampled endpoint trapezoids; conversion before averaging",
        ),
    ];
    let tmp = manifest.output.with_extension("nc.part");
    if let Some(parent) = tmp.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = NcClassicWriter::create(
        &tmp,
        NcFormat::Data64,
        vec![
            NcDim::fixed("south_north", ny),
            NcDim::fixed("west_east", nx),
        ],
        attrs,
        defs,
        0,
    )?;
    let jobs: Vec<_> = payloads
        .iter()
        .map(|(name, _)| (name.as_str(), None))
        .collect();
    writer.put_parallel(&jobs, |i| {
        Ok(NcSlab::Borrowed(NcData::Floats(&payloads[i].1)))
    })?;
    writer.finish()?;
    fs::rename(tmp, &manifest.output)?;
    fs::write(
        manifest.output.with_extension("receipt.json"),
        serde_json::to_vec_pretty(&json!({
            "schema":"gpuwm-hex.s2s-energy-reference/v1","manifest":path,"output":manifest.output,
            "requests":manifest.requests,"years":years,"target_start":manifest.target_start,
            "reference_type":"dated hindcast mean; not an observational climate normal",
            "fields":summaries,"seconds":started.elapsed().as_secs_f64(),"threads":rayon::current_num_threads(),"peak_vram_mib":0
        }))?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dated_reference_roundtrip_and_duplicate_year_refusal() -> Result<()> {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-output/dated-reference");
        fs::create_dir_all(&dir)?;
        let lat = vec![-1f32, -1., 1., 1.];
        let lon = vec![-1f32, 1., -1., 1.];
        let mut paths = Vec::new();
        for (year, value) in [(2024, 2.), (2025, 4.)] {
            let output = dir.join(year.to_string());
            fs::create_dir_all(&output)?;
            let body = json!({"config":{"rows":[{"id":"wind","kind":"wind","units":"m s-1","display_max":30.}],
                "table_sha256":"test-hash","heights_m_agl":[87.5],"climatology":null},
                "start":format!("{year}-10-20_00:00:00"),"cadence_hours":6,"days":30,"grid":"unused.nc",
                "output":output,"step_degrees":1.,"windows":[{"name":"week-1","low":0,"high":7}],
                "members":[{"id":"one","init":"unused.nc","histories":[]}]});
            let req: Request = serde_json::from_value(body.clone())?;
            let prods = products(&req.config)?;
            write_window(
                &req,
                &req.windows[0],
                &prods,
                &[vec![vec![value; 4]]],
                &lat,
                &lon,
                2,
                2,
            )?;
            let path = output.join("request.json");
            fs::write(&path, serde_json::to_vec(&body)?)?;
            paths.push(path);
        }
        let output = dir.join("normal.nc");
        let manifest = dir.join("reference.json");
        fs::write(
            &manifest,
            serde_json::to_vec(
                &json!({"target_start":"2026-10-20_00:00:00","requests":paths,"output":output}),
            )?,
        )?;
        build(&manifest)?;
        let file = netcrust::open(&output)?;
        assert_eq!(
            read_units(&file, &["week_1_wind_87p5m"], &[2, 2], &["m s-1"])?,
            vec![3.; 4]
        );
        assert_eq!(attr(&file, "s2s_reference_years")?, "[2024,2025]");
        fs::write(
            &manifest,
            serde_json::to_vec(
                &json!({"target_start":"2026-10-20_00:00:00","requests":[paths[0],paths[0]],"output":output}),
            )?,
        )?;
        assert!(
            build(&manifest)
                .unwrap_err()
                .to_string()
                .contains("distinct")
        );
        fs::write(
            &manifest,
            serde_json::to_vec(
                &json!({"target_start":"2025-10-20_00:00:00","requests":paths,"output":output}),
            )?,
        )?;
        assert!(
            build(&manifest)
                .unwrap_err()
                .to_string()
                .contains("preceding")
        );
        let mut body: serde_json::Value = serde_json::from_slice(&fs::read(&paths[0])?)?;
        body["windows"] = json!([]);
        fs::write(&paths[0], serde_json::to_vec(&body)?)?;
        fs::write(
            &manifest,
            serde_json::to_vec(&json!({
                "target_start":"2026-10-20_00:00:00", "requests":paths,"output":output
            }))?,
        )?;
        assert!(build(&manifest).unwrap_err().to_string().contains("window"));
        Ok(())
    }
}
