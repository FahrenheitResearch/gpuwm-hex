//! Parallel acquisition, Rust intermediate decode, case-8 remap and native surface stream.
mod intermediate;
mod remap;
mod surface_met;
use chrono::{Duration, NaiveDateTime, Timelike};
use rayon::prelude::*;
use reqwest::blocking::Client;
use rw_store::netcdf_classic::{
    NcAttr, NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn date(value: &str) -> Result<NaiveDateTime> {
    for format in ["%Y-%m-%d_%H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y%m%d%H%M%S"] {
        if let Ok(d) = NaiveDateTime::parse_from_str(value, format) {
            return Ok(d);
        }
    }
    for format in ["%Y-%m-%dT%H", "%Y%m%d%H"] {
        if let Ok(d) =
            NaiveDateTime::parse_from_str(&format!("{value}:00:00"), &format!("{format}:%M:%S"))
        {
            return Ok(d);
        }
    }
    Err(format!("invalid cycle {value:?}").into())
}
fn stamp(d: NaiveDateTime) -> String {
    d.format("%Y-%m-%d_%H:%M:%S").to_string()
}
fn render(value: &str, d: NaiveDateTime) -> String {
    let mut s = value.to_string();
    for (key, format) in [
        ("yyyymmddhh", "%Y%m%d%H"),
        ("yyyymmdd", "%Y%m%d"),
        ("yyyy", "%Y"),
        ("mm", "%m"),
        ("dd", "%d"),
        ("hh", "%H"),
    ] {
        s = s.replace(&format!("{{{key}}}"), &d.format(format).to_string());
    }
    s.replace("{lead:03d}", "000")
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn forecast_product(value: &Value, cycle: NaiveDateTime, valid: NaiveDateTime) -> Value {
    match value {
        Value::String(text) => {
            let mut text = text.clone();
            for (key, format) in [("yyyymmddhh", "%Y%m%d%H"), ("yyyymmdd", "%Y%m%d"),
                                  ("yyyy", "%Y"), ("mm", "%m"), ("dd", "%d"), ("hh", "%H")] {
                text = text.replace(&format!("{{valid_{key}}}"), &valid.format(format).to_string());
            }
            text = text.replace("{lead:03d}", &format!("{:03}", (valid-cycle).num_hours()));
            Value::String(render(&text, cycle))
        }
        Value::Array(values) => Value::Array(values.iter().map(|v| forecast_product(v, cycle, valid)).collect()),
        Value::Object(values) => Value::Object(values.iter().map(|(key,v)| (key.clone(), forecast_product(v, cycle, valid))).collect()),
        _ => value.clone(),
    }
}
fn urls(value: &Value, d: NaiveDateTime) -> Vec<String> {
    if let Some(s) = value.as_str() {
        vec![render(s, d)]
    } else {
        value
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(|s| render(s, d))
            .collect()
    }
}
fn get(client: &Client, url: &str, range: Option<(u64, Option<u64>)>) -> Result<Vec<u8>> {
    for attempt in 0..5 {
        let mut request = client.get(url);
        if let Some((a, b)) = range {
            request = request.header(
                "Range",
                format!("bytes={a}-{}", b.map(|n| n.to_string()).unwrap_or_default()),
            );
        }
        let response = match request.send() {
            Ok(response) => response,
            Err(error)
                if attempt < 4 && (error.is_timeout() || error.is_connect() || error.is_body()) =>
            {
                continue;
            }
            Err(error) => return Err(error.without_url().into()),
        };
        let status = response.status();
        if status.as_u16() == 429 || status.is_server_error() {
            std::thread::sleep(std::time::Duration::from_millis(200 * (1 << attempt)));
            continue;
        }
        let response = response
            .error_for_status()
            .map_err(reqwest::Error::without_url)?;
        if range.is_some() && status.as_u16() != 206 {
            return Err("server ignored byte-range request".into());
        }
        if let Some((a, b)) = range {
            let header = response
                .headers()
                .get("content-range")
                .and_then(|h| h.to_str().ok())
                .unwrap_or("");
            if !header.starts_with(&format!("bytes {a}-")) {
                return Err("wrong Content-Range".into());
            }
            let bytes = response.bytes()?.to_vec();
            if let Some(b) = b {
                if bytes.len() as u64 != b - a + 1 {
                    return Err("truncated byte range".into());
                }
            }
            return Ok(bytes);
        }
        return Ok(response.bytes()?.to_vec());
    }
    Err(format!("HTTP retry limit for {url}").into())
}
fn select_ranges(text: &str, records: &Value) -> Result<Vec<(u64, Option<u64>)>> {
    let selectors: Vec<_> = records
        .as_array()
        .ok_or("index needs select.records")?
        .iter()
        .map(|r| {
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>((
                regex::Regex::new(&format!("^(?:{})$", r[0].as_str().ok_or("variable regex")?))?,
                regex::Regex::new(&format!("^(?:{})$", r[1].as_str().ok_or("level regex")?))?,
            ))
        })
        .collect::<Result<_>>()?;
    let lines: Vec<_> = text
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.split(':').collect::<Vec<_>>())
        .collect();
    let offsets: Vec<u64> = lines
        .iter()
        .map(|line| {
            line.get(1)
                .ok_or("inventory offset")?
                .parse()
                .map_err(Into::into)
        })
        .collect::<Result<_>>()?;
    if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err("inventory offsets must not decrease".into());
    }
    if lines.iter().any(|line| line.len() < 5) {
        return Err("malformed NCEP inventory".into());
    }
    // Several fields can share one GRIB2 message. Include its full byte span
    // when any submessage is selected; a zero-length range loses that field.
    let mut ranges = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let mut end = i + 1;
        while end < lines.len() && offsets[end] == offsets[i] {
            end += 1;
        }
        if lines[i..end].iter().any(|line| {
            selectors
                .iter()
                .any(|(v, l)| v.is_match(line[3]) && l.is_match(line[4]))
        }) {
            ranges.push((offsets[i], offsets.get(end).map(|n| n - 1)));
        }
        i = end;
    }
    if ranges.is_empty() {
        return Err("surface selectors matched no inventory records".into());
    }
    Ok(ranges)
}
fn cds(client: &Client, product: &Value, d: NaiveDateTime) -> Result<(Vec<u8>, Value)> {
    // No credential files are opened. Authentication is supplied by the caller's environment.
    let key =
        std::env::var("CDS_API_KEY").map_err(|_| "CDS_API_KEY is required for CDS acquisition")?;
    let base =
        std::env::var("CDS_API_URL").unwrap_or("https://cds.climate.copernicus.eu/api".into());
    let dataset = product["cds"]["dataset"].as_str().ok_or("CDS dataset")?;
    let mut inputs = product["cds"].clone();
    inputs.as_object_mut().unwrap().remove("dataset");
    for (key, value) in [
        ("year", d.format("%Y").to_string()),
        ("month", d.format("%m").to_string()),
        ("day", d.format("%d").to_string()),
        ("time", d.format("%H:00").to_string()),
    ] {
        inputs[key] = json!([value]);
    }
    inputs["product_type"] = json!(["reanalysis"]);
    inputs["data_format"] = json!("grib");
    inputs["download_format"] = json!("unarchived");
    let response: Value = client
        .post(format!("{base}/retrieve/v1/processes/{dataset}/execution"))
        .header("PRIVATE-TOKEN", &key)
        .json(&json!({"inputs":inputs}))
        .send()?
        .error_for_status()?
        .json()?;
    let id = response["jobID"]
        .as_str()
        .or(response["request_id"].as_str())
        .ok_or("CDS did not return jobID")?;
    let job_url = format!("{base}/retrieve/v1/jobs/{id}");
    let start = Instant::now();
    loop {
        if start.elapsed().as_secs() > 3600 {
            return Err("CDS job exceeded one hour".into());
        }
        let job: Value = client
            .get(&job_url)
            .header("PRIVATE-TOKEN", &key)
            .send()?
            .error_for_status()?
            .json()?;
        match job["status"].as_str().unwrap_or("") {
            "successful" => break,
            "failed" | "dismissed" => return Err(format!("CDS job {id} failed").into()),
            _ => std::thread::sleep(std::time::Duration::from_secs(2)),
        }
    }
    let result: Value = client
        .get(format!("{job_url}/results"))
        .header("PRIVATE-TOKEN", &key)
        .send()?
        .error_for_status()?
        .json()?;
    let href = result["asset"]["value"]["href"]
        .as_str()
        .or(result["asset"]["href"].as_str())
        .ok_or("CDS results missing asset href")?;
    let bytes = get(client, href, None)?;
    Ok((
        bytes,
        json!({"dataset":dataset,"job_id":id,"request":inputs}),
    ))
}
fn acquire(
    client: &Client,
    product: &Value,
    d: NaiveDateTime,
    path: &Path,
    staged: Option<&Path>,
) -> Result<Value> {
    if let Some(file) = staged
        .map(|p| p.join(path.file_name().unwrap()))
        .filter(|p| p.is_file())
    {
        let bytes = fs::read(&file)?;
        fs::write(path, &bytes)?;
        return Ok(json!({"staged":file,"bytes":bytes.len(),"sha256":digest(&bytes)}));
    }
    if product["transport"] == "cds" {
        let (bytes, provenance) = cds(client, product, d)?;
        fs::write(path, &bytes)?;
        return Ok(json!({"cds":provenance,"bytes":bytes.len(),"sha256":digest(&bytes)}));
    }
    let mut errors = Vec::new();
    for url in urls(&product["url"], d) {
        let result = (|| -> Result<Value> {
            let ranges = if product.get("index").is_some() {
                let mut inventory = None;
                for index in urls(&product["index"]["url"], d) {
                    if let Ok(b) = get(client, &index, None) {
                        inventory = Some(b);
                        break;
                    }
                }
                select_ranges(
                    std::str::from_utf8(&inventory.ok_or("no inventory mirror answered")?)?,
                    &product["index"]["select"]["records"],
                )?
            } else {
                Vec::new()
            };
            let bytes = if ranges.is_empty() {
                get(client, &url, None)?
            } else {
                ranges
                    .par_iter()
                    .map(|r| get(client, &url, Some(*r)))
                    .collect::<Result<Vec<_>>>()?
                    .concat()
            };
            fs::write(path, &bytes)?;
            Ok(json!({"url":url,"ranges":ranges,"bytes":bytes.len(),"sha256":digest(&bytes)}))
        })();
        match result {
            Ok(v) => return Ok(v),
            Err(e) => errors.push(e.to_string()),
        }
    }
    Err(format!("surface acquisition failed: {}", errors.join("; ")).into())
}
fn writer(
    path: &Path,
    n: usize,
    records: usize,
    start: &str,
    snow: bool,
) -> Result<NcClassicWriter> {
    let mut vars = vec![
        NcVarDef::new("xtime", NcType::Char, vec![0, 2]),
        NcVarDef::new("Time", NcType::Float, vec![0]).with_attrs(vec![NcAttr::text(
            "units",
            format!("seconds since {}", start.replace('_', " ")),
        )]),
        NcVarDef::new("sst", NcType::Float, vec![0, 1])
            .with_attrs(vec![NcAttr::text("units", "K")]),
        NcVarDef::new("xice", NcType::Float, vec![0, 1])
            .with_attrs(vec![NcAttr::text("units", "1")]),
        NcVarDef::new("record_provenance", NcType::Char, vec![0, 3]),
    ];
    if snow {
        vars.push(
            NcVarDef::new("snowh", NcType::Float, vec![0, 1])
                .with_attrs(vec![NcAttr::text("units", "m")]),
        );
    }
    Ok(NcClassicWriter::create(
        path,
        NcFormat::Data64,
        vec![
            NcDim::record("Time"),
            NcDim::fixed("nCells", n),
            NcDim::fixed("StrLen", 64),
            NcDim::fixed("ProvenanceLen", 4096),
        ],
        vec![NcAttr::text(
            "source",
            "rw_sfc_series; MPAS-A v8.4.2 case-8 Rust remap",
        )],
        vars,
        records as u64,
    )?)
}
fn put(
    writer: &mut NcClassicWriter,
    record: usize,
    time: &str,
    elapsed: f32,
    sst: &[f32],
    ice: &[f32],
    snow: Option<&[f32]>,
    provenance: &Value,
) -> Result<()> {
    let mut text = time.as_bytes().to_vec();
    text.resize(64, b' ');
    let mut prov = serde_json::to_vec(provenance)?;
    if prov.len() > 4096 {
        return Err("record provenance exceeds 4096 bytes".into());
    }
    prov.resize(4096, b' ');
    for (name, data) in [
        ("xtime", NcData::Chars(&text)),
        ("Time", NcData::Floats(&[elapsed])),
        ("sst", NcData::Floats(sst)),
        ("xice", NcData::Floats(ice)),
        ("record_provenance", NcData::Chars(&prov)),
    ] {
        writer.put_record(name, record as u64, data)?;
    }
    if let Some(values) = snow {
        writer.put_record("snowh", record as u64, NcData::Floats(values))?;
    }
    Ok(())
}
fn record_dates(start: NaiveDateTime, hours: i64, interval: i64) -> Result<Vec<NaiveDateTime>> {
    if hours < 0 || interval < 1 {
        return Err("hours must be nonnegative and interval positive".into());
    }
    let last = hours / interval + i64::from(hours % interval != 0);
    last.checked_mul(interval)
        .and_then(Duration::try_hours)
        .and_then(|offset| start.checked_add_signed(offset))
        .ok_or("surface window exceeds the supported calendar")?;
    (0..=last)
        .map(|i| {
            i.checked_mul(interval)
                .and_then(Duration::try_hours)
                .and_then(|offset| start.checked_add_signed(offset))
                .ok_or_else(|| "surface window exceeds the supported calendar".into())
        })
        .collect()
}
fn decode_budget(threads: usize, records: usize) -> Vec<usize> {
    let workers = threads.min(records);
    (0..workers)
        .map(|i| threads / workers + usize::from(i < threads % workers))
        .collect()
}
fn run() -> Result<()> {
    let argv: Vec<_> = std::env::args().skip(1).collect();
    if argv.first().is_some_and(|arg| arg == "--decode") {
        return intermediate::run(argv[1..].to_vec()).map_err(Into::into);
    }
    if argv == ["--abi"] {
        println!("rw-sfc-series/v1 native-netcdf xtime sst xice snowh provenance forecast-cycle");
        return Ok(());
    }
    if argv == ["--help"] {
        println!(
            "rw_sfc_series --table TABLE --source ID --cycle YYYY-MM-DDTHH --hours HOURS --static STATIC --out SERIES --receipt JSON --vtable-root DATA [--jobs N] [--threads N] [--interval-hours N] [--staged DIR]"
        );
        return Ok(());
    }
    if argv.len() % 2 != 0 {
        return Err("arguments require --name value pairs; use --help".into());
    }
    let known = [
        "--table",
        "--source",
        "--cycle",
        "--hours",
        "--static",
        "--out",
        "--receipt",
        "--vtable-root",
        "--jobs",
        "--threads",
        "--interval-hours",
        "--staged",
        "--forecast-cycle",
    ];
    let mut seen = std::collections::BTreeSet::new();
    for pair in argv.chunks(2) {
        if !known.contains(&pair[0].as_str()) || !seen.insert(&pair[0]) {
            return Err(format!(
                "unknown or repeated argument {}; the intended setting would be lost",
                pair[0]
            )
            .into());
        }
    }
    let args: BTreeMap<_, _> = argv
        .chunks(2)
        .map(|p| (p[0].as_str(), p[1].as_str()))
        .collect();
    let required = |k: &str| args.get(k).copied().ok_or_else(|| format!("missing {k}"));
    let threads: usize = args
        .get("--threads")
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(std::thread::available_parallelism()?.get());
    let jobs: usize = args
        .get("--jobs")
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(threads.min(16));
    if jobs == 0 || threads == 0 {
        return Err("jobs and threads must be positive".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let fetch_pool = rayon::ThreadPoolBuilder::new().num_threads(jobs).build()?;
    let begin = Instant::now();
    let table: Value = serde_json::from_slice(&fs::read(required("--table")?)?)?;
    let source = required("--source")?;
    let row = table["sources"]
        .as_array()
        .ok_or("source table")?
        .iter()
        .find(|r| {
            r["id"] == source
                || r["aliases"]
                    .as_array()
                    .is_some_and(|a| a.iter().any(|v| v == source))
        })
        .ok_or("source not in table")?;
    let forecast_cycle = args.get("--forecast-cycle").map(|text| date(text)).transpose()?;
    let series = if forecast_cycle.is_some() {
        row["surface_series"].get("forecast").ok_or("source has no surface forecast row")?
    } else { &row["surface_series"] };
    let decode = series.get("decode").unwrap_or(&row["decode"]);
    let interval: i64 = args
        .get("--interval-hours")
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(
            series["interval_hours"]
                .as_i64()
                .ok_or("row has no surface-series cadence")?,
        );
    let hours: i64 = required("--hours")?.parse()?;
    if hours < 0 || interval < 1 {
        return Err("hours must be nonnegative and interval positive".into());
    }
    let start = date(required("--cycle")?)?;
    if let Some(cycle) = forecast_cycle {
        let last_lead = (start-cycle).num_hours() + hours;
        if cycle > start || last_lead > series["max_hours"].as_i64().ok_or("surface forecast maximum lead")? {
            return Err("surface forecast does not cover the requested window".into());
        }
    }
    let cadence = series["interval_hours"].as_i64().ok_or("surface cadence")?;
    if start.minute() != 0
        || start.second() != 0
        || (start.and_utc().timestamp() / 3600) % cadence != 0
        || interval % cadence != 0
    {
        return Err(
            "cycle and interval must align to the source's surface-analysis cadence".into(),
        );
    }
    let dates = record_dates(start, hours, interval)?;
    let out = PathBuf::from(required("--out")?);
    let receipt_path = PathBuf::from(required("--receipt")?);
    if out.exists() || receipt_path.exists() {
        return Err("output or receipt already exists; choose a fresh path".into());
    }
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let scratch = parent.join(format!(".sfc-series-{}", std::process::id()));
    fs::create_dir(&scratch)?;
    let mut deletion = Vec::new();
    let mut receipt = json!({"schema":"rw-sfc-series/v1","source":row["id"],"cycle":stamp(start),"hours":hours,"threads":threads,"fetch_concurrency":jobs,"records":[],"stage_seconds":{},"deleted":[]});
    let partial = scratch.join("series.partial");
    let result = (|| -> Result<()> {
        let products = series["products"]
            .as_array()
            .ok_or("row has no surface_series.products")?;
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(180))
            .user_agent("rw-sfc-series/1")
            .build()?;
        let t = Instant::now();
        let plans=fetch_pool.install(|| dates.par_iter().enumerate().map(|(i,d)| -> Result<_> {
            let mut provenance=Vec::new(); let mut files=Vec::new();
            for (j,template) in products.iter().enumerate() {
                let resolved = forecast_cycle.map(|cycle| forecast_product(template, cycle, *d));
                let product = resolved.as_ref().unwrap_or(template);
                let filename=render(product["filename"].as_str().ok_or("product filename")?,*d);
                if Path::new(&filename).components().count()!=1 {return Err("product filename must be a basename".into());}
                let path=scratch.join(format!("{i}-{j}-{filename}"));
                let staged=args.get("--staged").map(Path::new);
                // Staged source filenames are the table's names, not scratch names.
                let staged_file=staged.map(|s|s.join(&filename));
                let prov=if let Some(file)=staged_file.filter(|p|p.is_file()) {
                    let bytes=fs::read(&file)?;fs::write(&path,&bytes)?;
                    json!({"staged":file,"bytes":bytes.len(),"sha256":digest(&bytes)})
                } else {acquire(&client,product,*d,&path,None)?};
                provenance.push(prov);files.push(path);
            }
            let source_cycle = forecast_cycle.unwrap_or(*d);
            Ok((*d,files,json!({"valid_time":stamp(*d),"source_cycle":stamp(source_cycle),"lead_hours":(*d-source_cycle).num_hours(),"source":row["id"],"products":provenance})))
        }).collect::<Result<Vec<_>>>())?;
        receipt["stage_seconds"]["fetch"] = json!(t.elapsed().as_secs_f64());
        let t = Instant::now();
        let decoder_threads = decode_budget(threads, plans.len());
        let decode_workers = decoder_threads.len();
        let decode_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(decode_workers)
            .build()?;
        let decoded = decode_pool.install(|| {
            plans
                .par_iter()
                .enumerate()
                .map(|(i, (d, files, prov))| -> Result<_> {
                    let met = scratch.join(format!("SFC:{i}"));
                    let log = scratch.join(format!("decode-{i}.log"));
                    let vtable = Path::new(required("--vtable-root")?)
                        .join(decode["vtable"].as_str().ok_or("vtable")?);
                    let output = Command::new(std::env::current_exe()?)
                        .args(["--decode", "--vtable"])
                        .arg(vtable)
                        .args([
                            "--date",
                            &stamp(*d),
                            "--map-source",
                            decode["map_source"].as_str().ok_or("map_source")?,
                            "--out",
                        ])
                        .arg(&met)
                        .args(files)
                        .env(
                            "RAYON_NUM_THREADS",
                            decoder_threads[i % decode_workers].to_string(),
                        )
                        .output()?;
                    fs::write(&log, [output.stdout, output.stderr].concat())?;
                    if !output.status.success() {
                        return Err(format!("Rust intermediate failed: {}", log.display()).into());
                    }
                    Ok((*d, met, prov.clone()))
                })
                .collect::<Result<Vec<_>>>()
        })?;
        receipt["stage_seconds"]["decode"] = json!(t.elapsed().as_secs_f64());
        receipt["decode_workers"] = json!(decode_workers);
        receipt["decoder_threads_per_worker"] =
            if decoder_threads.iter().all(|n| *n == decoder_threads[0]) {
                json!(decoder_threads[0])
            } else {
                json!(decoder_threads)
            };
        receipt["decode_worker_thread_budgets"] = json!(decoder_threads);
        receipt["coverage_end"] = json!(stamp(*dates.last().ok_or("no surface timestamps")?));
        receipt["remap_threads"] = json!(threads);
        let t = Instant::now();
        let (lat, lon, land) = remap::read_mesh(Path::new(required("--static")?))?;
        let mut writer_out = None;
        let mut snow_present = None;
        let mut remap_seconds = 0.;
        let mut write_seconds = 0.;
        for batch in decoded.chunks(jobs) {
            let before = Instant::now();
            let fields = batch
                .par_iter()
                .map(|(_, path, _)| remap::remap(path, &lat, &lon, &land).map_err(Into::into))
                .collect::<Result<Vec<_>>>()?;
            remap_seconds += before.elapsed().as_secs_f64();
            let before = Instant::now();
            for ((d, _, prov), (sst, ice, snow)) in batch.iter().zip(fields) {
                let have_snow = snow.is_some();
                if snow_present.is_some_and(|s| s != have_snow) {
                    return Err("snow depth is missing from part of the series".into());
                }
                if writer_out.is_none() {
                    writer_out = Some(writer(
                        &partial,
                        land.len(),
                        dates.len(),
                        &stamp(start),
                        have_snow,
                    )?);
                    snow_present = Some(have_snow);
                }
                let record = receipt["records"].as_array().unwrap().len();
                put(
                    writer_out.as_mut().unwrap(),
                    record,
                    &stamp(*d),
                    (*d - start).num_seconds() as f32,
                    &sst,
                    &ice,
                    snow.as_deref(),
                    prov,
                )?;
                receipt["records"]
                    .as_array_mut()
                    .unwrap()
                    .push(prov.clone());
            }
            write_seconds += before.elapsed().as_secs_f64();
        }
        writer_out.ok_or("no records")?.finish()?;
        receipt["stage_seconds"]["remap"] = json!(remap_seconds);
        receipt["stage_seconds"]["write"] = json!(write_seconds);
        receipt["stage_seconds"]["mesh_and_stream"] = json!(t.elapsed().as_secs_f64());
        receipt["n_cells"] = json!(land.len());
        receipt["output_bytes"] = json!(fs::metadata(&partial)?.len());
        fs::rename(&partial, &out)?;
        Ok(())
    })();
    // Only this invocation's freshly created scratch files are removed.
    // Logs remain beside the receipt if decode failed.
    for entry in fs::read_dir(&scratch)? {
        let path = entry?.path();
        if path.is_file() && (result.is_ok() || path.extension().is_none_or(|e| e != "log")) {
            deletion.push(json!({"path":path,"bytes":fs::metadata(&path)?.len()}));
            fs::remove_file(path)?;
        }
    }
    if partial.is_file() {
        deletion.push(json!({"path":partial,"bytes":fs::metadata(&partial)?.len()}));
        fs::remove_file(&partial)?;
    }
    if fs::read_dir(&scratch)?.next().is_none() {
        fs::remove_dir(&scratch)?;
    }
    receipt["deleted"] = json!(deletion);
    receipt["wall_seconds"] = json!(begin.elapsed().as_secs_f64());
    receipt["status"] = json!(if result.is_ok() { "complete" } else { "failed" });
    if let Err(ref error) = result {
        receipt["error"] = json!(error.to_string());
    }
    fs::write(receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    result
}
fn main() {
    if let Err(error) = run() {
        eprintln!("rw_sfc_series: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn forecast_dates_preserve_cycle_directory_and_valid_filename() -> Result<()> {
        let cycle = date("2024-02-29T18")?;
        let valid = cycle + Duration::hours(30);
        let value = json!({"url":"/{yyyymmdd}/{yyyymmddhh}/flxf{valid_yyyymmddhh}.01.{yyyymmddhh}.grb2", "lead":"{lead:03d}"});
        let result = forecast_product(&value, cycle, valid);
        assert_eq!(result["url"], "/20240229/2024022918/flxf2024030200.01.2024022918.grb2");
        assert_eq!(result["lead"], "030");
        Ok(())
    }
    #[test]
    fn window_covers_non_aligned_endpoint_and_leap_day() -> Result<()> {
        let start = date("2024-02-29T18")?;
        assert_eq!(record_dates(start, 0, 6)?, vec![start]);
        assert_eq!(record_dates(start, 6, 6)?.len(), 2);
        let dates = record_dates(start, 7, 6)?;
        assert_eq!(dates.len(), 3);
        assert_eq!(stamp(dates[2]), "2024-03-01_06:00:00");
        assert!(record_dates(start, -1, 6).is_err());
        assert!(record_dates(start, 1, 0).is_err());
        assert!(record_dates(start, i64::MAX, 6).is_err());
        Ok(())
    }
    #[test]
    fn decoder_budget_uses_cores_for_short_series() {
        assert_eq!(decode_budget(24, 1), vec![24]);
        assert_eq!(decode_budget(24, 2), vec![12, 12]);
        assert_eq!(decode_budget(24, 121), vec![1; 24]);
        assert_eq!(decode_budget(5, 3), vec![2, 2, 1]);
    }
    #[test]
    fn record_round_trip() -> Result<()> {
        let path = std::env::temp_dir().join(format!("surface-record-{}.nc", std::process::id()));
        let time = "2024-02-29_18:00:00";
        let mut w = writer(&path, 3, 1, time, true)?;
        put(
            &mut w,
            0,
            time,
            0.,
            &[271., 290., 301.],
            &[1., 0., 0.],
            Some(&[0., 0.1, 0.2]),
            &json!({"source_cycle":time}),
        )?;
        w.finish()?;
        let f = netcrust::File::open(&path)?;
        assert_eq!(
            f.read_array_f64_first_record_or_all("sst")?.into_values(),
            vec![271., 290., 301.]
        );
        assert_eq!(
            f.read_array_f64_first_record_or_all("xice")?.into_values(),
            vec![1., 0., 0.]
        );
        drop(f);
        fs::remove_file(path)?;
        Ok(())
    }
    #[test]
    fn shared_grib_submessages_are_fetched_once_in_full() -> Result<()> {
        let text = "1.1:0:d=2023033100:UGRD:500 mb:anl:\n1.2:0:d=2023033100:VGRD:500 mb:anl:\n2:100:d=2023033100:TMP:surface:anl:\n";
        assert_eq!(
            select_ranges(text, &json!([["UGRD", "500 mb"]]))?,
            vec![(0, Some(99))]
        );
        Ok(())
    }
    #[test]
    fn inventory_selection_and_leap_day() -> Result<()> {
        let text = "1:0:d=2023033100:TMP:surface:anl:\n2:100:d=2023033100:TMP:2 m above ground:anl:\n3:200:d=2023033100:ICEC:surface:anl:\n";
        assert_eq!(
            select_ranges(text, &json!([["^(TMP|ICEC)$", "^surface$"]]))?,
            vec![(0, Some(99)), (200, None)]
        );
        assert_eq!(
            stamp(date("2024-02-29T18")? + Duration::hours(6)),
            "2024-03-01_00:00:00"
        );
        Ok(())
    }
}
