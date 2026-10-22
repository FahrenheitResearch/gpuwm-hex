//! Checked native array transport for the reference CUDA driver.
use crate::{load, nc::File, save};
use serde_json::{json, Value};
use std::{io::Write, path::Path};

pub fn run(path: &str) -> Result<(), String> {
    let plan: Value = load(path)?;
    let output = Path::new(plan["output"].as_str().ok_or("missing output")?);
    std::fs::create_dir_all(output).map_err(|e| e.to_string())?;
    let file = File::open(plan["input"].as_str().ok_or("missing input")?)?;
    let mut arrays = serde_json::Map::new();
    for item in plan["variables"].as_array().ok_or("missing variables")? {
        let name = item["name"].as_str().ok_or("missing name")?;
        if name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            return Err("invalid transport variable name".into());
        }
        let selection = if item["start"].is_null() { None } else {
            Some((item["start"].as_u64().ok_or("invalid start")? as usize,
                  item["count"].as_u64().ok_or("invalid count")? as usize))
        };
        let array = file.read_selection(name, selection)?;
        let target = output.join(format!("{name}.f64"));
        let mut writer = std::io::BufWriter::new(std::fs::File::create(&target).map_err(|e|e.to_string())?);
        for value in &array.values {
            writer.write_all(&value.to_le_bytes()).map_err(|e|e.to_string())?;
        }
        writer.flush().map_err(|e| e.to_string())?;
        arrays.insert(name.into(), json!({"path":target,"shape":array.shape,
            "dims":array.dims,"units":array.units,"dtype":"<f8","selection":item}));
    }
    save(output.join("arrays.json"), &json!({"schema":"gpuwm-hex.s2s-array-transport/v1",
        "input":plan["input"],"arrays":arrays,"status":"complete"}))
}
