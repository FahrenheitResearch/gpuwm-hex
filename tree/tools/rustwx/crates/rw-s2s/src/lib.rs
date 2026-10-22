//! Native S2S numerical work. Forecast-derived EOFs and synthetic reference
//! observations are never substituted for qualified fixed reference inputs.
pub mod mjo;
pub mod nc;
pub mod scores;
pub mod extract;
use serde_json::Value;
use std::path::Path;

// Keep schema decoding explicit without a proc-macro compiler dependency.
// Required fields reject null; absent Option fields decode as None.
#[macro_export]
macro_rules! deserialize_fields {
    ($name:ident { $($field:ident),* $(,)? }) => {
        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D:serde::Deserializer<'de>>(d:D)->Result<Self,D::Error> {
                let mut fields=<serde_json::Map<String,serde_json::Value> as serde::Deserialize>::deserialize(d)?;
                Ok(Self { $($field:serde_json::from_value(fields.remove(stringify!($field)).unwrap_or_else(|| {
                    if stringify!($field)=="minimum_bin_area_fraction" {serde_json::json!(0.9)}else{serde_json::Value::Null}
                })).map_err(serde::de::Error::custom)?),* })
            }
        }
    }
}

pub fn load<T: serde::de::DeserializeOwned>(path: impl AsRef<Path>) -> Result<T, String> {
    let p = path.as_ref();
    serde_json::from_slice(&std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?)
        .map_err(|e| format!("{}: {e}", p.display()))
}
pub fn save(path: impl AsRef<Path>, value: &Value) -> Result<(), String> {
    let p = path.as_ref();
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let tmp = p.with_extension("json.part");
    std::fs::write(
        &tmp,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    std::fs::rename(tmp, p).map_err(|e| e.to_string())
}
pub fn license(name: &str, opt_in: Option<&str>) -> Result<(), String> {
    if [
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "Apache-2.0",
        "CC0-1.0",
        "public-domain",
        "CC-BY-4.0",
    ]
    .contains(&name)
        || (!name.trim().is_empty() && opt_in == Some(name))
    {
        Ok(())
    } else {
        Err(format!(
            "reference data licence {name:?} needs explicit --s2s-reference-license {name} opt-in"
        ))
    }
}
pub fn finite(values: &[f64], label: &str) -> Result<(), String> {
    if values.is_empty() || values.iter().any(|x| !x.is_finite()) {
        Err(format!("{label}: empty or nonfinite values"))
    } else {
        Ok(())
    }
}
pub fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value[key]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("missing {key}"))
}
