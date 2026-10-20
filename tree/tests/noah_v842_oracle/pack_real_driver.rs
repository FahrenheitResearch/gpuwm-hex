//! Real initialized soil/surface states with explicitly synthetic atmosphere.
//! This is a boundary stress replay, never a native captured call or forecast.
use std::{env, fs};
include!("dump_layout.rs");
const SURFACES: &[(&str, &str)] = &[
    ("tsk", "skintemp"), ("tmn", "tmn"), ("xland", "xland"),
    ("xice", "xice"), ("vegfra", "vegfra"), ("shdmin", "shdmin"),
    ("shdmax", "shdmax"), ("snowc", "snowc"), ("albbck", "sfc_albbck"),
];
fn floats(path: &str) -> Vec<f32> {
    let b = fs::read(path).unwrap(); assert_eq!(b.len() % 4, 0);
    b.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect()
}
fn integers(path: &str) -> Vec<i32> {
    let b = fs::read(path).unwrap(); assert_eq!(b.len() % 4, 0);
    b.chunks_exact(4).map(|v| i32::from_le_bytes(v.try_into().unwrap())).collect()
}
fn slot(name: &str, k: usize) -> usize {
    FLOATS.iter().position(|&(n, s)| n == name && s == k).unwrap()
}
fn main() {
    let a: Vec<_> = env::args().collect();
    assert_eq!(a.len(), 5, "SYNTHETIC_DRIVER_PREFIX NATIVE_INIT_PREFIX DECODED_SURFACES OUTPUT_PREFIX");
    let base = floats(&format!("{}.f32", a[1]));
    let ibase = integers(&format!("{}.i32", a[1]));
    let init = floats(&format!("{}.f32", a[2]));
    let iinit = integers(&format!("{}.i32", a[2]));
    assert_eq!(base.len(), 1024 * FLOATS.len()); assert_eq!(ibase.len(), 1024 * 7);
    let n = init.len() / 15; assert_eq!(init.len(), n * 15); assert_eq!(iinit.len(), n * 4);
    assert_eq!(n % 4, 0); let physical = n / 4;
    let decoded: Vec<Vec<f64>> = (0..=SURFACES.len()).map(|k| {
        let b = fs::read(format!("{}/{k:04}.f64", a[3])).unwrap(); assert_eq!(b.len() % 8, 0);
        let v: Vec<_> = b.chunks_exact(8).map(|x| f64::from_le_bytes(x.try_into().unwrap())).collect();
        assert_eq!(v.len(), physical * if k == SURFACES.len() { 4 } else { 1 }); v
    }).collect();
    let mut output = Vec::with_capacity(n * FLOATS.len() * 4);
    let mut integer_output = Vec::with_capacity(n * 7 * 4);
    for c in 0..n {
        // Existing broad synthetic forcing; no invented conversion of real air.
        let b = 128 + c % (1024 - 128); let cell = c % physical;
        let mut x = base[b * FLOATS.len()..(b + 1) * FLOATS.len()].to_vec();
        let mut ix = ibase[b * 7..(b + 1) * 7].to_vec();
        for (j, &(name, _)) in SURFACES.iter().enumerate() { x[slot(name, 0)] = decoded[j][cell] as f32; }
        x[slot("albedo", 0)] = x[slot("albbck", 0)];
        for (j, name) in ["tslb", "smois", "sh2o"].iter().enumerate() {
            for k in 0..4 { x[slot(name, k)] = init[c * 15 + j * 4 + k]; }
        }
        for (j, name) in ["snoalb", "snow", "snowh"].iter().enumerate() { x[slot(name, 0)] = init[c * 15 + 12 + j]; }
        for k in 0..4 { x[slot("dzs", k)] = decoded[SURFACES.len()][cell * 4 + k] as f32; }
        ix[0] = iinit[c * 4]; ix[1] = iinit[c * 4 + 1];
        for v in x { output.extend_from_slice(&v.to_le_bytes()); }
        for v in ix { integer_output.extend_from_slice(&v.to_le_bytes()); }
    }
    fs::write(format!("{}.f32", a[4]), output).unwrap();
    fs::write(format!("{}.i32", a[4]), integer_output).unwrap();
    println!("{physical} real physical cells, {n} initialized-state driver replays; synthetic atmospheric forcing");
}
