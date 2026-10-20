//! Repair only invocation coordinates in the complete v78/v80 CUDA trace.
//! Pinned bl_mynn_subroutines.F90:553-559,1664-1670 calls length six times
//! with initialization, once otherwise. No scientific value is changed.
//! Every per-field/level cohort must validate against the erroneous 4/1
//! stride before its corrected 6/1 coordinate is admitted. No row is omitted.
use std::{collections::BTreeMap, env, fs::File, io::{BufRead, BufReader, BufWriter, Write}};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = env::args().collect();
    if a.len() != 5 { return Err("usage: repair <input> <output> <receipt> <columns>".into()); }
    let columns: usize = a[4].parse()?;
    if columns == 0 { return Err("columns must be positive".into()); }
    let mut totals: BTreeMap<String, usize> = BTreeMap::new();
    for line in BufReader::new(File::open(&a[1])?).lines() {
        let line = line?;
        let f: Vec<_> = line.split_whitespace().collect();
        if f.len() == 7 && f[0] == "R" && f[1] == "out:mym_length_entry_kts" {
            *totals.entry(f[2].into()).or_default() += 1;
        }
    }
    if totals.is_empty() { return Err("complete trace has no length calls".into()); }
    for n in totals.values() {
        if *n != columns && *n != 6 * columns { return Err("unexpected whole-driver length call count".into()); }
    }
    let mut counts: BTreeMap<(String,String,String,String),usize> = BTreeMap::new();
    let mut writer = BufWriter::new(File::create(&a[2])?);
    let (mut rows, mut changed) = (0usize,0usize);
    for line in BufReader::new(File::open(&a[1])?).lines() {
        let line = line?;
        let mut f: Vec<_> = line.split_whitespace().map(str::to_owned).collect();
        if f.len() == 7 && f[0] == "R" && f[1].starts_with("out:mym_length_") {
            let total = *totals.get(&f[2]).ok_or("length row outside admitted driver")?;
            let calls = total / columns;
            let key = (f[1].clone(),f[2].clone(),f[4].clone(),f[5].clone());
            let ordinal = counts.entry(key).or_default();
            let invocation = *ordinal / columns + 1;
            let column = *ordinal % columns;
            let old_stride = if calls == 6 {4} else {1};
            let old: usize = f[3].parse()?;
            if invocation > calls || old != column * old_stride + invocation {
                return Err(format!("trace cohort is not in validated CUDA lane order: {}",line).into());
            }
            let corrected = column * calls + invocation;
            if corrected != old { changed += 1; }
            f[3] = corrected.to_string();
            writeln!(writer,"{}",f.join(" "))?;
            *ordinal += 1;
        } else {
            writeln!(writer,"{}",line)?;
        }
        rows += 1;
    }
    for ((_,driver,_,_),count) in &counts {
        if *count != totals[driver] { return Err("incomplete per-field/level length cohort".into()); }
    }
    writer.flush()?;
    let mut receipt = File::create(&a[3])?;
    writeln!(receipt,"{{\"process_count\":1,\"columns\":{columns},\"rows\":{rows},\"changed_coordinate_rows\":{changed},\"validated_cohorts\":{},\"scientific_payload_words_changed\":0,\"omitted_rows\":0,\"excluded_words\":[],\"forecast_proven\":false}}",counts.len())?;
    println!("Validated {rows} complete rows; changed {changed} invocation coordinates, no payload words");
    Ok(())
}

fn main() {
    if let Err(e) = run() { eprintln!("{e}"); std::process::exit(2); }
}
