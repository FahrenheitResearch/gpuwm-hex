//! Exact differing ORCL0001 word sets for proposed native-defect witnesses.
//! These words count in the comparison; this tool never supplies replay inputs.
use std::{env, fs, path::Path};

fn main() {
    let args: Vec<_> = env::args().collect();
    assert!(args.len() >= 4, "native directory, candidate directory, field names required");
    let native = Path::new(&args[1]);
    let candidate = Path::new(&args[2]);
    let mut files: Vec<_> = fs::read_dir(candidate).unwrap().map(|v|v.unwrap().path())
        .filter(|p| {
            let name = p.file_name().unwrap().to_str().unwrap();
            name.contains(".post.") && args[3..].iter().any(|f| name.ends_with(&format!(".post.{}.bin", f)))
        }).collect();
    files.sort();
    assert!(!files.is_empty(), "No candidate records");
    println!("{{\"scope\":\"exact differing native/candidate boundary word sets; no exclusion applied\",\"excluded_words\":0,\"records\":[");
    let mut total = 0;
    for (row, path) in files.iter().enumerate() {
        let name = path.file_name().unwrap().to_str().unwrap();
        let a = fs::read(native.join(name)).unwrap();
        let b = fs::read(path).unwrap();
        assert_eq!(a.len(), b.len());
        assert_eq!(&a[..8], b"ORCL0001");
        let integer = |i: usize| i32::from_le_bytes(a[8+i*4..12+i*4].try_into().unwrap());
        let offset = 860 + integer(16) as usize * 4;
        assert_eq!(&a[..offset], &b[..offset], "Record metadata changed: {}", name);
        let width = match integer(0) {1|3|4=>4, 2|6=>8, _=>panic!("Unsupported boundary word type")};
        assert_eq!((a.len()-offset) % width, 0);
        if row > 0 {println!(",");}
        print!("{{\"file\":\"{}\",\"word_bytes\":{},\"words\":{},\"different\":[", name, width, (a.len()-offset)/width);
        let mut count = 0;
        for (index, (x, y)) in a[offset..].chunks_exact(width).zip(b[offset..].chunks_exact(width)).enumerate() {
            if x == y {continue;}
            if count > 0 {print!(",");}
            let hex = |v: &[u8]|v.iter().rev().map(|b|format!("{:02x}",b)).collect::<String>();
            print!("{{\"index\":{},\"native_hex\":\"{}\",\"candidate_hex\":\"{}\"}}", index, hex(x), hex(y));
            count += 1;
        }
        print!("],\"different_words\":{}}}", count);
        total += count;
    }
    println!("],\"total_different_words\":{}}}", total);
}
