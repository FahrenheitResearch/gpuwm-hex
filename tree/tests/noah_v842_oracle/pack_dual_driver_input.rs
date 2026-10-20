//! Byte-only transport into the strict standalone native driver conversion ABI.
use std::{env, fs, path::PathBuf};

fn main() {
    let arguments: Vec<String> = env::args().collect();
    if arguments.get(1).is_some_and(|value| value == "--extract-record") {
        assert_eq!(arguments.len(), 6, "--extract-record INPUT WIDTH INDEX OUTPUT");
        let bytes = fs::read(&arguments[2]).unwrap();
        let width: usize = arguments[3].parse().unwrap();
        let index: usize = arguments[4].parse().unwrap();
        assert_eq!(bytes.len() % (width * 4), 0);
        let start = index * width * 4;
        fs::write(&arguments[5], &bytes[start..start + width * 4]).unwrap();
        return;
    }
    assert_eq!(arguments.len(), 5, "INPUT METADATA CELLS OUTPUT");
    let input = PathBuf::from(&arguments[1]);
    let metadata = fs::read_to_string(&arguments[2]).unwrap();
    let cells: usize = arguments[3].parse().unwrap();
    let output = PathBuf::from(&arguments[4]);
    fs::create_dir_all(&output).unwrap();
    if input.is_file() {
        // Inverse opaque transport from native io_to's cell-major field ABI.
        let bytes = fs::read(&input).unwrap();
        let rows: Vec<Vec<&str>> = metadata.lines().map(|line| line.split('\t').collect()).collect();
        let width: usize = rows.iter().filter(|row| row[1] == "float")
            .map(|row| row[3].parse::<usize>().unwrap()).sum();
        assert_eq!(bytes.len(), cells * width * 4);
        let mut field_offset = 0;
        for row in rows {
            if row[1] != "float" { continue; }
            let levels: usize = row[3].parse().unwrap();
            let mut field = Vec::with_capacity(cells * levels * 4);
            for level in 0..levels {
                for cell in 0..cells {
                    let offset = (cell * width + field_offset + level) * 4;
                    field.extend_from_slice(&bytes[offset..offset + 4]);
                }
            }
            fs::write(output.join(format!("{}.post.bin", row[0])), field).unwrap();
            field_offset += levels;
        }
        println!("{{\"cells\":{cells},\"float_width\":{width},\"unpacked\":true}}");
        return;
    }
    let mut floats = Vec::new();
    let mut integers = Vec::new();
    for line in metadata.lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        assert_eq!(parts.len(), 4);
        let name = parts[0];
        let kind = parts[1];
        let source_levels: usize = parts[2].parse().unwrap();
        let target_levels: usize = parts[3].parse().unwrap();
        assert!(source_levels >= target_levels);
        let path = input.join(format!("{name}.input.bin"));
        let bytes = if path.is_file() {
            fs::read(path).unwrap()
        } else {
            assert!(matches!(name, "rainncv" | "raincv" | "sr"));
            vec![0; cells * source_levels * 4]
        };
        assert_eq!(bytes.len(), cells * source_levels * 4);
        let mut cell_major = Vec::with_capacity(bytes.len());
        for cell in 0..cells {
            for level in 0..source_levels {
                let offset = (level * cells + cell) * 4;
                cell_major.extend_from_slice(&bytes[offset..offset + 4]);
            }
        }
        fs::write(output.join(format!("{name}.cellmajor.bin")), cell_major).unwrap();
        match kind {
            "float" => floats.push((bytes, target_levels)),
            "int" => {
                assert_eq!(target_levels, 1);
                integers.push(bytes);
            }
            _ => panic!("unknown scalar kind"),
        }
    }
    assert_eq!(integers.len(), 2);
    let width: usize = floats.iter().map(|(_, width)| width).sum();
    let mut float_bytes = Vec::with_capacity(cells * width * 4);
    let mut int_bytes = Vec::with_capacity(cells * 2 * 4);
    for cell in 0..cells {
        for (source, levels) in &floats {
            for level in 0..*levels {
                let offset = (level * cells + cell) * 4;
                float_bytes.extend_from_slice(&source[offset..offset + 4]);
            }
        }
        for source in &integers {
            int_bytes.extend_from_slice(&source[cell * 4..cell * 4 + 4]);
        }
    }
    fs::write(output.join("fields.input.bin"), float_bytes).unwrap();
    fs::write(output.join("categories.input.bin"), int_bytes).unwrap();
    println!("{{\"cells\":{cells},\"float_width\":{width},\"integer_width\":2}}");
}
