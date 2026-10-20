//! Lossless transport for pristine MPAS standalone column records.
//! lw_binio.F90 little-endian records or SWD1 big-endian records.
use std::{env, fs, io::{self, Read}, path::PathBuf, collections::HashSet};

fn i32_at(data: &[u8], offset: &mut usize, big_endian: bool) -> io::Result<i32> {
    let bytes = data.get(*offset..*offset + 4)
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "record header"))?;
    *offset += 4;
    Ok(if big_endian { i32::from_be_bytes(bytes.try_into().unwrap()) }
       else { i32::from_le_bytes(bytes.try_into().unwrap()) })
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 3, "native_tagged_transport INPUT NEW_OUTPUT_DIRECTORY");
    let out = PathBuf::from(&args[2]);
    fs::create_dir(&out)?;
    let mut data = Vec::new();
    fs::File::open(&args[1])?.read_to_end(&mut data)?;
    let big_endian = data.starts_with(b"SWD1");
    let mut offset = if big_endian { 4 } else { 0 };
    let mut names = HashSet::new();
    let mut metadata = Vec::new();
    while offset < data.len() {
        let length = usize::try_from(i32_at(&data, &mut offset, big_endian)?).expect("positive name length");
        assert!(length > 0 && length <= 4096);
        let name = std::str::from_utf8(data.get(offset..offset + length)
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "name"))?).unwrap();
        offset += length;
        assert!(name.bytes().all(|c| c.is_ascii_alphanumeric() || b"/_-".contains(&c)));
        assert!(names.insert(name.to_string()), "duplicate record");
        let dtype = i32_at(&data, &mut offset, big_endian)?;
        assert!(dtype == 0 || dtype == 1);
        let rank = usize::try_from(i32_at(&data, &mut offset, big_endian)?).expect("nonnegative rank");
        assert!(rank <= 8);
        let mut dims = Vec::new();
        let mut words = 1usize;
        for _ in 0..rank {
            let dimension = usize::try_from(i32_at(&data, &mut offset, big_endian)?).expect("nonnegative dimension");
            words = words.checked_mul(dimension).expect("dimension overflow");
            dims.push(dimension);
        }
        let bytes = words.checked_mul(4).expect("payload overflow");
        let end = offset.checked_add(bytes).expect("offset overflow");
        let payload = data.get(offset..end)
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "payload"))?;
        // Byte swapping preserves every IEEE word, including NaN payloads.
        let mut decoded = payload.to_vec();
        if big_endian {
            for word in decoded.chunks_exact_mut(4) { word.reverse(); }
        }
        let filename = format!("{:06}.raw", metadata.len());
        fs::write(out.join(&filename), &decoded)?;
        // Integer records contain only launch/configuration metadata.
        let values = if dtype == 1 {
            let integers: Vec<i32> = decoded.chunks_exact(4)
                .map(|v| i32::from_le_bytes(v.try_into().unwrap())).collect();
            format!(",\"integer_metadata\":{:?}", integers)
        } else { String::new() };
        metadata.push(format!("{{\"name\":\"{}\",\"dtype\":{},\"shape\":{:?},\"words\":{},\"file\":\"{}\"{}}}", name, dtype, dims, words, filename, values));
        offset = end;
    }
    fs::write(out.join("manifest.json"), format!("{{\"schema\":\"mpas-native-tagged-columns/v1\",\"order\":\"Fortran\",\"records\":[{}]}}\n", metadata.join(",")))?;
    println!("{{\"records\":{},\"original_bytes\":{},\"complete\":true}}", metadata.len(), data.len());
    Ok(())
}
