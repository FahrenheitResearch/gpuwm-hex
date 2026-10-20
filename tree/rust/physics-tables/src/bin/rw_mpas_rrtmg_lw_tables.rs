//! MPAS: mpas_atmphys_rrtmg_lwinit.F:59-1593 (lwlookuptable/lw_kgb01..16).
//! Decode the pristine SINGLE_PRECISION Fortran table records. No reduction
//! or floating arithmetic occurs here; CUDA performs native initialization.
use std::io::{self, Read, Write};
use std::path::Path;
const BANDS: &[&[(&str,usize)]] = &[
    &[("fracrefao",16),("fracrefbo",16),("kao",1040),("kbo",3760),("kao_mn2",304),("kbo_mn2",304),("selfrefo",160),("forrefo",64)],
    &[("fracrefao",16),("fracrefbo",16),("kao",1040),("kbo",3760),("selfrefo",160),("forrefo",64)],
    &[("fracrefao",144),("fracrefbo",80),("kao",9360),("kbo",18800),("kao_mn2o",2736),("kbo_mn2o",1520),("selfrefo",160),("forrefo",64)],
    &[("fracrefao",144),("fracrefbo",80),("kao",9360),("kbo",18800),("selfrefo",160),("forrefo",64)],
    &[("fracrefao",144),("fracrefbo",80),("kao",9360),("kbo",18800),("kao_mo3",2736),("ccl4o",16),("selfrefo",160),("forrefo",64)],
    &[("fracrefao",16),("kao",1040),("kao_mco2",304),("cfc11adjo",16),("cfc12o",16),("selfrefo",160),("forrefo",64)],
    &[("fracrefao",144),("fracrefbo",16),("kao",9360),("kbo",3760),("kao_mco2",2736),("kbo_mco2",304),("selfrefo",160),("forrefo",64)],
    &[("fracrefao",16),("fracrefbo",16),("kao",1040),("kbo",3760),("kao_mco2",304),("kbo_mco2",304),("kao_mn2o",304),("kbo_mn2o",304),("kao_mo3",304),("cfc12o",16),("cfc22adjo",16),("selfrefo",160),("forrefo",64)],
    &[("fracrefao",144),("fracrefbo",16),("kao",9360),("kbo",3760),("kao_mn2o",2736),("kbo_mn2o",304),("selfrefo",160),("forrefo",64)],
    &[("fracrefao",16),("fracrefbo",16),("kao",1040),("kbo",3760),("selfrefo",160),("forrefo",64)],
    &[("fracrefao",16),("fracrefbo",16),("kao",1040),("kbo",3760),("kao_mo2",304),("kbo_mo2",304),("selfrefo",160),("forrefo",64)],
    &[("fracrefao",144),("kao",9360),("selfrefo",160),("forrefo",64)],
    &[("fracrefao",144),("fracrefbo",16),("kao",9360),("kao_mco2",2736),("kao_mco",2736),("kbo_mo3",304),("selfrefo",160),("forrefo",64)],
    &[("fracrefao",16),("fracrefbo",16),("kao",1040),("kbo",3760),("selfrefo",160),("forrefo",64)],
    &[("fracrefao",144),("kao",9360),("kao_mn2",2736),("selfrefo",160),("forrefo",64)],
    &[("fracrefao",144),("fracrefbo",16),("kao",9360),("kbo",3760),("selfrefo",160),("forrefo",64)],
];

fn decode(mut input: impl Read) -> io::Result<(String, Vec<u8>)> {
    let mut output = Vec::new();
    let mut manifest = String::from("{\"format\":\"mpas-rrtmg-lw-raw-f32-v1\",\"arrays\":[");
    let mut first = true;
    for (band, arrays) in BANDS.iter().enumerate() {
        let expected: usize = arrays.iter().map(|(_, n)| n * 4).sum();
        let mut marker = [0u8; 4];
        input.read_exact(&mut marker)?;
        let length = u32::from_be_bytes(marker) as usize;
        if length != expected {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("band {} record length {} != {}", band+1, length, expected)));
        }
        let mut payload = vec![0u8; length];
        input.read_exact(&mut payload)?;
        input.read_exact(&mut marker)?;
        if u32::from_be_bytes(marker) as usize != length {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "record trailer mismatch"));
        }
        for (name, words) in *arrays {
            if !first { manifest.push(','); }
            first = false;
            manifest.push_str(&format!("{{\"module\":\"rrlw_kg{:02}\",\"name\":\"{}\",\"offset\":{},\"words\":{}}}",
                band+1, name, output.len(), words));
            // Output offsets follow pristine READ order, with Fortran storage
            // unchanged. Rewrite byte order without loading a float.
            let offset = output.len();
            output.resize(offset + words * 4, 0);
        }
        let offset = output.len()-length;
        for (source, destination) in payload.chunks_exact(4).zip(output[offset..].chunks_exact_mut(4)) {
            destination.copy_from_slice(&u32::from_be_bytes(source.try_into().unwrap()).to_le_bytes());
        }
    }
    let mut extra = [0u8;1];
    if input.read(&mut extra)? != 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "unexpected trailing table data"));
    }
    manifest.push_str("]}");
    Ok((manifest,output))
}

fn main() -> io::Result<()> {
    let path = std::env::args_os().nth(1).ok_or_else(||
        io::Error::new(io::ErrorKind::InvalidInput,"RRTMG_LW_DATA path required"))?;
    let (manifest, payload) = decode(std::fs::File::open(Path::new(&path))?)?;
    let mut out = io::BufWriter::new(io::stdout().lock());
    writeln!(out,"{}",manifest)?;
    out.write_all(&payload)?;
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn refuses_invalid_first_record_before_allocating_payload() {
        assert_eq!(decode(&[0u8;4][..]).unwrap_err().kind(),io::ErrorKind::InvalidData);
    }
    #[test]
    fn refuses_truncated_record_marker() {
        assert_eq!(decode(&[0u8;3][..]).unwrap_err().kind(),io::ErrorKind::UnexpectedEof);
    }
}
