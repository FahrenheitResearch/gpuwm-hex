//! MPAS-A v8.4.2 RRTMG_SW_DATA, FP32 big-endian sequential Fortran records.
//!
//! MPAS mpas_atmphys_rrtmg_swinit.F:46-1206 owns this record roster;
//! module_ra_rrtmg_sw.F:272-1170 owns the original coefficient dimensions.
//! This decoder changes byte order only. G-point reduction, interpolation
//! and lookup-table arithmetic belong to the CUDA initializer.
use std::fs;
use std::io::{self, Read};
use std::path::Path;

#[derive(Debug)]
pub struct Coefficient {
    pub name: String,
    pub shape: Vec<usize>,
    pub integer: bool,
    /// Unmodified IEEE-754 or integer words, in native Fortran array order.
    pub words: Vec<u32>,
}

struct Spec { name: &'static str, shape: &'static [usize], integer: bool }
macro_rules! real { ($name:literal) => { Spec { name: $name, shape: &[], integer: false } };
    ($name:literal, $($dim:expr),+) => { Spec { name: $name, shape: &[$($dim),+], integer: false } }; }
macro_rules! layer { () => { Spec { name: "layreffr", shape: &[], integer: true } }; }

// MPAS mpas_atmphys_rrtmg_swinit.F:145-1206, the fourteen complete READ lists.
fn record_specs(band: u8) -> Vec<Spec> {
    match band {
        // sw_kgb16:145-224; original dimensions SW:272-335.
        16 => vec![real!("rayl"), real!("strrat1"), layer!(), real!("kao",9,5,13,16),
            real!("kbo",5,47,16), real!("selfrefo",10,16), real!("forrefo",3,16), real!("sfluxrefo",16)],
        // sw_kgb17:227-307.
        17 => vec![real!("rayl"), real!("strrat"), layer!(), real!("kao",9,5,13,16),
            real!("kbo",5,5,47,16), real!("selfrefo",10,16), real!("forrefo",4,16), real!("sfluxrefo",16,5)],
        // sw_kgb18:310-389; sw_kgb19:392-471; sw_kgb22:640-719.
        18|19|22 => vec![real!("rayl"), real!("strrat"), layer!(), real!("kao",9,5,13,16),
            real!("kbo",5,47,16), real!("selfrefo",10,16), real!("forrefo",3,16), real!("sfluxrefo",16,9)],
        // sw_kgb20:474-555.
        20 => vec![real!("rayl"), layer!(), real!("absch4o",16), real!("kao",5,13,16),
            real!("kbo",5,47,16), real!("selfrefo",10,16), real!("forrefo",4,16), real!("sfluxrefo",16)],
        // sw_kgb21:558-637.
        21 => vec![real!("rayl"), real!("strrat"), layer!(), real!("kao",9,5,13,16),
            real!("kbo",5,5,47,16), real!("selfrefo",10,16), real!("forrefo",4,16), real!("sfluxrefo",16,9)],
        // sw_kgb23:722-790.
        23 => vec![real!("raylo",16), real!("givfac"), layer!(), real!("kao",5,13,16),
            real!("selfrefo",10,16), real!("forrefo",3,16), real!("sfluxrefo",16)],
        // sw_kgb24:793-880.
        24 => vec![real!("raylao",16,9), real!("raylbo",16), real!("strrat"), layer!(),
            real!("abso3ao",16), real!("abso3bo",16), real!("kao",9,5,13,16), real!("kbo",5,47,16),
            real!("selfrefo",10,16), real!("forrefo",3,16), real!("sfluxrefo",16,9)],
        // sw_kgb25:883-939; sw_kgb26:942-974.
        25 => vec![real!("raylo",16), layer!(), real!("abso3ao",16), real!("abso3bo",16),
            real!("kao",5,13,16), real!("sfluxrefo",16)],
        26 => vec![real!("raylo",16), real!("sfluxrefo",16)],
        // sw_kgb27:977-1047; sw_kgb28:1050-1115.
        27 => vec![real!("raylo",16), real!("scalekur"), layer!(), real!("kao",5,13,16),
            real!("kbo",5,47,16), real!("sfluxrefo",16)],
        28 => vec![real!("rayl"), real!("strrat"), layer!(), real!("kao",9,5,13,16),
            real!("kbo",5,5,47,16), real!("sfluxrefo",16,5)],
        // sw_kgb29:1118-1206.
        29 => vec![real!("rayl"), layer!(), real!("absh2oo",16), real!("absco2o",16),
            real!("kao",5,13,16), real!("kbo",5,47,16), real!("selfrefo",10,16),
            real!("forrefo",4,16), real!("sfluxrefo",16)],
        _ => unreachable!("only bands16..29 have MPAS SW records"),
    }
}

fn malformed(message: String) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, message) }
fn word<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut bytes=[0;4]; reader.read_exact(&mut bytes)?; Ok(u32::from_be_bytes(bytes))
}

/// Reject wrong precision, endian, framing and truncation before publication.
/// MPAS mpas_atmphys_rrtmg_swinit.F:205-206,287-288,370-371,452-453,
/// 536-537,618-619,700-701,772-773,857-859,922-923,961-962,1030-1031,
/// 1098-1099,1186-1187.
pub fn decode<R: Read>(mut reader: R) -> io::Result<Vec<Coefficient>> {
    let mut result=Vec::new();
    for band in 16..=29 {
        let specs=record_specs(band);
        let count: usize=specs.iter().map(|s|s.shape.iter().product::<usize>()).sum();
        let bytes=(count*4) as u32;
        if word(&mut reader)?!=bytes { return Err(malformed(format!("band{band}: expected {bytes} byte FP32 big-endian record"))); }
        for spec in specs {
            let count=spec.shape.iter().product::<usize>();
            let words=(0..count).map(|_|word(&mut reader)).collect::<io::Result<Vec<_>>>()?;
            result.push(Coefficient { name: format!("kg{band}/{}",spec.name), shape:spec.shape.to_vec(), integer:spec.integer, words });
        }
        if word(&mut reader)?!=bytes { return Err(malformed(format!("band{band}: trailing Fortran record length differs"))); }
    }
    let mut trailing=[0];
    if reader.read(&mut trailing)?!=0 { return Err(malformed("unexpected data after band29".into())); }
    Ok(result)
}

/// Publish typed little-endian raw carriers; Python only uploads these words.
/// No output directory is created until the complete input validates.
pub fn decode_file(input: &Path, output: &Path) -> io::Result<()> {
    let fields=decode(fs::File::open(input)?)?;
    fs::create_dir(output)?;
    let mut descriptors=Vec::new();
    for field in fields {
        let file=field.name.replace('/',"_")+".bin";
        let data:Vec<u8>=field.words.iter().flat_map(|w|w.to_le_bytes()).collect();
        fs::write(output.join(&file),data)?;
        let dtype=if field.integer {"<i4"} else {"<f4"};
        descriptors.push(format!("{{\"name\":\"{}\",\"file\":\"{file}\",\"shape\":{:?},\"dtype\":\"{dtype}\",\"words\":{}}}",field.name,field.shape,field.words.len()));
    }
    fs::write(output.join("manifest.json"),format!("{{\"schema\":\"mpas-rrtmg-sw-raw-v842/1\",\"order\":\"F\",\"fields\":[{}]}}\n",descriptors.join(",")))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn stream() -> Vec<u8> {
        let mut bytes=Vec::new();
        for band in 16..=29 {
            let count:usize=record_specs(band).iter().map(|s|s.shape.iter().product::<usize>()).sum();
            bytes.extend(((count*4) as u32).to_be_bytes());
            for i in 0..count { bytes.extend((0x80000000|i as u32).to_be_bytes()); }
            bytes.extend(((count*4) as u32).to_be_bytes());
        }
        bytes
    }
    #[test]
    fn preserves_signed_zero_and_nonfinite_words() {
        let mut bytes=stream();
        bytes[4..8].copy_from_slice(&0x80000000u32.to_be_bytes());
        bytes[8..12].copy_from_slice(&0x7fa12345u32.to_be_bytes());
        let fields=decode(&bytes[..]).unwrap();
        assert_eq!(fields[0].words,[0x80000000]);
        assert_eq!(fields[1].words,[0x7fa12345]);
        assert_eq!(fields[3].shape,[9,5,13,16]);
        assert_eq!(fields.last().unwrap().name,"kg29/sfluxrefo");
    }
    #[test]
    fn rejects_bad_framing_precision_and_partial_streams() {
        let bytes=stream();
        assert!(decode(&bytes[..bytes.len()-1]).is_err());
        let mut bad=bytes.clone(); bad[0..4].copy_from_slice(&106776u32.to_be_bytes());
        assert!(decode(&bad[..]).is_err());
        bad=bytes.clone();let n=u32::from_be_bytes(bad[..4].try_into().unwrap()) as usize;
        bad[4+n]^=1;assert!(decode(&bad[..]).is_err());
        bad=bytes;bad.push(0);assert!(decode(&bad[..]).is_err());
    }
}
