//! Lossless ORCL0001 transport for the RRFS MYNN-EDMF replay.
//! Reuses p14 generic Rust oracle transport; no numerical kernel is reused.
//! Science payloads are opaque bytes. Only scalar arguments are decoded here.
//! Format authority: f1-native-tap native_physics_reference/oracle/FORMAT.md.
use std::{
    env, fs,
    io::{self, Read, Write},
    path::Path,
};

const FIXED: usize = 860;
struct Record {
    raw: Vec<u8>,
    h: [i32; 21],
    offset: usize,
}
fn bad(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
impl Record {
    fn parse(raw: Vec<u8>) -> io::Result<Self> {
        if raw.len() < FIXED || &raw[..8] != b"ORCL0001" {
            return Err(bad("invalid oracle header"));
        }
        let mut h = [0; 21];
        for (i, value) in h.iter_mut().enumerate() {
            *value = i32::from_le_bytes(raw[8 + 4 * i..12 + 4 * i].try_into().unwrap());
        }
        if !(0..=5).contains(&h[2])
            || h[3..8].iter().any(|v| *v < 0)
            || h[16] < 0
            || h[19] != 256
            || h[20] != 0
        {
            return Err(bad("invalid dimensions/version"));
        }
        let width = match h[0] {
            1 | 3 | 4 => 4,
            2 => 8,
            5 if h[1] > 0 => h[1],
            _ => return Err(bad("invalid type")),
        };
        if width != h[1] {
            return Err(bad("type width mismatch"));
        }
        if h[17] < 0 || h[17] > h[2] || h[18] < 0 || h[18] > h[2] || (h[17] > 0 && h[17] == h[18]) {
            return Err(bad("invalid axes"));
        }
        let ids = if h[17] == 0 { 0 } else { h[2 + h[17] as usize] };
        if h[16] != ids {
            return Err(bad("ID count mismatch"));
        }
        if h[0] != 5 && h[3 + h[2] as usize..8].iter().any(|v| *v != 1) {
            return Err(bad("inactive dimensions must equal one"));
        }
        let count = h[3..8]
            .iter()
            .try_fold(1usize, |a, v| a.checked_mul(*v as usize))
            .ok_or_else(|| bad("shape overflow"))?;
        let offset = (h[16] as usize)
            .checked_mul(4)
            .and_then(|n| n.checked_add(FIXED))
            .ok_or_else(|| bad("ID overflow"))?;
        let end = count
            .checked_mul(width as usize)
            .and_then(|n| n.checked_add(offset))
            .ok_or_else(|| bad("payload overflow"))?;
        if end != raw.len() {
            return Err(bad("truncated or trailing payload"));
        }
        Ok(Self { raw, h, offset })
    }
    fn read(path: &str) -> io::Result<Self> {
        Self::parse(fs::read(path)?)
    }
    fn payload(&self) -> &[u8] {
        &self.raw[self.offset..]
    }
    fn metadata(&self) -> io::Result<String> {
        let scalar = if self.h[2] == 0 && self.h[0] != 5 {
            let b = self.payload();
            let s = match self.h[0] {
                1 => f32::from_le_bytes(b.try_into().unwrap()).to_string(),
                2 => f64::from_le_bytes(b.try_into().unwrap()).to_string(),
                3 => i32::from_le_bytes(b.try_into().unwrap()).to_string(),
                4 => (i32::from_le_bytes(b.try_into().unwrap()) != 0).to_string(),
                _ => return Err(bad("invalid scalar")),
            };
            format!("\"{s}\"")
        } else {
            "null".into()
        };
        Ok(format!(
            "{{\"code\":{},\"width\":{},\"shape\":{:?},\"lower\":{:?},\"scalar\":{scalar}}}",
            self.h[0],
            self.h[1],
            &self.h[3..3 + self.h[2] as usize],
            &self.h[8..8 + self.h[2] as usize]
        ))
    }
    fn post(mut self, mode: &str, replacement: Vec<u8>) -> io::Result<Self> {
        let name = String::from_utf8_lossy(&self.raw[604..860]);
        let name = name.trim_end();
        match mode {
            "stdin" => {
                if replacement.len() != self.payload().len() {
                    return Err(bad("replacement payload size mismatch"));
                }
                self.raw[self.offset..].copy_from_slice(&replacement);
            }
            "copy" => {}
            "success" if name == "errflg" && self.h[0] == 3 && self.h[2] == 0 => {
                self.raw[self.offset..].fill(0)
            }
            "success" if name == "errmsg" && self.h[0] == 5 => self.raw[self.offset..].fill(b' '),
            _ => return Err(bad("success applies only to error status/message")),
        }
        self.raw[348..604].fill(b' ');
        self.raw[348..352].copy_from_slice(b"post");
        Ok(self)
    }
    fn write(&self, path: &str) -> io::Result<()> {
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(Path::new(path))?
            .write_all(&self.raw)
    }
}
fn stdin_bytes() -> io::Result<Vec<u8>> {
    let mut b = Vec::new();
    io::stdin().read_to_end(&mut b)?;
    Ok(b)
}
fn list(s: &str) -> io::Result<Vec<i32>> {
    if s == "-" {
        return Ok(vec![]);
    }
    s.split(',')
        .map(|v| v.parse().map_err(|_| bad("invalid dimension")))
        .collect()
}
fn new_record(
    code: i32,
    dims: &[i32],
    lower: &[i32],
    name: &str,
    data: Vec<u8>,
) -> io::Result<Record> {
    if dims.len() > 5 || dims.len() != lower.len() || name.len() > 256 {
        return Err(bad("invalid fixture schema"));
    }
    let mut h = [0i32; 21];
    h[0] = code;
    h[1] = if code == 2 {
        8
    } else if code == 5 {
        1
    } else {
        4
    };
    h[2] = dims.len() as i32;
    h[3..8].fill(1);
    h[8..13].fill(1);
    h[3..3 + dims.len()].copy_from_slice(dims);
    h[8..8 + lower.len()].copy_from_slice(lower);
    if code == 5 && dims.is_empty() {
        h[3] = data
            .len()
            .try_into()
            .map_err(|_| bad("character length overflow"))?;
    }
    h[19] = 256;
    let mut raw = b"ORCL0001".to_vec();
    for v in h {
        raw.extend(v.to_le_bytes());
    }
    raw.resize(FIXED, b' ');
    raw[92..101].copy_from_slice(b"synthetic");
    raw[348..351].copy_from_slice(b"pre");
    raw[604..604 + name.len()].copy_from_slice(name.as_bytes());
    raw.extend(data);
    Record::parse(raw)
}
fn main() -> io::Result<()> {
    let a: Vec<String> = env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("metadata") if a.len() == 3 => println!("{}", Record::read(&a[2])?.metadata()?),
        Some("payload") if a.len() == 3 => {
            io::stdout().write_all(Record::read(&a[2])?.payload())?
        }
        Some("blank") if a.len() == 3 => {
            io::stdout().write_all(&vec![0; Record::read(&a[2])?.payload().len()])?
        }
        Some("write") if a.len() == 5 => {
            let b = if a[4] == "stdin" {
                stdin_bytes()?
            } else {
                vec![]
            };
            Record::read(&a[2])?.post(&a[4], b)?.write(&a[3])?;
        }
        Some("new") if a.len() == 7 => new_record(
            a[3].parse().map_err(|_| bad("invalid code"))?,
            &list(&a[4])?,
            &list(&a[5])?,
            &a[6],
            stdin_bytes()?,
        )?
        .write(&a[2])?,
        _ => {
            return Err(bad(
                "usage: metadata INPUT | payload INPUT | blank TEMPLATE | write TEMPLATE OUTPUT stdin|copy|success | new OUTPUT CODE DIMS LOWER NAME (fixture bytes on stdin)",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn opaque_words_survive() {
        let b = [0x80000000u32, 0x7fc01234, 0x00000001, 0x3f800000]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let r = new_record(1, &[2, 2], &[-1, 4], "wind", b.clone())
            .unwrap()
            .post("stdin", b.clone())
            .unwrap();
        assert_eq!(r.payload(), b);
        assert_eq!(&r.raw[348..352], b"post");
        assert_eq!(&r.h[8..10], &[-1, 4]);
    }
    #[test]
    fn rejects_malformed_and_wrong_sizes() {
        let r = new_record(1, &[1], &[1], "x", vec![0; 4]).unwrap();
        assert!(Record::parse(r.raw[..FIXED + 3].to_vec()).is_err());
        let mut raw = r.raw.clone();
        raw[12..16].copy_from_slice(&8i32.to_le_bytes());
        assert!(Record::parse(raw).is_err());
        assert!(r.post("stdin", vec![0; 8]).is_err());
    }
    #[test]
    fn no_science_success_override() {
        assert!(
            new_record(3, &[], &[], "itimestep", vec![0; 4])
                .unwrap()
                .post("success", vec![])
                .is_err()
        );
        assert!(
            new_record(1, &[], &[], "x", vec![0; 4])
                .unwrap()
                .post("success", vec![])
                .is_err()
        );
        let r = new_record(3, &[], &[], "errflg", (-1i32).to_le_bytes().to_vec())
            .unwrap()
            .post("success", vec![])
            .unwrap();
        assert_eq!(r.payload(), [0; 4]);
    }
    #[test]
    fn invalid_axis_ids_and_overflow() {
        let r = new_record(1, &[1], &[1], "x", vec![0; 4]).unwrap();
        let mut raw = r.raw.clone();
        raw[76..80].copy_from_slice(&1i32.to_le_bytes());
        assert!(Record::parse(raw).is_err());
        assert!(new_record(1, &[i32::MAX; 5], &[1; 5], "x", vec![]).is_err());
    }
    #[test]
    fn global_ids_and_lower_bounds_survive() {
        let r = new_record(1, &[2, 1], &[-1, 3], "wind", vec![0; 8]).unwrap();
        let mut raw = r.raw;
        raw[72..76].copy_from_slice(&2i32.to_le_bytes());
        raw[76..80].copy_from_slice(&1i32.to_le_bytes());
        raw[80..84].copy_from_slice(&2i32.to_le_bytes());
        raw.splice(
            FIXED..FIXED,
            [0i32, 42].into_iter().flat_map(i32::to_le_bytes),
        );
        let r = Record::parse(raw)
            .unwrap()
            .post("stdin", vec![1; 8])
            .unwrap();
        assert_eq!(&r.raw[FIXED..FIXED + 8], &[0, 0, 0, 0, 42, 0, 0, 0]);
        assert_eq!(&r.h[8..10], &[-1, 3]);
    }
    #[test]
    fn scalar_character_uses_length_extent() {
        let r = new_record(5, &[], &[], "errmsg", vec![b'x'; 512])
            .unwrap()
            .post("success", vec![])
            .unwrap();
        assert_eq!(r.h[2], 0);
        assert_eq!(r.h[3], 512);
        assert!(r.payload().iter().all(|b| *b == b' '));
    }
}
