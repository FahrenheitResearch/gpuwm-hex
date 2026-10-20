//! Typed MPAS oracle stream decoding, exact word comparison and column extraction.
//! No arithmetic tolerance is introduced. Python delegates all field I/O here.
use std::{env, fs, io, path::Path};
use std::io::Write;
const FIXED: usize = 8 + 21 * 4 + 3 * 256;
struct Record { header: [i32;21], data: Vec<u8> }
fn bad(message: &str) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, message) }
fn read(path: &Path) -> io::Result<Record> {
    let raw = fs::read(path)?;
    if raw.len() < FIXED || &raw[..8] != b"ORCL0001" { return Err(bad("invalid oracle header")); }
    let mut h=[0i32;21];
    for (i,v) in h.iter_mut().enumerate() { *v=i32::from_le_bytes(raw[8+4*i..12+4*i].try_into().unwrap()); }
    if !(0..=5).contains(&h[2]) || h[3..8].iter().any(|v| *v<0) || h[16]<0 || h[19]!=256 || h[20]!=0 {
        return Err(bad("malformed dimensions/version"));
    }
    let width=match h[0] {1|3|4=>4,2|6=>8,5 if h[1]>0=>h[1],_=>return Err(bad("invalid type code/width"))};
    if h[1]!=width { return Err(bad("type width mismatch")); }
    if h[17]<0 || h[17]>h[2] || h[18]<0 || h[18]>h[2] || (h[17]>0 && h[17]==h[18]) {return Err(bad("invalid axes"));}
    let expected_ids=if h[17]>0 {h[2+h[17] as usize]} else {0};
    if h[16]!=expected_ids {return Err(bad("ID count mismatch"));}
    if h[0]!=5 && h[3+h[2] as usize..8].iter().any(|v|*v!=1) {return Err(bad("inactive dimensions must equal one"));}
    let elements=h[3..8].iter().try_fold(1usize,|a,v|a.checked_mul(*v as usize)).ok_or_else(||bad("shape overflow"))?;
    let offset=FIXED.checked_add((h[16] as usize).checked_mul(4).ok_or_else(||bad("ID overflow"))?).ok_or_else(||bad("offset overflow"))?;
    let size=elements.checked_mul(width as usize).and_then(|v|v.checked_add(offset)).ok_or_else(||bad("payload overflow"))?;
    if raw.len()!=size {return Err(bad("truncated or trailing payload"));}
    Ok(Record{header:h,data:raw[offset..].to_vec()})
}

use std::collections::BTreeMap;
fn main()->io::Result<()> {
 let a:Vec<String>=env::args().collect();
 if a.len()!=4{return Err(bad("usage: real_columns CAPTURE_PREFIX JULIAN OUTPUT_TEXT"));}
 let julian:f32=a[2].parse().map_err(|_|bad("invalid Julian day"))?;
 let fields=["p3d","t3d","dz8w","pi3d","qv3d","qc3d","qi3d","qs3d","cldfra3d","re_cloud","re_ice","re_snow"];
 let surfaces=["xlat","xlong","albedo","tsk","xland","xice","snow"];
 let ints=["icloud","cldovrlp","idcor","o3input","has_reqc","has_reqi","has_reqs","aer_opt","julday"];
 let controls=["xtime","gmt","radt","degrad","declin","solcon"];
 let mut records=BTreeMap::new();
 for n in fields.iter().chain(surfaces.iter()).chain(ints.iter()).chain(controls.iter()).chain(["p8w","t8w","pin","o3clim","kts","kte","kms","its","ite","ims"].iter()) {
  records.insert(*n,read(Path::new(&format!("{}.pre.{}.bin",a[1],n)))?);
 }
 let int=|n:&str| -> i32 {i32::from_le_bytes(records[n].data[..4].try_into().unwrap())};
 let value=|n:&str,col:usize,level:usize| -> f32 {
  let r=&records[n];assert_eq!(r.header[0],1);
  let index=if r.header[2]>1 {col+r.header[3] as usize*level}else{col};
  f32::from_le_bytes(r.data[index*4..index*4+4].try_into().unwrap())
 };
 let first=(int("its")-int("ims")) as usize;let last=(int("ite")-int("ims")+1) as usize;
 let kfirst=(int("kts")-int("kms")) as usize;let nz=(int("kte")-int("kts")+1) as usize;
 let npin=records["pin"].data.len()/4;
 if int("aer_opt")!=0{return Err(bad("real input includes aerosol optics: use complete aerosol capture adapter"));}
 let mut out=io::BufWriter::new(fs::File::create(&a[3])?);
 writeln!(out,"{} {} {}",last-first,nz,npin)?;
 for col in first..last {
  for n in ints{write!(out,"{} ",int(n))?;}writeln!(out)?;
  for n in controls{write!(out,"{:.9e} ",value(n,0,0))?;}writeln!(out,"{julian:.9e}")?;
  for n in surfaces{write!(out,"{:.9e} ",value(n,col,0))?;}writeln!(out)?;
  for k in 0..npin{write!(out,"{:.9e} ",value("pin",k,0))?;}writeln!(out)?;
  for k in 0..npin{write!(out,"{:.9e} ",value("o3clim",col,k))?;}writeln!(out)?;
  for k in kfirst..kfirst+nz{for n in fields{write!(out,"{:.9e} ",value(n,col,k))?;}writeln!(out)?;}
  for k in kfirst..kfirst+nz+1{writeln!(out,"{:.9e} {:.9e}",value("p8w",col,k),value("t8w",col,k))?;}
 }
 println!("{{\"columns\":{},\"levels\":{},\"ozone_levels\":{npin},\"input_ref\":\"MPAS-v8.4.2 real capture\",\"native_ref\":\"RRFS-1d6e370e\"}}",last-first,nz);
 Ok(())
}
