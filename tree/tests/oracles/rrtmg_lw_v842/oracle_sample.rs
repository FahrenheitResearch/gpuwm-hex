//! Test-only f1 ORCL0001 transport. Preserve original words; no field arithmetic.
//! MPAS driver boundary: mpas_atmphys_driver_radiation_lw.F:865-885.
use std::{env,fs,io::{self,Write},path::Path};
const FIXED:usize=860;
struct Record{header:[i32;21],ids:Vec<i32>,data:Vec<u8>}
fn bad(s:&str)->io::Error{io::Error::new(io::ErrorKind::InvalidData,s)}
fn read(p:&Path)->io::Result<Record>{
 decode(&fs::read(p)?)
}
fn decode(raw:&[u8])->io::Result<Record>{
 if raw.len()<FIXED || &raw[..8]!=b"ORCL0001"{return Err(bad("oracle magic"));}
 let mut h=[0;21];
 for(i,v)in h.iter_mut().enumerate(){*v=i32::from_le_bytes(raw[8+4*i..12+4*i].try_into().unwrap());}
 if ![1,3,4].contains(&h[0]) || h[1]!=4 || !(0..=5).contains(&h[2]) || h[16]<0 || h[19]!=256 || h[20]!=0 {return Err(bad("unsupported native record"));}
 if !(0..=h[2]).contains(&h[17]) || !(0..=h[2]).contains(&h[18]) || (h[17]>0 && h[17]==h[18]){return Err(bad("invalid axes"));}
 let expected_ids=if h[17]>0{h[2+h[17]as usize]}else{0};
 if h[16]!=expected_ids{return Err(bad("ID count mismatch"));}
 let count=h[3..8].iter().try_fold(1usize,|n,d|if *d<0{None}else{n.checked_mul(*d as usize)}).ok_or_else(||bad("shape overflow"))?;
 let offset=FIXED+4*h[16]as usize;
 if raw.len()!=offset+count*4{return Err(bad("record length"));}
 let ids=raw[FIXED..offset].chunks_exact(4).map(|x|i32::from_le_bytes(x.try_into().unwrap())).collect();
 Ok(Record{header:h,ids,data:raw[offset..].to_vec()})
}
fn tagged(w:&mut impl Write,name:&str,r:&Record,indices:&[usize])->io::Result<()>{
 let mut dims=r.header[3..3+r.header[2]as usize].to_vec();
 let axis=r.header[17];
 let mut payload=Vec::new();
 if axis>0{
  let a=axis as usize-1;
  let stride=dims[..a].iter().fold(1usize,|n,d|n*(*d as usize));
  let old=dims[a]as usize;
  // Each slower-axis slab emits the selected columns in original order.
  let slab=stride*old*4;
  for section in r.data.chunks_exact(slab){for i in indices{payload.extend_from_slice(&section[i*stride*4..(i+1)*stride*4]);}}
  dims[a]=indices.len()as i32;
 }else{payload.extend_from_slice(&r.data);}
 w.write_all(&(name.len()as i32).to_le_bytes())?;w.write_all(name.as_bytes())?;
 w.write_all(&(if r.header[0]==1{0i32}else{1i32}).to_le_bytes())?;
 w.write_all(&(dims.len()as i32).to_le_bytes())?;
 for d in dims{w.write_all(&d.to_le_bytes())?;}
 w.write_all(&payload)
}
fn main()->io::Result<()>{
 let a:Vec<String>=env::args().collect();
 if a.len()!=4{return Err(bad("usage: oracle_sample CAPTURE_PREFIX TAGGED_OUTPUT MAX_COLUMNS"));}
 let prefix=Path::new(&a[1]);let root=prefix.parent().ok_or_else(||bad("prefix parent"))?;
 let base=prefix.file_name().unwrap().to_str().ok_or_else(||bad("prefix utf8"))?;
 let p=read(&root.join(format!("{base}.pre.p3d.bin")))?;
 if p.header[17]!=1 || p.ids.len()!=p.header[3]as usize{return Err(bad("expected native column axis"));}
 let count=a[3].parse::<usize>().map_err(|_|bad("column count"))?.min(p.ids.len());
 if count==0{return Err(bad("empty selection"));}
 let indices:Vec<usize>=(0..count).map(|i|i*(p.ids.len()-1)/count.max(2).saturating_sub(1)).collect();
 let mut out=fs::OpenOptions::new().write(true).create_new(true).open(&a[2])?;
 let mut files:Vec<_>=fs::read_dir(root)?.map(|e|e.map(|x|x.path())).collect::<Result<_,_>>()?;files.sort();
 for path in files{
  let name=path.file_name().unwrap().to_str().unwrap();
  if let Some(field)=name.strip_prefix(&format!("{base}." )).and_then(|s|s.strip_suffix(".bin")){
   if !(field.starts_with("pre.")||field.starts_with("post.")) || field.contains(".state.") || field.contains(".local."){continue;}
   let r=read(&path)?;
   if r.header[17]>0 && r.ids!=p.ids{return Err(bad("column ID mismatch"));}
   tagged(&mut out,&field.replacen('.',"/",1),&r,&indices)?;
  }
 }
 println!("{{\"total_columns\":{},\"sample_columns\":{},\"global_ids\":{:?}}}",p.ids.len(),count,indices.iter().map(|i|p.ids[*i]).collect::<Vec<_>>());
 Ok(())
}

#[cfg(test)]mod tests{
 use super::*;
 #[test]fn malformed_transport_is_rejected(){
  assert!(decode(b"ORCL0001").is_err());
  let mut raw=vec![0;FIXED];raw[..8].copy_from_slice(b"ORCL0001");
  assert!(decode(&raw).is_err());
 }
 #[test]fn sampling_preserves_signed_zero_nan_and_fortran_order(){
  let mut h=[0;21];h[0]=1;h[1]=4;h[2]=2;h[3..8].copy_from_slice(&[3,2,1,1,1]);h[17]=1;
  let words=[0x80000000u32,1,0x7fc12345,2,3,0x12345678];
  let r=Record{header:h,ids:vec![1,2,3],data:words.iter().flat_map(|v|v.to_le_bytes()).collect()};
  let mut out=Vec::new();tagged(&mut out,"x",&r,&[0,2]).unwrap();
  assert_eq!(&out[21..],&[0x80000000u32,0x7fc12345,2,0x12345678].iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>());
 }
}
