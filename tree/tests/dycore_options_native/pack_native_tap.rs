//! Native MPAS ORCL0001 data transport for the complete non-LES replay.
//! No weather arithmetic: retain all Fortran-order words and bounds.
use std::{env,fs,io::{self,Write},path::Path};
fn main()->Result<(),Box<dyn std::error::Error>> {
 let a:Vec<_>=env::args().collect(); if a.len()!=4 {return Err("pack_native_tap RECORD-LIST INPUT-ROOT OUTPUT".into());}
 let mut out=io::BufWriter::new(fs::File::create(&a[3])?);
 for name in fs::read_to_string(&a[1])?.lines() {
  if name.is_empty() || name.contains('/') || name.contains("..") {return Err("unsafe record name".into());}
  let b=fs::read(Path::new(&a[2]).join(name))?;
  if b.len()<860 || &b[..8]!=b"ORCL0001" {return Err("invalid ORCL0001 header".into());}
  let h:Vec<i32>=b[8..92].chunks_exact(4).map(|v|i32::from_le_bytes(v.try_into().unwrap())).collect();
  if !(0..=5).contains(&h[2]) || ![1,3].contains(&h[0]) || h[1]!=4 || h[16]<0 || h[19]!=256 || h[20]!=0 {return Err("unexpected native type/schema".into());}
  let n=h[3..8].iter().try_fold(1usize,|n,&s| if s<0 {None} else {n.checked_mul(s as usize)}).ok_or("invalid native extent")?;
  let off=860+4*h[16] as usize;
  if off+4*n!=b.len() {return Err("truncated/trailing native record".into());}
  out.write_all(&b[off..])?;
 }
 out.flush()?;Ok(())
}
