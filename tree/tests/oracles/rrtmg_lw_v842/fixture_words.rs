// MPAS: module_ra_rrtmg_lw.F:2099-2697,11574-12535; whole tagged oracle word census.
use std::{collections::BTreeMap,env,fs,io};
#[derive(Debug)] struct Record { dtype:i32, dims:Vec<i32>, words:Vec<u32> }
fn err(s:&str)->io::Error { io::Error::new(io::ErrorKind::InvalidData,s) }
fn i32_at(bytes:&[u8],at:&mut usize)->io::Result<i32> {
 let v=bytes.get(*at..*at+4).ok_or_else(||err("truncated metadata"))?;
 *at+=4;Ok(i32::from_le_bytes(v.try_into().unwrap()))
}
fn read(path:&str)->io::Result<BTreeMap<String,Record>> {
 let bytes=fs::read(path)?;let mut at=0;let mut records=BTreeMap::new();
 while at<bytes.len() {
  let n=i32_at(&bytes,&mut at)?;if !(1..4096).contains(&n){return Err(err("invalid name length"));}
  let name=String::from_utf8(bytes.get(at..at+n as usize).ok_or_else(||err("truncated name"))?.to_vec()).map_err(|_|err("invalid name"))?;at+=n as usize;
  if !name.bytes().all(|x|x.is_ascii()&&x>=32&&x!=b'"'&&x!=b'\\'){return Err(err("unsafe JSON record name"));}
  let dtype=i32_at(&bytes,&mut at)?;if dtype!=0&&dtype!=1{return Err(err("invalid dtype"));}
  let rank=i32_at(&bytes,&mut at)?;if !(0..=4).contains(&rank){return Err(err("invalid rank"));}
  let mut dims=Vec::new();let mut count=1usize;
  for _ in 0..rank {let d=i32_at(&bytes,&mut at)?;if d<0{return Err(err("negative dimension"));}count=count.checked_mul(d as usize).ok_or_else(||err("overflow"))?;dims.push(d);}
  let size=count.checked_mul(4).ok_or_else(||err("overflow"))?;let data=bytes.get(at..at+size).ok_or_else(||err("truncated data"))?;
  let words=data.chunks_exact(4).map(|v|u32::from_le_bytes(v.try_into().unwrap())).collect();at+=size;
  if records.insert(name,Record{dtype,dims,words}).is_some(){return Err(err("duplicate name"));}
 }
 Ok(records)
}
fn export(path:&str, directory:&str)->io::Result<()> {
 let records=read(path)?;let out=std::path::Path::new(directory);fs::create_dir_all(out)?;let mut rows=Vec::new();
 for (i,(name,r)) in records.iter().enumerate() {
  let filename=format!("{i:04}.raw");let mut bytes=Vec::with_capacity(r.words.len()*4);
  for cindex in 0..r.words.len() {
   let mut rem=cindex;let mut coords=vec![0usize;r.dims.len()];
   for j in (0..r.dims.len()).rev(){coords[j]=rem%(r.dims[j] as usize);rem/=r.dims[j] as usize;}
   let mut findex=0usize;let mut step=1usize;
   for (j,dim) in r.dims.iter().enumerate(){findex+=coords[j]*step;step*=*dim as usize;}
   bytes.extend_from_slice(&r.words[findex].to_le_bytes());
  }
  fs::write(out.join(&filename),bytes)?;
  rows.push(format!("{{\"name\":\"{name}\",\"dtype\":{},\"shape\":{:?},\"words\":{},\"file\":\"{filename}\"}}",r.dtype,r.dims,r.words.len()));
 }
 fs::write(out.join("manifest.json"),format!("{{\"schema\":\"p22-rust-tagged-fixture-export-v54\",\"order\":\"C\",\"arrays\":[{}]}}\n",rows.join(",")))?;Ok(())
}
fn main()->io::Result<()> {
 let args:Vec<_>=env::args().collect();if args.len()==4&&args[1]=="export"{return export(&args[2],&args[3]);}if args.len()!=3{return Err(err("usage: fixture_words NATIVE_A NATIVE_B"));}
 let a=read(&args[1])?;let b=read(&args[2])?;if a.keys().collect::<Vec<_>>()!=b.keys().collect::<Vec<_>>(){return Err(err("different record sets"));}
 let mut total=0usize;let mut changed=0usize;let mut rows=Vec::new();
 for (name,x) in &a {
  let y=&b[name];if x.dtype!=y.dtype||x.dims!=y.dims||x.words.len()!=y.words.len(){return Err(err("different record shape"));}
  let mut indices=Vec::new();let mut pairs=Vec::new();
  for (i,(u,v)) in x.words.iter().zip(&y.words).enumerate(){if u!=v {indices.push(i);pairs.push(format!("[\"{u:08x}\",\"{v:08x}\"]"));}}
  total+=x.words.len();changed+=indices.len();
  rows.push(format!("{{\"name\":\"{name}\",\"dtype\":{},\"shape\":{:?},\"words\":{},\"differing_words\":{},\"different_indices_fortran_zero_based\":{:?},\"word_pairs_hex\":[{}]}}",x.dtype,x.dims,x.words.len(),indices.len(),indices,pairs.join(",")));
 }
 println!("{{\"schema\":\"p22-tagged-fixture-word-census-v51\",\"compared_words\":{total},\"differing_words\":{changed},\"process_count_per_fixture\":1,\"arrays\":[{}]}}",rows.join(","));Ok(())
}
