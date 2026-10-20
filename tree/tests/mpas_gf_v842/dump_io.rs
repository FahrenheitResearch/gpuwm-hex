//! Test-only GF transport: f1 ORCL0001 parser reused from p5's Rust adapter.
//! MPAS module_cu_gf.mpas.F:54-453; convection.F:484-513.
//! Rust owns decoding, packing, held-interface preservation and encoding.
use std::{env, fs, io, path::{Path, PathBuf}};
const FIXED: usize = 8 + 21 * 4 + 3 * 256;
const LEVELS: [&str;15] = ["u","v","w","t","q","p","pi","rho","dz8w","p8w","rthften","rqvften","rthraten","rthblten","rqvblten"];
const SURFACE: [&str;8] = ["ht","hfx","qfx","xland","gsw","dxcell","areacell","dt"];
const OUT: [&str;6] = ["rthcuten","rqvcuten","rqccuten","rqicuten","gdc","gdc2"];
const SOUT: [&str;6] = ["htop","hbot","xmb_total","xmb_shallow","raincv","pratec"];
const IOUT: [&str;3] = ["k22_shallow","kbcon_shallow","ktop_shallow"];
#[derive(Clone)]
struct Record {
    head: Vec<u8>,
    h: [i32; 21],
    data: Vec<u8>,
}
fn bad(s: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, s)
}
fn read(p: &Path) -> io::Result<Record> {
    let bytes = fs::read(p)?;
    if bytes.len() < FIXED || &bytes[..8] != b"ORCL0001" {
        return Err(bad("invalid ORCL0001 header"));
    }
    let mut h = [0; 21];
    for (i, v) in h.iter_mut().enumerate() {
        *v = i32::from_le_bytes(bytes[8 + 4 * i..12 + 4 * i].try_into().unwrap());
    }
    if !(0..=5).contains(&h[2])
        || h[1] <= 0
        || h[3..8].iter().any(|v| *v < 0)
        || h[16] < 0
        || h[19] != 256
        || h[20] != 0
    {
        return Err(bad("invalid oracle dimensions"));
    }
    let count = h[3..8]
        .iter()
        .try_fold(1usize, |a, v| a.checked_mul(*v as usize))
        .ok_or_else(|| bad("size overflow"))?;
    let offset = FIXED
        .checked_add((h[16] as usize) * 4)
        .ok_or_else(|| bad("offset overflow"))?;
    let expected = count
        .checked_mul(h[1] as usize)
        .and_then(|n| n.checked_add(offset))
        .ok_or_else(|| bad("size overflow"))?;
    if bytes.len() != expected {
        return Err(bad("truncated/trailing oracle payload"));
    }
    Ok(Record {
        head: bytes[..offset].to_vec(),
        h,
        data: bytes[offset..].to_vec(),
    })
}
fn path(prefix: &Path, phase: &str, name: &str) -> PathBuf {
    PathBuf::from(format!("{}.{}.{}.bin", prefix.display(), phase, name))
}
fn pre(prefix: &Path, name: &str) -> io::Result<Record> {
    read(&path(prefix, "pre", name))
}
fn integer(prefix: &Path, name: &str) -> io::Result<i32> {
    let r = pre(prefix, name)?;
    if r.h[2] != 0 || r.h[1] != 4 {
        return Err(bad("expected scalar word"));
    }
    Ok(i32::from_le_bytes(r.data[..4].try_into().unwrap()))
}
fn scalar(prefix: &Path, name: &str) -> io::Result<Vec<u8>> {
    let r = pre(prefix, name)?;
    if r.h[2] != 0 || r.h[1] != 4 {
        return Err(bad("expected scalar word"));
    }
    Ok(r.data)
}
fn write_record(out: &Path, name: &str, mut r: Record) -> io::Result<()> {
    for (i, v) in r.h.iter().enumerate() {
        r.head[8 + 4 * i..12 + 4 * i].copy_from_slice(&v.to_le_bytes());
    }
    for (start, value) in [(92 + 256, "post"), (92 + 512, name)] {
        r.head[start..start + 256].fill(b' ');
        r.head[start..start + value.len()].copy_from_slice(value.as_bytes());
    }
    let p = path(out, "post", name);
    if p.exists() {
        return Err(bad("output record already exists"));
    }
    r.head.extend(r.data);
    fs::write(p, r.head)
}

struct Tile { first:usize, nc:usize, nz:usize }
fn tile(prefix:&Path)->io::Result<Tile>{
 let t=pre(prefix,"t")?;
 let its=integer(prefix,"its")?; let ite=integer(prefix,"ite")?;
 let nz=integer(prefix,"kte")?;
 if integer(prefix,"kts")?!=1 || integer(prefix,"jts")?!=1 || integer(prefix,"jte")?!=1
 || t.h[8]!=1 || t.h[9]!=1 || t.h[10]!=1 || its<1 || ite<its || ite>t.h[3]
 || nz<1 || nz>t.h[4] || ite>=integer(prefix,"ide")? {
  return Err(bad("GF capture does not match the native owned one-row driver ABI"));
 }
 Ok(Tile {first:(its-1) as usize,nc:(ite-its+1) as usize,nz:nz as usize})
}
fn packed(prefix:&Path,names:&[&str],t:&Tile,vertical:bool)->io::Result<Vec<u8>>{
 let nl=if vertical {t.nz} else {1};
 let mut result=vec![0;names.len()*nl*t.nc*4];
 for (a,name) in names.iter().enumerate(){
  let r=pre(prefix,name)?;
  if r.h[1]!=4 || (r.h[2]!=0 && (r.h[3] as usize)<t.first+t.nc) ||
     (vertical && (r.h[4] as usize)<nl){return Err(bad("invalid GF input extent/kind"));}
  for c in 0..t.nc {for k in 0..nl {
   let i=if r.h[2]==0 {0} else {(k*r.h[3] as usize+t.first+c)*4};
   let o=((c*names.len()+a)*nl+k)*4;
   result[o..o+4].copy_from_slice(&r.data[i..i+4]);
  }}
 }
 Ok(result)
}
fn pack(prefix:&Path,dir:&Path)->io::Result<()> {
 fs::create_dir_all(dir)?; let t=tile(prefix)?;
 for (name,names,vertical) in [("levels",&LEVELS[..],true),("surface",&SURFACE[..],false),("kpbl",&["kpbl"][..],false)]{
  fs::write(dir.join(format!("{name}.bin")),packed(prefix,names,&t,vertical)?)?;
 }
 for (name,params) in [("control",&["itimestep","ichoice_deep","ichoice_shallow","ishallow_g3"][..]),("constants",&["xlv","cp","g","r_v"][..])]{
  let mut result=vec![]; for p in params {result.extend(scalar(prefix,p)?);}
  fs::write(dir.join(format!("{name}.bin")),result)?;
 }
 println!("{{\"nc\":{},\"nz\":{},\"first\":{}}}",t.nc,t.nz,t.first);
 Ok(())
}
fn unpack(prefix:&Path,dir:&Path,out:&Path)->io::Result<()> {
 let t=tile(prefix)?;
 for (file,names,vertical) in [("levels",&OUT[..],true),("surface",&SOUT[..],false),("integer",&IOUT[..],false)]{
  let b=fs::read(dir.join(format!("{file}-out.bin")))?;
  let nl=if vertical {t.nz} else {1};
  if b.len()!=names.len()*nl*t.nc*4 {return Err(bad("incorrect GF output extent"));}
  for (a,name) in names.iter().enumerate(){
   let mut r=pre(prefix,name)?;
   if r.h[1]!=4 || (r.h[3] as usize)<t.first+t.nc || (vertical && (r.h[4] as usize)<nl){return Err(bad("invalid GF output kind/extent"));}
   for c in 0..t.nc {for k in 0..nl{
    let i=(k*r.h[3] as usize+t.first+c)*4;
    let o=((c*names.len()+a)*nl+k)*4;
    r.data[i..i+4].copy_from_slice(&b[o..o+4]);
   }}
   write_record(out,name,r)?;
  }
 }
 // GF has no mutable module/local SAVE state. Framework state is held by
 // this isolated scheme call, so preserve its entry words and schema.
 let parent=prefix.parent().ok_or_else(||bad("prefix has no parent"))?;
 let stem=format!("{}.pre.",prefix.file_name().unwrap().to_string_lossy());
 for entry in fs::read_dir(parent)?{
  let entry=entry?; let filename=entry.file_name();let filename=filename.to_string_lossy();
  let Some(name)=filename.strip_prefix(&stem).and_then(|v|v.strip_suffix(".bin")) else {continue;};
  if name.starts_with("state.") || name.starts_with("local.") || name.strip_suffix(".present").is_some_and(|n| OUT.contains(&n) || SOUT.contains(&n) || IOUT.contains(&n)) {
   write_record(out,name,pre(prefix,name)?)?;
  }
 }
 Ok(())
}
fn main()->io::Result<()> {
 let a=env::args().collect::<Vec<_>>();
 if a.len()<4{return Err(bad("usage: dump_io pack PREFIX DIRECTORY | unpack PREFIX DIRECTORY OUTPUT_PREFIX"));}
 match a[1].as_str(){
 "pack"=>pack(Path::new(&a[2]),Path::new(&a[3])),
 "unpack" if a.len()==5=>unpack(Path::new(&a[2]),Path::new(&a[3]),Path::new(&a[4])),
 _=>Err(bad("invalid operation")),
 }
}
