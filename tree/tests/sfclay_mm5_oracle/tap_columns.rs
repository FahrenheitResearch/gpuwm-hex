//! Decode f1 ORCL0001 arguments into the MM5 lowest-level packed test carrier.
//! Reader validation adapted from f1 native-tap oracle_words.rs (same format).
//! MPAS driver_sfclayer.F:892-1010; wrappers select KTS and active ITS:ITE/JTS.
use std::{env,fs,io::{self,Read,Write},path::Path};
const FIXED: usize = 8 + 21 * 4 + 3 * 256;
#[derive(Clone)]
struct Record { header: [i32;21], strings: Vec<u8>, ids: Vec<i32>, data: Vec<u8> }
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
    let ids=raw[FIXED..offset].chunks_exact(4).map(|x|i32::from_le_bytes(x.try_into().unwrap())).collect();
    Ok(Record{header:h, strings:raw[92..FIXED].to_vec(), ids,data:raw[offset..].to_vec()})
}

fn record(prefix:&Path,phase:&str,name:&str)->io::Result<Record> {read(Path::new(&format!("{}.{phase}.{name}.bin",prefix.display())))}
fn integer(prefix:&Path,name:&str)->i32 {let r=record(prefix,"pre",name).unwrap();assert_eq!(r.header[0],3);i32::from_le_bytes(r.data[..4].try_into().unwrap())}
fn values(r:&Record,its:i32,ite:i32,jts:i32,kts:i32)->Vec<u8> {
 assert_eq!(r.header[0],1);assert_eq!(r.header[1],4);let mut out=Vec::new();
 for i in its..=ite {
  let mut index=(i-r.header[8]) as usize;
  if r.header[2]==0 {index=0;}
  else if r.header[2]==2 {index+=r.header[3] as usize*(jts-r.header[9]) as usize;}
  else if r.header[2]==3 {index+=r.header[3] as usize*((kts-r.header[9]) as usize+r.header[4] as usize*(jts-r.header[10]) as usize);}
  else {assert_eq!(r.header[2],1);}
  out.extend_from_slice(&r.data[index*4..index*4+4]);
 }
 out
}
fn write_post(output:&Path,name:&str,mut r:Record)->io::Result<()> {
 r.strings[256..512].fill(b' ');r.strings[256..260].copy_from_slice(b"post");
 let mut f=fs::OpenOptions::new().write(true).create_new(true).open(format!("{}.post.{name}.bin",output.display()))?;
 f.write_all(b"ORCL0001")?;for v in r.header {f.write_all(&v.to_le_bytes())?;}
 f.write_all(&r.strings)?;for id in r.ids {f.write_all(&id.to_le_bytes())?;}f.write_all(&r.data)
}
fn main()->io::Result<()> {
 let args:Vec<String>=env::args().collect();let prefix=Path::new(&args[1]);let revised=args[2]=="1";
 let (its,ite,jts,kts)=(integer(prefix,"its"),integer(prefix,"ite"),integer(prefix,"jts"),integer(prefix,"kts"));
 assert_eq!(integer(prefix,"jte"),jts,"single MPAS j row required");let n=(ite-its+1) as usize;
 let fields=["znt","ust","ustm","mol","hfx","qfx","qsfc","zol","regime","psim","psih","fm","fh","lh","u10","v10","th2","t2","q2","chs","chs2","cqs2","flhc","flqc","qgh","rmol","wspd","br","gz1oz0","cpm","ck","cka","cd","cda","u3d","v3d","t3d","qv3d","p3d","dz8w","psfc","tsk","pblh","mavail","xland","lakemask","dx","water_depth"];
 let mut pre=Vec::new();let mut post=Vec::new();
 for (row,name) in fields.iter().enumerate() {
  let before=if !revised && matches!(*name,"lakemask"|"water_depth") {vec![0;4*n]} else {values(&record(prefix,"pre",name)?,its,ite,jts,kts)};
  // Only comparison mode reads POST. Encoding and --inputs require PRE alone.
  let after=if row<34 && args.len()==3 {values(&record(prefix,"post",name)?,its,ite,jts,kts)} else {before.clone()};
  pre.extend(before);post.extend(after);
 }
 if args.len()==5 {
  let output=Path::new(&args[3]);let mut candidate=Vec::new();if args[4]=="-" {io::stdin().read_to_end(&mut candidate)?;} else {candidate=fs::read(&args[4])?;}
  assert_eq!(candidate.len(),4*(48*n+6006));
  for (row,name) in fields[..34].iter().enumerate() {
   let mut r=record(prefix,"pre",name)?;assert_eq!(r.header[2],2);
   for (col,i) in (its..=ite).enumerate() {
    let index=(i-r.header[8]) as usize+r.header[3] as usize*(jts-r.header[9]) as usize;
    r.data[index*4..index*4+4].copy_from_slice(&candidate[(row*n+col)*4..(row*n+col+1)*4]);
   }
   write_post(output,name,r)?;
   if matches!(*name,"ustm"|"ck"|"cka"|"cd"|"cda") {write_post(output,&format!("{name}.present"),record(prefix,"pre",&format!("{name}.present"))?)?;}
  }
  let states=if revised {vec![("state.sf_sfclayrev.psim_stab",0),("state.sf_sfclayrev.psih_stab",1),("state.sf_sfclayrev.psim_unstab",2),("state.sf_sfclayrev.psih_unstab",3)]} else {vec![("state.module_sf_sfclay.psimtb",4),("state.module_sf_sfclay.psihtb",5)]};
  for (name,table) in states {
   let mut r=record(prefix,"pre",name)?;assert_eq!(r.data.len(),1001*4);
   r.data.copy_from_slice(&candidate[(48*n+table*1001)*4..(48*n+(table+1)*1001)*4]);write_post(output,name,r)?;
  }
  if revised {
   let mut r=record(prefix,"pre","errflg")?;r.data.copy_from_slice(&0i32.to_le_bytes());write_post(output,"errflg",r)?;
   let mut r=record(prefix,"pre","errmsg")?;r.data.fill(b' ');let msg=b"sf_sfclayrev_run OK";r.data[..msg.len()].copy_from_slice(msg);write_post(output,"errmsg",r)?;
  }
  return Ok(());
 }
 let mut out=io::stdout().lock();out.write_all(&(n as u64).to_le_bytes())?;out.write_all(&pre)?;out.write_all(&post)?;
 for name in ["cp","g","rovcp","r","xlv","svp1","svp2","svp3","svpt0","ep1","ep2","karman","p1000mb"] {let r=record(prefix,"pre",name)?;assert_eq!(r.header[0],1);assert_eq!(r.header[2],0);out.write_all(&r.data)?;}
 for name in ["isfflx","isftcflx","iz0tlnd","scm_force_flux","shalwater_z0"] {let val=if !revised && name=="shalwater_z0" {0} else {integer(prefix,name)};out.write_all(&val.to_le_bytes())?;}
 Ok(())
}
