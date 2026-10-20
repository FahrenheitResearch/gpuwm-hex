//! Diagnostic raw-word provenance search; never supplies data to a replay.
//! MPAS module_ra_cam.F:364-365,678-679 copies unwritten LW-only scratch.
use std::{collections::HashMap,env,fs,path::Path};
fn words(path:&Path)->(i32,Vec<u32>) {
 let bytes=fs::read(path).unwrap(); assert!(bytes.len()>=860);assert_eq!(&bytes[..8],b"ORCL0001");
 let int=|i:usize|i32::from_le_bytes(bytes[8+i*4..12+i*4].try_into().unwrap());
 let kind=int(0);assert!(int(16)>=0);let offset=860+int(16) as usize*4;assert!(offset<=bytes.len());assert_eq!((bytes.len()-offset)%4,0);
 let values=bytes[offset..].chunks_exact(4).map(|v|u32::from_le_bytes(v.try_into().unwrap())).collect();
 (kind,values)
}
fn main() {
 let args:Vec<_>=env::args().collect(); let root=Path::new(&args[1]);
 let mut provenance:HashMap<u32,(String,usize)>=HashMap::new();let mut rounded:HashMap<u32,(String,usize)>=HashMap::new();let mut files:Vec<_>=fs::read_dir(root).unwrap().map(|e|e.unwrap().path()).collect();files.sort();
 let mut pre_records=0;let mut pre_words=0;
 for file in &files {
  let name=file.file_name().unwrap().to_str().unwrap();if !name.contains(".pre."){continue;}
  let (kind,data)=words(file);if kind!=1 && kind!=2{continue;}pre_records+=1;pre_words+=data.len();
  if kind==2 {assert_eq!(data.len()%2,0);for (index,pair) in data.chunks_exact(2).enumerate(){let value=f64::from_bits(u64::from(pair[0])|(u64::from(pair[1])<<32));let word=(value as f32).to_bits();rounded.entry(word).or_insert_with(||(name.to_string(),index));}}
  for (index,word) in data.into_iter().enumerate(){provenance.entry(word).or_insert_with(||(name.to_string(),index));}
 }
 println!("{{\"scope\":\"raw and rounded-real8 PRE provenance diagnostic; no reproduction or exclusion\",\"pre_records\":{},\"pre_words\":{},\"fields\":[",pre_records,pre_words);
 let mut comma=false;
 for field in ["taucldc","taucldi"] {
  let native=files.iter().find(|p|p.file_name().unwrap().to_str().unwrap().ends_with(&format!(".post.{}.bin",field))).unwrap();
  let (_,native)=words(native);let (_,cuda)=words(&Path::new(&args[2]).join(files.iter().find(|p|p.file_name().unwrap().to_str().unwrap().ends_with(&format!(".post.{}.bin",field))).unwrap().file_name().unwrap()));assert_eq!(native.len(),cuda.len());
  let differing:Vec<_>=native.iter().zip(&cuda).enumerate().filter(|(_, (a,b))|a!=b).collect();
  let matched=differing.iter().filter(|(_, (a,_))|provenance.contains_key(a)).count();
  let rounded_matched=differing.iter().filter(|(_, (a,_))|rounded.contains_key(a)).count();
  if comma{println!(",");}comma=true;
  print!("{{\"name\":\"{}\",\"words\":{},\"differing_words\":{},\"matching_pre_words\":{},\"matching_rounded_real8_pre_values\":{},\"samples\":[",field,native.len(),differing.len(),matched,rounded_matched);
  for (sample,(index,(a,b))) in differing.iter().take(32).enumerate() {
   if sample>0{print!(",");}print!("{{\"index\":{},\"native\":\"{:08x}\",\"cuda\":\"{:08x}\",\"entry_match\":",index,a,b);
   if let Some((name,index))=provenance.get(a){print!("{{\"file\":\"{}\",\"index\":{}}}",name,index);}else{print!("null");}print!(",\"rounded_real8_entry_match\":");if let Some((name,index))=rounded.get(a){print!("{{\"file\":\"{}\",\"index\":{}}}",name,index);}else{print!("null");}print!("}}");
  }
  print!("]}}");
 }
 println!("]}}");
}
