//! Diagnostic SWD1 decoder. Rust owns endian conversion and field I/O.
use std::{env,fs,io::{self,Read,Seek},path::Path};
fn integer(f:&mut fs::File)->io::Result<i32>{let mut b=[0;4];f.read_exact(&mut b)?;Ok(i32::from_be_bytes(b))}
fn main()->io::Result<()> {
 let args:Vec<String>=env::args().collect();let mut f=fs::File::open(&args[1])?;
 let out=Path::new(&args[2]);fs::create_dir_all(out)?;
 let mut magic=[0;4];f.read_exact(&mut magic)?;assert_eq!(&magic,b"SWD1");
 let mut rows=Vec::new();let mut index=0;
 let grouped=args.get(3).is_some_and(|s|s=="columns");
 let mut groups:std::collections::BTreeMap<String,(i32,Vec<i32>,Vec<u8>,usize)>=std::collections::BTreeMap::new();
 let end=f.metadata()?.len();
 while f.stream_position()?<end {
  let length=integer(&mut f)?;
  assert!((1..1024).contains(&length));let mut name=vec![0;length as usize];f.read_exact(&mut name)?;
  let name=String::from_utf8(name).unwrap();let typ=integer(&mut f)?;let rank=integer(&mut f)?;
  assert!((0..=5).contains(&rank));assert!((0..=1).contains(&typ));
  let mut shape=Vec::new();let mut count=1usize;
  for _ in 0..rank{let n=integer(&mut f)?;assert!(n>=0);shape.push(n);count=count.checked_mul(n as usize).unwrap();}
  let mut data=vec![0u8;count*4];f.read_exact(&mut data)?;
  for word in data.chunks_exact_mut(4){word.reverse();}
  if grouped {
   let (_,key)=name.split_once('/').expect("native column prefix");
   let entry=groups.entry(key.to_string()).or_insert((typ,shape.clone(),Vec::new(),0));
   assert_eq!(entry.0,typ);assert_eq!(entry.1,shape);entry.2.extend(data);entry.3+=1;
   continue;
  }
  let file=format!("{index:05}.bin");fs::write(out.join(&file),data)?;
  rows.push(format!("{{\"name\":\"{name}\",\"file\":\"{file}\",\"dtype\":\"{}\",\"shape\":{shape:?}}}",if typ==0{"<f4"}else{"<i4"}));index+=1;
 }
 for (name,(typ,mut shape,data,count)) in groups {
  shape.push(count as i32);let file=format!("{index:05}.bin");fs::write(out.join(&file),data)?;
  rows.push(format!("{{\"name\":\"{name}\",\"file\":\"{file}\",\"dtype\":\"{}\",\"shape\":{shape:?}}}",if typ==0{"<f4"}else{"<i4"}));index+=1;
 }
 fs::write(out.join("manifest.json"),format!("{{\"order\":\"F\",\"fields\":[{}]}}\n",rows.join(",")))
}
