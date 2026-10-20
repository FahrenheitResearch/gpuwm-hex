//! Compare full-module diagnostic integer-word traces, with no tolerance.
use std::{collections::BTreeMap,env,fs,io};
type Key=(i32,i32,String,String);
fn read(path:&str)->io::Result<(Vec<Key>,BTreeMap<Key,Vec<i32>>)> {
 let mut order=Vec::new();let mut records=BTreeMap::new();
 for line in fs::read_to_string(path)?.lines(){
  let Some(start)=line.find("GFLTRACE")else{continue;};
  let values:Vec<_>=line[start..].split_whitespace().collect();
  if values.len()<6{return Err(io::Error::other("truncated trace"));}
  let parse=|s:&str|s.parse::<i32>().map_err(io::Error::other);
  // GNU list-directed adjacent CHARACTER fields can be concatenated.
  let routines=["get_partition_liq_ice","cup_forcing_ens_3d","cup_output_ens_3d","get_lateral_massflux","get_inversion_layers","get_melting_profile","cup_up_moisture","cup_dd_moisture","cup_env_clev","rates_up_pdf","cup_minimi","cup_kbcon","cup_up_aa0","cup_dd_edt","dellas_nosub","cup_env"];
  let (routine,field,offset)=if routines.contains(&values[3]){
   (values[3].into(),values[4].into(),5)
  }else if let Some(r)=routines.iter().find(|r|values[3].starts_with(**r)&&values[3].len()>r.len()){
   ((*r).to_string(),values[3][r.len()..].to_string(),4)
  }else{(values[3].into(),values[4].into(),5)};
  let key=(parse(values[1])?,parse(values[2])?,routine,field);
  let words=values[offset..].iter().map(|s|parse(s)).collect::<io::Result<Vec<_>>>()?;
  if records.insert(key.clone(),words).is_some(){return Err(io::Error::other("duplicate trace key; compare one capture at a time"));}
  order.push(key);
 }
 if order.is_empty(){return Err(io::Error::other("no trace records"));}
 Ok((order,records))
}
fn main()->io::Result<()> {
 let args:Vec<_>=env::args().collect();if args.len()!=3{return Err(io::Error::other("compare_trace LEFT RIGHT"));}
 let (order,left)=read(&args[1])?;let (_,right)=read(&args[2])?;
 let mut differing=0;let mut missing=0;let mut first=true;
 for key in order{
  let l=&left[&key];let Some(r)=right.get(&key)else{missing+=1;continue;};
  if l.len()!=r.len(){return Err(io::Error::other("trace word count differs"));}
  let count=l.iter().zip(r).filter(|(a,b)|a!=b).count();differing+=count;
  if count>0{
   let index=l.iter().zip(r).position(|(a,b)|a!=b).unwrap();
   println!("{{\"imid\":{},\"phase\":{},\"routine\":\"{}\",\"field\":\"{}\",\"different_words\":{},\"first_index\":{},\"left_word\":{},\"right_word\":{},\"first_divergence\":{}}}",key.0,key.1,key.2,key.3,count,index,l[index],r[index],first);
   first=false;
  }
 }
 missing+=right.keys().filter(|k|!left.contains_key(*k)).count();
 println!("{{\"different_words\":{differing},\"unpaired_records\":{missing},\"left_records\":{},\"right_records\":{}}}",left.len(),right.len());
 Ok(())
}
