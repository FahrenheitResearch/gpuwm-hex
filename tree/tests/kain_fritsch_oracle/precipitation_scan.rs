//! Test-only native precipitation sign census. NetCDF field decoding stays
//! in Rust. Negative zero is counted separately; no values are corrected.
//! Native accumulation: mpas_atmphys_driver_convection.F:1061-1102.
use std::{ffi::{CString,CStr,c_char,c_int,c_void},ptr};
#[link(name="netcdf")]
unsafe extern "C" {
 fn nc_open(p:*const c_char,m:c_int,id:*mut c_int)->c_int;
 fn nc_close(id:c_int)->c_int;
 fn nc_inq_varid(id:c_int,n:*const c_char,v:*mut c_int)->c_int;
 fn nc_inq_var(id:c_int,v:c_int,n:*mut c_char,t:*mut c_int,nd:*mut c_int,d:*mut c_int,a:*mut c_int)->c_int;
 fn nc_inq_dimlen(id:c_int,d:c_int,len:*mut usize)->c_int;
 fn nc_get_var(id:c_int,v:c_int,p:*mut c_void)->c_int;
 fn nc_strerror(e:c_int)->*const c_char;
 fn nc_create(p:*const c_char,m:c_int,id:*mut c_int)->c_int;
 fn nc_def_dim(id:c_int,n:*const c_char,len:usize,d:*mut c_int)->c_int;
 fn nc_def_var(id:c_int,n:*const c_char,t:c_int,nd:c_int,d:*const c_int,v:*mut c_int)->c_int;
 fn nc_enddef(id:c_int)->c_int;
 fn nc_put_var(id:c_int,v:c_int,p:*const c_void)->c_int;
}
fn check(e:c_int)->Result<(),String>{if e==0{Ok(())}else{Err(unsafe{CStr::from_ptr(nc_strerror(e))}.to_string_lossy().into_owned())}}
fn q(s:&str)->String{let mut o=String::from("\"");for c in s.chars(){match c{'"'=>o.push_str("\\\""),'\\'=>o.push_str("\\\\"),'\n'=>o.push_str("\\n"),'\r'=>o.push_str("\\r"),'\t'=>o.push_str("\\t"),c if c<' '=>o.push_str(&format!("\\u{:04x}",c as u32)),c=>o.push(c)}}o.push('"');o}
struct File(c_int);impl Drop for File{fn drop(&mut self){unsafe{nc_close(self.0);}}}
#[derive(Debug)]
struct Stats{words:usize,negative:usize,negative_zero:usize,nonfinite:usize,min:f64,first:Option<(usize,u64)>}
fn stats(bits:&[u64],double:bool)->Stats{
 let mut s=Stats{words:bits.len(),negative:0,negative_zero:0,nonfinite:0,min:0.,first:None};
 for (i,&word) in bits.iter().enumerate(){
  let x=if double{f64::from_bits(word)}else{f32::from_bits(word as u32) as f64};
  if !x.is_finite(){s.nonfinite+=1;}
  if x==0. && x.is_sign_negative(){s.negative_zero+=1;}
  if x<0.{s.negative+=1;if s.first.is_none(){s.first=Some((i,word));}s.min=s.min.min(x);}
 }s
}
fn scan(path:&str,name:&str)->Result<Option<Stats>,String>{
 let mut id=0;check(unsafe{nc_open(CString::new(path).map_err(|e|e.to_string())?.as_ptr(),0,&mut id)})?;let file=File(id);let mut v=0;
 let e=unsafe{nc_inq_varid(file.0,CString::new(name).map_err(|e|e.to_string())?.as_ptr(),&mut v)};
 if e == -49{return Ok(None);}check(e)?;
 let mut ty=0;let mut nd=0;let mut dims=[0;1024];check(unsafe{nc_inq_var(id,v,ptr::null_mut(),&mut ty,&mut nd,dims.as_mut_ptr(),ptr::null_mut())})?;
 let mut total=1usize;for &d in &dims[..nd as usize]{let mut n=0;check(unsafe{nc_inq_dimlen(id,d,&mut n)})?;total=total.checked_mul(n).ok_or("shape overflow")?;}
 let bits=match ty{
  5=>{let mut p=vec![0u32;total];check(unsafe{nc_get_var(id,v,p.as_mut_ptr().cast())})?;p.into_iter().map(u64::from).collect::<Vec<_>>()},
  6=>{let mut p=vec![0u64;total];check(unsafe{nc_get_var(id,v,p.as_mut_ptr().cast())})?;p},
  _=>return Err(format!("{name}: expected native float32/float64, NetCDF type {ty}"))};
 Ok(Some(stats(&bits,ty==6)))
}
fn self_test(path:&str)->Result<(),String>{
 let bits=[0x80000001u32,(-1.5f32).to_bits(),(-0.0f32).to_bits(),0,2f32.to_bits(),f32::INFINITY.to_bits(),0x7fc01234];
 let mut id=0;check(unsafe{nc_create(CString::new(path).map_err(|e|e.to_string())?.as_ptr(),4,&mut id)})?;let file=File(id);
 let mut d=0;check(unsafe{nc_def_dim(id,c"cells".as_ptr(),bits.len(),&mut d)})?;let mut v=0;
 check(unsafe{nc_def_var(id,c"rainc".as_ptr(),5,1,&d,&mut v)})?;check(unsafe{nc_enddef(id)})?;check(unsafe{nc_put_var(id,v,bits.as_ptr().cast())})?;drop(file);
 let s=scan(path,"rainc")?.ok_or("missing test field")?;
 if (s.words,s.negative,s.negative_zero,s.nonfinite,s.first)!= (7,2,1,2,Some((0,0x80000001))) || s.min!= -1.5{return Err(format!("sign/word transport regression: {s:?}"));}
 if scan(path,"absent")?.is_some(){return Err("unexpected absent field".into());}
 println!("{{\"self_test_passed\":true,\"negative_subnormal_word\":\"0x80000001\",\"negative_zero_count\":1}}");Ok(())
}
fn run(args:&[String])->Result<(),String>{
 if args.len()==3 && args[1]=="--self-test"{return self_test(&args[2]);}
 if args.len()<3{return Err("usage: precipitation-scan FIELD FILE.nc... | --self-test SCRATCH.nc".into());}
 let mut words=0;let mut negatives=0;let mut nonfinite=0;let mut rows=Vec::new();
 for path in &args[2..]{
  if let Some(s)=scan(path,&args[1])?{
   words+=s.words;negatives+=s.negative;nonfinite+=s.nonfinite;
   let first=s.first.map(|(i,b)|format!("{{\"flat_index\":{i},\"word\":\"0x{b:x}\"}}")).unwrap_or_else(||"null".into());
   let min=if s.min.is_finite(){s.min.to_string()}else{q(&s.min.to_string())};
   rows.push(format!("{{\"file\":{},\"words\":{},\"negative_words\":{},\"negative_zero_words\":{},\"nonfinite_words\":{},\"minimum_negative\":{},\"first_negative\":{}}}",q(path),s.words,s.negative,s.negative_zero,s.nonfinite,min,first));
  }
 }
 if rows.is_empty(){return Err("no requested precipitation field in supplied files".into());}
 println!("{{\"schema\":\"mpas-native-precipitation-sign/v1\",\"field\":{},\"words\":{},\"negative_words\":{},\"nonfinite_words\":{},\"files\":[{}]}}",q(&args[1]),words,negatives,nonfinite,rows.join(","));Ok(())
}
fn main(){if let Err(e)=run(&std::env::args().collect::<Vec<_>>()){eprintln!("{e}");std::process::exit(2);}}
