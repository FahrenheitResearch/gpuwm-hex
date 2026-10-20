//! Read-only numerical observer of pristine RRFS-MPAS 1d6e370e.
//! Native authority: diagnostics/mpas_convective_diagnostics.F:79-1124;
//! physics/mpas_atmphys_driver_microphysics.F:805-827,1350-1381;
//! mpas_atm_core.F:972-1003 (all below src/core_atmosphere).
//! Ptrace/register and GNU descriptor mechanism derived from r1's
//! oracle/replay/native_{entry,argument}_watch.rs. No native numerical data,
//! heap, source, selector, ASLR setting or register argument is changed.
use std::{collections::BTreeMap,env,fs,io::{self,Write},path::{Path,PathBuf},process::{Command,exit},os::unix::process::CommandExt,ffi::{c_void,CStr,c_char}};
unsafe extern "C" {fn ptrace(r:i32,p:i32,a:*mut c_void,d:*mut c_void)->i64;fn waitpid(p:i32,s:*mut i32,o:i32)->i32;fn __errno_location()->*mut i32;fn process_vm_readv(p:i32,l:*const IoVec,nl:usize,r:*const IoVec,nr:usize,f:usize)->isize;}
#[repr(C)] struct IoVec{base:*mut c_void,len:usize}
#[repr(C)] #[derive(Default)] struct Registers{r15:u64,r14:u64,r13:u64,r12:u64,rbp:u64,rbx:u64,r11:u64,r10:u64,r9:u64,r8:u64,rax:u64,rcx:u64,rdx:u64,rsi:u64,rdi:u64,orig_rax:u64,rip:u64,cs:u64,eflags:u64,rsp:u64,ss:u64,fs_base:u64,gs_base:u64,ds:u64,es:u64,fs:u64,gs:u64}
fn bad(s:&str)->io::Error{io::Error::other(s)}
fn trace(r:i32,p:i32,a:u64,d:u64)->io::Result<i64>{unsafe{*__errno_location()=0;let v=ptrace(r,p,a as *mut _,d as *mut _);if v==-1&&*__errno_location()!=0{Err(io::Error::last_os_error())}else{Ok(v)}}}
fn word(p:i32,a:u64)->io::Result<u64>{Ok(trace(2,p,a,0)? as u64)}
fn wait(p:i32)->io::Result<i32>{let mut s=0;if unsafe{waitpid(p,&mut s,0)}!=p{return Err(io::Error::last_os_error())}Ok(s)}
fn arg(p:i32,r:&Registers,i:usize)->io::Result<u64>{if i<6{Ok([r.rdi,r.rsi,r.rdx,r.rcx,r.r8,r.r9][i])}else{word(p,r.rsp+8*(i as u64-5))}}
fn bytes(p:i32,a:u64,n:usize)->io::Result<Vec<u8>>{if n>256*1024*1024{return Err(bad("capture extent exceeds 256MiB"))}let mut v=vec![0;n];let mut i=0;while i<n{let l=IoVec{base:v[i..].as_mut_ptr() as *mut _,len:n-i};let r=IoVec{base:(a+i as u64) as *mut _,len:n-i};let got=unsafe{process_vm_readv(p,&l,1,&r,1,0)};if got<=0{return Err(io::Error::last_os_error())}i+=got as usize;}Ok(v)}
fn text(p:i32,a:u64,n:usize)->io::Result<String>{if n>512{return Err(bad("native key too long"))}let v=bytes(p,a,n)?;let s=String::from_utf8(v).map_err(|_|bad("invalid native key"))?.trim_end().to_owned();if !s.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'_'){return Err(bad(&format!("unsafe native key {s:?}, address={a:x}, length={n}")))}Ok(s)}
#[derive(Clone)] struct Array{at:u64,code:u32,width:usize,shape:Vec<usize>,lower:Vec<i64>,stride:Vec<i64>}
impl Array{
 fn scalar(at:u64,code:u32,width:usize)->Self{Self{at,code,width,shape:vec![],lower:vec![],stride:vec![]}}
 fn descriptor(p:i32,a:u64,rank:usize,code:u32)->io::Result<Option<Self>>{let d=bytes(p,a,40+24*rank)?;let u=|i|u64::from_le_bytes(d[i..i+8].try_into().unwrap());if u(0)==0{return Ok(None)}if u(16)!=4||d[28]!=rank as u8||d[29]!=(if code==1{3}else{1})||u(32)!=4{return Err(bad("GNU descriptor ABI mismatch"))}let mut shape=vec![];let mut lower=vec![];let mut stride=vec![];for i in 0..rank{let lo=u(48+24*i) as i64;let hi=u(56+24*i) as i64;if hi<lo-1||hi-lo>10000000{return Err(bad("native bounds invalid"))}shape.push((hi-lo+1) as usize);lower.push(lo);stride.push((u(40+24*i) as i64)*4);}Ok(Some(Self{at:u(0),code,width:4,shape,lower,stride}))}
 fn read(&self,p:i32)->io::Result<Vec<u8>>{let n=self.shape.iter().try_fold(self.width,|n,&d|n.checked_mul(d)).ok_or_else(||bad("extent overflow"))?;let mut expected=self.width as i64;let mut contiguous=true;for(&s,&st)in self.shape.iter().zip(&self.stride){if s>1&&expected!=st{contiguous=false}expected*=s as i64;}if contiguous{return bytes(p,self.at,n)}let mut v=Vec::with_capacity(n);for i in 0..n/self.width{let mut j=i;let mut off=0i64;for(&s,&st)in self.shape.iter().zip(&self.stride){off+=(j%s)as i64*st;j/=s;}v.extend(bytes(p,(self.at as i128+off as i128).try_into().map_err(|_|bad("strided address overflow"))?,self.width)?);}Ok(v)}
 fn save(&self,p:i32,path:&Path)->io::Result<()>{if path.exists(){return Ok(())}let payload=self.read(p)?;let mut f=fs::OpenOptions::new().create_new(true).write(true).open(path)?;f.write_all(b"DXR10001")?;for v in [self.code,self.width as u32,self.shape.len() as u32]{f.write_all(&v.to_le_bytes())?;}for i in 0..3{f.write_all(&(self.shape.get(i).copied().unwrap_or(1) as u64).to_le_bytes())?;f.write_all(&self.lower.get(i).copied().unwrap_or(1).to_le_bytes())?;}f.write_all(&payload)}
}
#[derive(Clone)] enum Kind{Boundary(String),Get{rank:usize,code:u32,mode:u8},Decision(bool),Interval,WindTail,PhysicsPublish,ReturnGet{rank:usize,code:u32,mode:u8,name:String,out:u64},ReturnDecision{inclusion:bool,name:String},ReturnInterval(u64),ReturnBoundary}
struct Trap{byte:u8,kind:Kind}
fn setbyte(p:i32,a:u64,b:u8)->io::Result<()>{let w=word(p,a)?;trace(5,p,a,(w&!255)|b as u64)?;Ok(())}
fn arm(p:i32,t:&mut BTreeMap<u64,Trap>,a:u64,k:Kind)->io::Result<()>{if t.contains_key(&a){return Err(bad("overlapping native observer return address"))}let b=word(p,a)?as u8;setbyte(p,a,204)?;t.insert(a,Trap{byte:b,kind:k});Ok(())}
fn step(p:i32,r:&mut Registers,a:u64,b:u8,rearm:bool)->io::Result<()>{setbyte(p,a,b)?;r.rip=a;trace(13,p,0,r as *mut _ as u64)?;trace(9,p,0,0)?;if wait(p)?!=(5<<8|127){return Err(bad("native single instruction step failed"))}if rearm{setbyte(p,a,204)?}Ok(())}
struct Active{kind:String,dir:PathBuf,ret:u64,start:u64,end:u64,metadata:BTreeMap<String,String>,hail_ready:bool}
fn jsonmap(v:&BTreeMap<String,String>)->String{format!("{{{}}}",v.iter().map(|(k,v)|format!("\"{k}\":{v}")).collect::<Vec<_>>().join(","))}
fn capture_outputs(p:i32,a:&Active,cache:&BTreeMap<String,Array>,outputs:&[String],phase:&str)->io::Result<()>{for(n,v)in cache{if outputs.contains(n)||n.ends_with("_acc")||n=="do_diag_dbz"||n=="refl10cm_1km_max"{if a.kind=="hail"&&(!n.ends_with("_acc")||!a.hail_ready){continue}if a.kind=="physics_reset"&&!n.ends_with("_acc")&&n!="refl10cm_1km_max"{continue}v.save(p,&a.dir.join(format!("{phase}.{n}.bin")))?;}}Ok(())}
fn observe(executable:&Path,output:&Path,fields:&Path)->io::Result<()>{
 let executable=fs::canonicalize(executable)?;fs::create_dir_all(output)?;let outputs=fs::read_to_string(fields)?.lines().map(str::to_owned).collect::<Vec<_>>();
 let nm=Command::new("nm").arg(&executable).output()?;if !nm.status.success(){return Err(bad("nm failed"))}let symbols=String::from_utf8_lossy(&nm.stdout).lines().filter_map(|l|{let v=l.split_whitespace().collect::<Vec<_>>();if v.len()==3{Some((v[2].to_owned(),u64::from_str_radix(v[0],16).ok()?))}else{None}}).collect::<BTreeMap<_,_>>();
 let mut cmd=Command::new(&executable);unsafe{cmd.pre_exec(||{trace(0,0,0,0)?;Ok(())});}let child=cmd.spawn()?;let pid=child.id()as i32;if wait(pid)?&255!=127{return Err(bad("missing exec stop"))}trace(0x4200,pid,0,0x100000)?;
 let maps=fs::read_to_string(format!("/proc/{pid}/maps"))?;let bias=maps.lines().find_map(|l|{let v=l.split_whitespace().collect::<Vec<_>>();if v.len()>=6&&v[5]==executable.to_string_lossy()&&v[2]=="00000000"{u64::from_str_radix(v[0].split('-').next()?,16).ok()}else{None}}).ok_or_else(||bad("native load bias missing"))?;
 let sym=|n:&str|symbols.get(n).copied().map(|v|v+bias).ok_or_else(||bad(&format!("missing native symbol {n}")));
 let mut traps=BTreeMap::new();let mut probes=Vec::new();
 for k in ["setup","update","compute","reset"]{let n=format!("__mpas_convective_diagnostics_MOD_convective_diagnostics_{k}");arm(pid,&mut traps,sym(&n)?,Kind::Boundary(k.into()))?;}
 arm(pid,&mut traps,sym("__atm_core_MOD_atm_reset_diagnostics")?,Kind::Boundary("physics_reset".into()))?;
 arm(pid,&mut traps,sym("__mpas_atmphys_driver_microphysics_MOD_driver_microphysics")?,Kind::Boundary("hail".into()))?;
 for rank in 0..4{for(code,ty)in[(1,"real"),(2,"int")]{probes.push((sym(&format!("__mpas_pool_routines_MOD_mpas_pool_get_array_{rank}d_{ty}"))?,Kind::Get{rank,code,mode:0}));}}
 probes.push((sym("__mpas_pool_routines_MOD_mpas_pool_get_dimension_0d")?,Kind::Get{rank:0,code:2,mode:1}));
 probes.push((sym("__mpas_pool_routines_MOD_mpas_pool_get_config_char")?,Kind::Get{rank:0,code:3,mode:2}));
 for(inclusion,n)in[(true,"mpas_stream_inclusion_count"),(false,"mpas_field_will_be_written")]{probes.push((sym(&format!("__mpas_atm_diagnostics_utils_MOD_{n}"))?,Kind::Decision(inclusion)));}
 probes.push((sym("__mpas_timekeeping_MOD_mpas_interval_division")?,Kind::Interval));
 probes.push((sym("__mpas_atmphys_interface_MOD_microphysics_to_mpas")?,Kind::PhysicsPublish));
 // Derive the allocation return from disassembly of the source-pinned update.
 // The first malloc is allocate(z_agl(nVertLevelsP1)), the next allocate(pres).
 let dis=Command::new("objdump").args(["-d","--disassemble=__mpas_convective_diagnostics_MOD_convective_diagnostics_update"]).arg(&executable).output()?;
 let listing=String::from_utf8_lossy(&dis.stdout);let lines=listing.lines().collect::<Vec<_>>();let allocation=lines.iter().position(|l|l.contains("call")&&l.contains("<malloc@plt>" )).ok_or_else(||bad("z_agl allocation call missing"))?;
 let tail_ip=u64::from_str_radix(lines[allocation+1].trim().split(':').next().unwrap(),16).map_err(|_|bad("allocation return instruction"))?+bias;
 probes.push((tail_ip,Kind::WindTail));
 let mut active:Option<Active>=None;let mut cache:BTreeMap<String,Array>=BTreeMap::new();let mut call=0;let rank=env::var("PMI_RANK").unwrap_or_else(|_|"0".into());if rank!="0"{return Err(bad("this capture contract fixes MPI1"))}trace(7,pid,0,0)?;
 loop{let status=wait(pid)?;if status&127==0{if active.is_some(){return Err(bad("native exited in boundary"))}exit((status>>8)&255)}if status&255!=127{exit(128+(status&127))}let sig=(status>>8)&255;let mut r=Registers::default();trace(12,pid,0,&mut r as *mut _ as u64)?;let at=r.rip-1;
  let Some(trap)=traps.get(&at)else{trace(7,pid,0,sig as u64)?;continue};if sig!=5{trace(7,pid,0,sig as u64)?;continue}let byte=trap.byte;let kind=trap.kind.clone();let mut rearm=true;
  // Observe direct source call sites only. Stream queries themselves invoke
  // framework pool getters; their arrays are not diagnostic routine inputs.
  if matches!(kind,Kind::Get{..}|Kind::Decision(_)|Kind::Interval){
   let a=active.as_ref().ok_or_else(||bad("probe outside boundary"))?;
   let caller=word(pid,r.rsp)?;
   if caller<a.start||caller>=a.end{step(pid,&mut r,at,byte,true)?;trace(7,pid,0,0)?;continue}
  }
  match kind{
   Kind::Boundary(kind)=>{if active.is_some(){return Err(bad("nested service boundary"))}let dir=output.join(format!("call{call:06}-{kind}"));fs::create_dir(&dir)?;call+=1;let ret=word(pid,r.rsp)?;arm(pid,&mut traps,ret,Kind::ReturnBoundary)?;let end=symbols.values().map(|v|v+bias).filter(|&v|v>at).min().ok_or_else(||bad("native routine upper bound"))?;let mut a=Active{kind:kind.clone(),dir,ret,start:at,end,metadata:BTreeMap::from([("kind".into(),format!("\"{kind}\"")),("rank".into(),"0".into())]),hail_ready:false};
    if kind=="hail"{for(i,n)in[(8,"itimestep"),(9,"its"),(10,"ite")]{a.metadata.insert(n.into(),(word(pid,arg(pid,&r,i)?)?as i32).to_string());}}
    if kind=="update"{Array::scalar(arg(pid,&r,2)?,1,4).save(pid,&a.dir.join("pre.dt.bin"))?;}
    capture_outputs(pid,&a,&cache,&outputs,"pre")?;active=Some(a);for(ip,k)in&probes{arm(pid,&mut traps,*ip,k.clone())?;}
   }
   Kind::Get{rank,code,mode}=>{let name=text(pid,arg(pid,&r,1)?,arg(pid,&r,if mode==1{3}else{4})?as usize)?;let out=arg(pid,&r,2)?;let ret=word(pid,r.rsp)?;arm(pid,&mut traps,ret,Kind::ReturnGet{rank,code,mode,name,out})?;}
   Kind::ReturnGet{rank,code,mode,name,out}=>{rearm=false;let a=active.as_mut().ok_or_else(||bad("getter return outside boundary"))?;
    if a.kind!="hail"||name.ends_with("_acc"){
     let value=if rank==0{let loc=word(pid,out)?;if loc==0{None}else{Some(Array::scalar(loc,code,if mode==2{512}else{4}))}}else{Array::descriptor(pid,out,rank,code)?};
     if let Some(v)=value{if mode==1{a.metadata.insert(name.clone(),(word(pid,v.at)?as i32).to_string());}if mode==2{let val=String::from_utf8_lossy(&v.read(pid)?).trim_end().to_owned();a.metadata.insert(name.clone(),format!("\"{val}\""));}v.save(pid,&a.dir.join(format!("pre.{name}.bin")))?;cache.insert(name.clone(),v);}
     else{a.metadata.insert(format!("absent_{name}"),"true".into());}
    }
    if a.kind=="hail"&&name=="max_hail_diameter_column_acc"{
     a.hail_ready=true;for n in ["jts","jte"]{a.metadata.insert(n.into(),(word(pid,sym(&format!("__mpas_atmphys_vars_MOD_{n}"))?)?as i32).to_string());}
     for n in ["max_hail_diameter_sfc_p","max_hail_diameter_column_p"]{let v=Array::descriptor(pid,sym(&format!("__mpas_atmphys_vars_MOD_{n}"))?,2,1)?.ok_or_else(||bad("native hail tile absent"))?;v.save(pid,&a.dir.join(format!("pre.{n}.bin")))?;}
    }
   }
   Kind::Decision(inclusion)=>{let name=text(pid,arg(pid,&r,0)?,arg(pid,&r,if inclusion{3}else{1})?as usize)?;arm(pid,&mut traps,word(pid,r.rsp)?,Kind::ReturnDecision{inclusion,name})?;}
   Kind::ReturnDecision{inclusion,name}=>{rearm=false;active.as_mut().unwrap().metadata.insert(format!("{}_{name}",if inclusion{"included"}else{"written"}),(r.rax as i32).to_string());}
   Kind::Interval=>{arm(pid,&mut traps,word(pid,r.rsp)?,Kind::ReturnInterval(arg(pid,&r,3)?))?;}
   Kind::ReturnInterval(out)=>{rearm=false;active.as_mut().unwrap().metadata.insert("nsteps".into(),(word(pid,out)?as i64).to_string());}
   Kind::WindTail=>{let a=active.as_ref().unwrap();if a.kind=="update"{let nl=a.metadata.get("nVertLevels").ok_or_else(||bad("z_agl dimension not captured"))?.parse::<u64>().map_err(|_|bad("level dimension"))?;Array::scalar(r.rax+nl*4,1,4).save(pid,&a.dir.join("pre.wind_z_agl_tail.bin"))?;}}
   Kind::PhysicsPublish=>{let a=active.as_ref().unwrap();if a.kind=="hail"&&a.hail_ready{capture_outputs(pid,a,&cache,&outputs,"post")?;}}
   Kind::ReturnBoundary=>{rearm=false;let a=active.take().unwrap();if at!=a.ret{return Err(bad("boundary return mismatch"))}capture_outputs(pid,&a,&cache,&outputs,"post")?;
    for(n,ip)in&symbols{if n.starts_with("__mpas_convective_diagnostics_MOD_is_needed_"){let v=word(pid,ip+bias)?as i32;let _=v;}}
    fs::write(a.dir.join("metadata.json"),jsonmap(&a.metadata))?;for(ip,_)in&probes{let t=traps.remove(ip).ok_or_else(||bad("probe trap lost"))?;setbyte(pid,*ip,t.byte)?;}
   }
  }
  step(pid,&mut r,at,byte,rearm)?;if !rearm{traps.remove(&at);}trace(7,pid,0,0)?;
 }
}
// Rust alone decodes DXR payloads, preserves every bit, gathers qv and exposes
// resident host ABI buffers. Python only launches CUDA and copies ABI storage.
pub struct HostArray{words:Vec<u32>,shape:Vec<usize>,lower:Vec<i64>,code:u32,width:usize}
fn load(path:&Path,select_first:i32,trim_last:usize)->io::Result<HostArray>{let v=fs::read(path)?;if v.len()<68||&v[..8]!=b"DXR10001"{return Err(bad("capture header"))}let u32at=|i|u32::from_le_bytes(v[i..i+4].try_into().unwrap());let u64at=|i|u64::from_le_bytes(v[i..i+8].try_into().unwrap());let code=u32at(8);let width=u32at(12)as usize;let rank=u32at(16)as usize;if rank>3||![4,8,512].contains(&width){return Err(bad("capture rank/width"))}let mut shape=(0..rank).map(|i|u64at(20+16*i)as usize).collect::<Vec<_>>();let mut lower=(0..rank).map(|i|u64at(28+16*i)as i64).collect::<Vec<_>>();let mut payload=v[68..].to_vec();if payload.len()!=shape.iter().product::<usize>()*width{return Err(bad("capture byte extent"))}if trim_last!=0{let n=shape.last_mut().ok_or_else(||bad("scalar trimming"))?;if trim_last>*n{return Err(bad("trim grows native extent"))}payload.truncate(payload.len()/ *n *trim_last);*n=trim_last;}
 if select_first>=0{let n=*shape.first().ok_or_else(||bad("scalar selection"))?;let s=select_first as usize;if s>=n{return Err(bad("qv index outside native slot"))}payload=payload.chunks_exact(n*width).flat_map(|p|p[s*width..(s+1)*width].iter().copied()).collect();shape.remove(0);lower.remove(0);}
 let words=payload.chunks_exact(4).map(|p|u32::from_le_bytes(p.try_into().unwrap())).collect();shape.reverse();lower.reverse();Ok(HostArray{words,shape,lower,code,width})}
#[unsafe(no_mangle)] pub unsafe extern "C" fn dx_load(path:*const c_char,select:i32,trim:usize,shape:*mut usize,lower:*mut i64,rank:*mut usize,code:*mut u32,width:*mut usize)->*mut HostArray{let result=(||{let path=unsafe{CStr::from_ptr(path)}.to_str().map_err(|_|bad("path utf8"))?;load(Path::new(path),select,trim)})();match result{Ok(a)=>{unsafe{*rank=a.shape.len();*code=a.code;*width=a.width;for(i,&n)in a.shape.iter().enumerate(){*shape.add(i)=n;*lower.add(i)=a.lower[i];}}Box::into_raw(Box::new(a))},Err(e)=>{eprintln!("DX load: {e}");std::ptr::null_mut()}}}
#[unsafe(no_mangle)] pub unsafe extern "C" fn dx_data(p:*const HostArray)->*const u32{unsafe{(*p).words.as_ptr()}}
#[unsafe(no_mangle)] pub unsafe extern "C" fn dx_words(p:*const HostArray)->usize{unsafe{(*p).words.len()}}
#[unsafe(no_mangle)] pub unsafe extern "C" fn dx_free(p:*mut HostArray){if !p.is_null(){drop(unsafe{Box::from_raw(p)});}}
#[unsafe(no_mangle)] pub unsafe extern "C" fn dx_compare(p:*const HostArray,actual:*const u32,n:usize,first:*mut usize)->i64{let a=unsafe{&*p};if n!=a.words.len(){return -1}let b=unsafe{std::slice::from_raw_parts(actual,n)};let mut d=0;unsafe{*first=usize::MAX;}for(i,(&x,&y))in a.words.iter().zip(b).enumerate(){if x!=y{if d==0{unsafe{*first=i;}}d+=1;}}d}
fn audit(root:&Path)->io::Result<()>{
 let mut paths=Vec::new();for d in fs::read_dir(root)?{let d=d?;if !d.file_type()?.is_dir(){continue}for f in fs::read_dir(d.path())?{let f=f?;let name=f.file_name().to_string_lossy().into_owned();if name=="pre.wind_z_agl_tail.bin"||name=="post.wind02_integral_max.bin"||name.starts_with("pre.max_hail_diameter_"){paths.push(f.path());}}}paths.sort();let mut rows=Vec::new();
 for path in paths{let a=load(&path,-1,0)?;let nonzero=a.words.iter().filter(|&&w|w&0x7fffffff!=0).count();let nan=a.words.iter().filter(|&&w|w&0x7fffffff>0x7f800000).count();let bits=if a.shape.is_empty(){format!("\"{:08x}\"",a.words[0])}else{"null".into()};rows.push(format!("{{\"path\":\"{}\",\"words\":{},\"nonzero_words\":{nonzero},\"nan_words\":{nan},\"scalar_bits\":{bits}}}",path.strip_prefix(root).unwrap().display(),a.words.len()));}
 println!("{{\"schema\":\"mpas.r12-native-word-audit/v1\",\"fields\":[{}]}}",rows.join(","));Ok(())
}
fn main()->io::Result<()>{let args=env::args_os().collect::<Vec<_>>();if args.len()==3&&args[1]=="--audit"{return audit(Path::new(&args[2]));}if args.len()!=4{return Err(bad("diagnostics-watch NATIVE_EXECUTABLE CAPTURE_DIRECTORY FIELD_NAMES"))}observe(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]))}
#[cfg(test)] mod tests{
 use super::*;
 fn fixture(shape:&[usize],lower:&[i64],data:&[u32])->PathBuf{
  let path=PathBuf::from(env::var("TMPDIR").expect("data disk TMPDIR required")).join(format!("r12-reader-{}-{}.dxr",std::process::id(),shape.len()));
  fs::create_dir_all(path.parent().unwrap()).unwrap();let mut b=b"DXR10001".to_vec();for v in[1u32,4,shape.len()as u32]{b.extend(v.to_le_bytes());}for i in 0..3{b.extend((shape.get(i).copied().unwrap_or(1)as u64).to_le_bytes());b.extend(lower.get(i).copied().unwrap_or(1).to_le_bytes());}for v in data{b.extend(v.to_le_bytes());}fs::write(&path,b).unwrap();path
 }
 #[test]fn slot_gather_preserves_native_nan_payloads_and_subnormals(){
  let p=fixture(&[2,2,2],&[-1,1,0],&[0x7fc01234,1,0x80000000,0x007fffff,0xffc05678,0x80000001,0x3f800000,0x807fffff]);
  let a=load(&p,1,0).unwrap();assert_eq!(a.shape,vec![2,2]);assert_eq!(a.lower,vec![0,1]);assert_eq!(a.words,vec![1,0x007fffff,0x80000001,0x807fffff]);
  let a=load(&p,0,1).unwrap();assert_eq!(a.shape,vec![1,2]);assert_eq!(a.words,vec![0x7fc01234,0x80000000]);assert!(load(&p,2,0).is_err());assert!(load(&p,-1,3).is_err());fs::remove_file(p).unwrap();
 }
 #[test]fn malformed_payload_cannot_produce_a_partial_comparison(){let p=fixture(&[4],&[1],&[1,2,3]);assert!(load(&p,-1,0).is_err());fs::remove_file(p).unwrap();}
 #[test]fn scalar_abi_and_signed_zero_compare_every_word(){let p=fixture(&[],&[],&[0x80000000]);let a=load(&p,-1,0).unwrap();assert!(a.shape.is_empty());assert_eq!(a.words,vec![0x80000000]);let mut first=0;let different=[0u32];assert_eq!(unsafe{dx_compare(&a,different.as_ptr(),1,&mut first)},1);assert_eq!(first,0);assert_eq!(unsafe{dx_compare(&a,different.as_ptr(),0,&mut first)},-1);fs::remove_file(p).unwrap();}
}
