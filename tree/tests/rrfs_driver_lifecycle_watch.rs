//! External observation of the complete pristine RRFS driver lifecycle/call trace.
//! MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver.F:127-419; mpas_atmphys_manager.F:137-832; mpas_atmphys_init.F:84-477; mpas_atmphys_interface.F:284-1286
//! at 1d6e370e. Ptrace and GNU descriptors reuse r1/r12 observer mechanisms.
//! Only breakpoint instruction bytes change; numerical storage is read-only.
//! Global IDs, full pool contents and hidden callbacks are NOT inferred.
use std::{collections::BTreeMap,env,fs,io::{self,Write},path::Path,process::{Command,exit},os::unix::process::CommandExt,ffi::c_void};
unsafe extern "C" {fn ptrace(r:i32,p:i32,a:*mut c_void,d:*mut c_void)->i64;fn waitpid(p:i32,s:*mut i32,o:i32)->i32;fn __errno_location()->*mut i32;fn process_vm_readv(p:i32,l:*const IoVec,nl:usize,r:*const IoVec,nr:usize,f:usize)->isize;}
#[repr(C)] struct IoVec{base:*mut c_void,len:usize}
#[repr(C)] #[derive(Default)] struct Registers{r15:u64,r14:u64,r13:u64,r12:u64,rbp:u64,rbx:u64,r11:u64,r10:u64,r9:u64,r8:u64,rax:u64,rcx:u64,rdx:u64,rsi:u64,rdi:u64,orig_rax:u64,rip:u64,cs:u64,eflags:u64,rsp:u64,ss:u64,fs_base:u64,gs_base:u64,ds:u64,es:u64,fs:u64,gs:u64}
fn bad(s:&str)->io::Error{io::Error::other(s)}
fn trace(r:i32,p:i32,a:u64,d:u64)->io::Result<i64>{unsafe{*__errno_location()=0;let v=ptrace(r,p,a as *mut _,d as *mut _);if v==-1&&*__errno_location()!=0{Err(io::Error::last_os_error())}else{Ok(v)}}}
fn word(p:i32,a:u64)->io::Result<u64>{Ok(trace(2,p,a,0)? as u64)}
fn wait(p:i32)->io::Result<i32>{let mut s=0;if unsafe{waitpid(p,&mut s,0)}!=p{return Err(io::Error::last_os_error())}Ok(s)}
fn arg(p:i32,r:&Registers,i:usize)->io::Result<u64>{if i<6{Ok([r.rdi,r.rsi,r.rdx,r.rcx,r.r8,r.r9][i])}else{word(p,r.rsp+8*(i as u64-5))}}
fn bytes(p:i32,a:u64,n:usize)->io::Result<Vec<u8>>{if n>256*1024*1024{return Err(bad("capture extent exceeds 256MiB"))}let mut v=vec![0;n];let mut i=0;while i<n{let l=IoVec{base:v[i..].as_mut_ptr() as *mut _,len:n-i};let r=IoVec{base:(a+i as u64) as *mut _,len:n-i};let got=unsafe{process_vm_readv(p,&l,1,&r,1,0)};if got<=0{return Err(io::Error::last_os_error())}i+=got as usize;}Ok(v)}

struct Trap { byte:u8, entry:Option<String>, returns:Vec<usize> }
struct Frame { id:usize, name:String, depth:usize, ret:u64, args:BTreeMap<String,u64> }
fn setbyte(p:i32,a:u64,b:u8)->io::Result<()>{let w=word(p,a)?;trace(5,p,a,(w&!255)|b as u64)?;Ok(())}
fn selected(s:&str)->bool {
 let Some((module,routine))=s.split_once("_MOD_")else{return false};
 ["__mpas_atmphys_control","__mpas_atmphys_manager","__mpas_atmphys_driver","__mpas_atmphys_init","__mpas_atmphys_interface","__mpas_atmphys_finalize","__mpas_atmphys_update_surface","__mpas_atmphys_update","__mpas_atmphys_o3climatology","__mpas_atmphys_sfc_diagnostics","__mpas_atmphys_todynamics"].contains(&module)
 || (module.starts_with("__mpas_atmphys_driver_") && ["driver_","allocate_","deallocate_","init_","update_convection_step"].iter().any(|v|routine.starts_with(v)))
 || (module=="__atm_core" && routine=="atm_reset_diagnostics")
}
fn emit(p:i32,f:&mut fs::File,phase:&str,frame:&Frame,globals:&[(String,u64)])->io::Result<()> {
 write!(f,"{{\"phase\":\"{phase}\",\"id\":{},\"depth\":{},\"symbol\":\"{}\",\"words\":{{",frame.id,frame.depth,frame.name)?;
 for(i,(name,at))in globals.iter().chain(frame.args.iter().map(|(n,a)|(n,a)).map(|(n,a)|(n.clone(),*a)).collect::<Vec<_>>().iter()).enumerate(){
  if i>0{write!(f,",")?}let raw=bytes(p,*at,4)?;write!(f,"\"{name}\":\"{:02x}{:02x}{:02x}{:02x}\"",raw[0],raw[1],raw[2],raw[3])?;
 }
 writeln!(f,"}}}}")?;f.flush()
}
fn observe(executable:&Path,output:&Path)->io::Result<()> {
 let executable=fs::canonicalize(executable)?;let mut log=fs::OpenOptions::new().write(true).create_new(true).open(output)?;
 let nm=Command::new("nm").arg("-S").arg(&executable).output()?;if !nm.status.success(){return Err(bad("nm failed"))}
 let symbols=String::from_utf8_lossy(&nm.stdout).lines().filter_map(|l|{let v=l.split_whitespace().collect::<Vec<_>>();if v.len()==4{Some((v[3].to_owned(),(u64::from_str_radix(v[0],16).ok()?,u64::from_str_radix(v[1],16).ok()?,v[2].to_owned())))}else{None}}).collect::<BTreeMap<_,_>>();
 let mut cmd=Command::new(&executable);unsafe{cmd.pre_exec(||{trace(0,0,0,0)?;Ok(())});}let child=cmd.spawn()?;let pid=child.id()as i32;
 if wait(pid)?&255!=127{return Err(bad("missing exec stop"))}trace(0x4200,pid,0,0x100000)?;
 let maps=fs::read_to_string(format!("/proc/{pid}/maps"))?;let bias=maps.lines().find_map(|l|{let v=l.split_whitespace().collect::<Vec<_>>();if v.len()>=6&&v[5]==executable.to_string_lossy()&&v[2]=="00000000"{u64::from_str_radix(v[0].split('-').next()?,16).ok()}else{None}}).ok_or_else(||bad("load bias absent"))?;
 let globals=symbols.iter().filter(|(n,(_,size,kind))|*size==4&&["B","D","b","d"].contains(&kind.as_str())&&["__mpas_atmphys_manager_MOD_","__mpas_atmphys_vars_MOD_","__mpas_atmphys_control_MOD_"].iter().any(|p|n.starts_with(p))).map(|(n,(a,_,_))|(n.clone(),a+bias)).collect::<Vec<_>>();
 let mut traps:BTreeMap<u64,Trap>=BTreeMap::new();for(n,(a,_,k))in &symbols{if k=="T"&&selected(n){let ip=a+bias;if traps.contains_key(&ip){return Err(bad("aliased entry symbols"))}let byte=word(pid,ip)?as u8;setbyte(pid,ip,204)?;traps.insert(ip,Trap{byte,entry:Some(n.clone()),returns:vec![]});}}
 let mut frames:Vec<Frame>=vec![];let mut count=0;trace(7,pid,0,0)?;
 loop {
  let status=wait(pid)?;if status&127==0{if !frames.is_empty(){return Err(bad("native exited inside observed routine"))}exit((status>>8)&255)}if status&255!=127{exit(128+(status&127))}
  let sig=(status>>8)&255;let mut r=Registers::default();trace(12,pid,0,&mut r as *mut _ as u64)?;let ip=r.rip-1;
  if sig!=5||!traps.contains_key(&ip){trace(7,pid,0,sig as u64)?;continue}
  let t=traps.get(&ip).unwrap();let byte=t.byte;let entry=t.entry.clone();let returning=!t.returns.is_empty();
  if returning {
   while frames.last().map(|x|x.ret==ip).unwrap_or(false){let frame=frames.pop().unwrap();let ids=&mut traps.get_mut(&ip).unwrap().returns;if ids.pop()!=Some(frame.id){return Err(bad("native return nesting mismatch"))}emit(pid,&mut log,"post",&frame,&globals)?;}
   if !traps.get(&ip).unwrap().returns.is_empty(){return Err(bad("unmatched native return trap"))}
  }
  if let Some(name)=entry {
   let ret=word(pid,r.rsp)?;let mut args=BTreeMap::new();
   if name=="__mpas_atmphys_manager_MOD_physics_timetracker"{for(i,key)in[(1,"argument.dt"),(3,"argument.itimestep"),(4,"argument.xtime_s")]{args.insert(key.to_owned(),arg(pid,&r,i)?);}}
   if name=="__mpas_atmphys_driver_MOD_physics_driver"{for(i,key)in[(1,"argument.itimestep"),(2,"argument.xtime_s")]{args.insert(key.to_owned(),arg(pid,&r,i)?);}}
   let frame=Frame{id:count,name,depth:frames.len(),ret,args};count+=1;emit(pid,&mut log,"pre",&frame,&globals)?;
   if !traps.contains_key(&ret){let b=word(pid,ret)?as u8;setbyte(pid,ret,204)?;traps.insert(ret,Trap{byte:b,entry:None,returns:vec![]});}
   traps.get_mut(&ret).unwrap().returns.push(frame.id);frames.push(frame);
  }
  setbyte(pid,ip,byte)?;r.rip=ip;trace(13,pid,0,&mut r as *mut _ as u64)?;trace(9,pid,0,0)?;
  if wait(pid)?!=(5<<8|127){return Err(bad("native single step failed"))}
  let keep=traps.get(&ip).map(|t|t.entry.is_some()||!t.returns.is_empty()).unwrap_or(false);if keep{setbyte(pid,ip,204)?}else{traps.remove(&ip);}trace(7,pid,0,0)?;
 }
}
fn main()->io::Result<()>{let args=env::args_os().collect::<Vec<_>>();if args.len()!=3{return Err(bad("rrfs-lifecycle-watch NATIVE_EXECUTABLE TRACE_JSONL"))}observe(Path::new(&args[1]),Path::new(&args[2]))}
