//! Observe CAM boundaries in the unchanged pinned native process.
//! MPAS module_ra_cam.F:195-232,335-336,678-679. No native heap/code-file edit.
use std::{env,fs::{self,File},io,os::{unix::{fs::FileExt,process::CommandExt}},path::{Path,PathBuf},process::{Child,Command},ffi::c_void};
unsafe extern "C" {
 fn ptrace(request:usize,pid:i32,address:*mut c_void,data:*mut c_void)->i64;
 fn waitpid(pid:i32,status:*mut i32,options:i32)->i32;
}
#[repr(C)] #[derive(Clone,Copy,Default)]
struct Registers {r15:u64,r14:u64,r13:u64,r12:u64,rbp:u64,rbx:u64,r11:u64,r10:u64,r9:u64,r8:u64,rax:u64,rcx:u64,rdx:u64,rsi:u64,rdi:u64,orig_rax:u64,rip:u64,cs:u64,eflags:u64,rsp:u64,ss:u64,fs_base:u64,gs_base:u64,ds:u64,es:u64,fs:u64,gs:u64}
struct Owned(Child,bool);
impl Drop for Owned {fn drop(&mut self){if !self.1 {let _=self.0.kill();let _=self.0.wait();}}}
fn control(request:usize,pid:i32,address:u64,data:u64) {
 let result=unsafe {ptrace(request,pid,address as *mut c_void,data as *mut c_void)};
 assert_eq!(result,0,"ptrace {request}: {}",io::Error::last_os_error());
}
fn wait(pid:i32)->i32 {let mut status=0;assert_eq!(unsafe{waitpid(pid,&mut status,0)},pid);status}
fn regs(pid:i32)->Registers {let mut r=Registers::default();control(12,pid,0,(&mut r as *mut Registers) as u64);r}
fn set_regs(pid:i32,r:&Registers){control(13,pid,0,(r as *const Registers) as u64)}
fn read(mem:&File,address:u64,bytes:usize)->Vec<u8>{let mut data=vec![0;bytes];mem.read_exact_at(&mut data,address).unwrap();data}
fn u64_at(mem:&File,address:u64)->u64{u64::from_le_bytes(read(mem,address,8).try_into().unwrap())}
fn argument(mem:&File,r:&Registers,index:usize)->u64 {
 if index<6 {[r.rdi,r.rsi,r.rdx,r.rcx,r.r8,r.r9][index]}else{u64_at(mem,r.rsp+8*(index-5) as u64)}
}
fn scalar(mem:&File,r:&Registers,index:usize)->i32 {i32::from_le_bytes(read(mem,argument(mem,r,index),4).try_into().unwrap())}
fn breakpoint(mem:&File,pid:i32,address:u64)->u64 {let old=u64_at(mem,address);control(5,pid,address,(old&!255)|204);old}
fn run(binary:&Path,entry:u64,out:&Path,indices:&[usize])->i32 {
 assert_eq!(indices.len(),8);fs::create_dir_all(out).unwrap();
 let mut command=Command::new(binary);
 unsafe{command.pre_exec(||{if ptrace(0,0,std::ptr::null_mut(),std::ptr::null_mut())<0 {return Err(io::Error::last_os_error());}Ok(())});}
 let mut child=Owned(command.spawn().unwrap(),false);let pid=child.0.id() as i32;
 let status=wait(pid);assert_eq!(status&255,127,"native did not stop after exec");
 let maps=fs::read_to_string(format!("/proc/{pid}/maps")).unwrap();
 let mapped_binary=fs::canonicalize(binary).unwrap();let mapped_binary=mapped_binary.to_str().unwrap();
 let base=maps.lines().find_map(|line|{let cols:Vec<_>=line.split_whitespace().collect();if cols.len()>5 && cols[2]=="00000000" && cols[5]==mapped_binary {Some(u64::from_str_radix(cols[0].split('-').next().unwrap(),16).unwrap())}else{None}}).expect("native PIE load address absent");
 let mem=File::open(format!("/proc/{pid}/mem")).unwrap();let address=base+entry;let original=breakpoint(&mem,pid,address);let rank=env::var("PMI_RANK").unwrap_or_else(|_|"0".into());
 let mut pending:Option<(u64,u64,[u64;2],Vec<i32>)>=None;let mut calls=0;
 control(7,pid,0,0);
 loop {
  let status=wait(pid);
  if status&127==0 {child.1=true;return (status>>8)&255;}
  if status&255!=127 {child.1=true;return 128+(status&127);}
  let signal=(status>>8)&255;
  if signal!=5 {control(7,pid,0,signal as u64);continue;}
  let mut r=regs(pid);let at=r.rip-1;
  if at==address && pending.is_none() {
   control(5,pid,address,original);r.rip=address;set_regs(pid,&r);
   let bounds:Vec<_>=indices[2..].iter().map(|i|scalar(&mem,&r,*i)).collect();
   for pair in bounds.chunks(2){assert!(pair[1]>=pair[0] && pair[1]-pair[0]<1_000_000);}
   let pointers=[argument(&mem,&r,indices[0]),argument(&mem,&r,indices[1])];
   let ret=u64_at(&mem,r.rsp);let old=breakpoint(&mem,pid,ret);pending=Some((ret,old,pointers,bounds));control(7,pid,0,0);
  } else if let Some((ret,old,pointers,bounds))=pending.take() {
   assert_eq!(at,ret,"unexpected native trap; no capture inferred");control(5,pid,ret,old);r.rip=ret;set_regs(pid,&r);calls+=1;
   let count=bounds.chunks(2).try_fold(1usize,|a,p|a.checked_mul((p[1]-p[0]+1) as usize)).unwrap();assert!(count<100_000_000);
   for (name,pointer) in ["taucldc","taucldi"].into_iter().zip(pointers) {
    let data=read(&mem,pointer,count*4);let nonzero=data.chunks_exact(4).filter(|v|*v!=[0,0,0,0]).count();
    let path=out.join(format!("rank{rank}-call{calls}-{name}.raw"));fs::write(&path,&data).unwrap();
    println!("CAM_NATIVE_TRACE {{\"rank\":{},\"call\":{},\"field\":\"{}\",\"bounds\":{:?},\"words\":{},\"nonzero_words\":{},\"path\":\"{}\"}}",rank,calls,name,bounds,count,nonzero,path.display());
   }
   assert_eq!(breakpoint(&mem,pid,address),original);control(7,pid,0,0);
  } else {panic!("unexpected native trap at {at:x}");}
 }
}
fn main(){let args:Vec<_>=env::args().collect();let entry=u64::from_str_radix(args[2].trim_start_matches("0x"),16).unwrap();let indices:Vec<_>=args[4].split(',').map(|v|v.parse().unwrap()).collect();let out=PathBuf::from(&args[3]);std::process::exit(run(Path::new(&args[1]),entry,&out,&indices));}
