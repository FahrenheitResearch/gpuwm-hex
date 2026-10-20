//! Read the pristine fork's deep-entry words without rebuilding its executable.
//! Debugger stops are supporting evidence and must pass a full output comparison.
//! Authority: GFL/MPAS/module_cu_gfl_mpas.F90:116,233-253,462-550,
//! module_cu_gfl_deep.F90:85-166 at RRFS-MPAS 1d6e370e/GFL bf6a4b0.
use std::{env,fs,io::{self,Write},os::unix::process::CommandExt,path::PathBuf,process::{Command,exit}};
use std::ffi::c_void;
unsafe extern "C" {
 fn ptrace(request:i32,pid:i32,address:*mut c_void,data:*mut c_void)->i64;
 fn waitpid(pid:i32,status:*mut i32,options:i32)->i32;
 fn __errno_location()->*mut i32;
}
#[repr(C)] #[derive(Default)]
pub struct Registers {r15:u64,r14:u64,r13:u64,r12:u64,rbp:u64,rbx:u64,r11:u64,r10:u64,r9:u64,r8:u64,
 rax:u64,rcx:u64,rdx:u64,rsi:u64,rdi:u64,orig_rax:u64,rip:u64,cs:u64,eflags:u64,rsp:u64,ss:u64,
 fs_base:u64,gs_base:u64,ds:u64,es:u64,fs:u64,gs:u64}
impl Registers {
 pub fn ip(&self)->u64{self.rip}
 pub fn set_ip(&mut self,ip:u64){self.rip=ip;}
 pub fn stack(&self)->u64{self.rsp}
}
pub fn trace(request:i32,pid:i32,address:u64,data:u64)->io::Result<i64> {
 unsafe{*__errno_location()=0; let result=ptrace(request,pid,address as *mut _,data as *mut _);
  if result == -1 && *__errno_location()!=0 {Err(io::Error::last_os_error())} else {Ok(result)}}
}
pub fn word(pid:i32,address:u64)->io::Result<u64>{Ok(trace(2,pid,address,0)? as u64)}
pub fn integer(pid:i32,address:u64)->io::Result<i32>{Ok(word(pid,address)? as i32)}
pub fn argument(pid:i32,regs:&Registers,index:usize)->io::Result<u64>{
 if index<6 {Ok([regs.rdi,regs.rsi,regs.rdx,regs.rcx,regs.r8,regs.r9][index])}
 else {word(pid,regs.rsp+8*(index as u64-6+1))}
}
pub fn wait(pid:i32)->io::Result<i32>{let mut s=0;unsafe{if waitpid(pid,&mut s,0)!=pid{return Err(io::Error::last_os_error())}} Ok(s)}
fn compare_masks(left:&std::ffi::OsStr,right:&std::ffi::OsStr)->io::Result<()> {
 let l=fs::read(left)?;let r=fs::read(right)?;
 if l.len()!=r.len() || l.is_empty() || l.len()%4!=0 {return Err(io::Error::other("entry word extent mismatch"))}
 let mut values=0;let mut masks=0;let mut left_zero=0;let mut right_zero=0;let mut first=None;
 for (index,(a,b)) in l.chunks_exact(4).zip(r.chunks_exact(4)).enumerate(){
  let a=i32::from_le_bytes(a.try_into().unwrap());let b=i32::from_le_bytes(b.try_into().unwrap());
  if a!=b {values+=1;} if a==0 {left_zero+=1;} if b==0 {right_zero+=1;}
  if (a==0)!=(b==0) {masks+=1;if first.is_none(){first=Some(index);}}
 }
 let first=first.map_or_else(||"null".to_string(),|i|i.to_string());
 println!("{{\"compared_words\":{},\"different_values\":{values},\"different_zero_tests\":{masks},\"left_zero_words\":{left_zero},\"right_zero_words\":{right_zero},\"first_different_zero_test\":{first}}}",l.len()/4);
 Ok(())
}
fn main()->io::Result<()> {
 let args:Vec<_>=env::args_os().collect();
 if args.len()==4 && args[1]=="--expect-i32" {
  let expected:i32=args[2].to_string_lossy().parse().map_err(|_|io::Error::other("invalid expected integer"))?;
  let data=fs::read(&args[3])?;
  if data.is_empty() || data.len()%4!=0{return Err(io::Error::other("native integer extent mismatch"))}
  let different=data.chunks_exact(4).filter(|v|i32::from_le_bytes((*v).try_into().unwrap())!=expected).count();
  println!("{{\"expected_value\":{expected},\"compared_words\":{},\"unexpected_words\":{different}}}",data.len()/4);
  return Ok(())
 }
 if args.len()==4 && args[1]=="--compare" {return compare_masks(&args[2],&args[3]);}
 if args.len()!=3 {return Err(io::Error::other("native_entry_watch EXECUTABLE OUTPUT; --compare LEFT RIGHT"))}
 let executable=fs::canonicalize(&args[1])?;let output=PathBuf::from(&args[2]);fs::create_dir_all(&output)?;
 let symbol=Command::new("nm").arg(&executable).output()?;
 if !symbol.status.success(){return Err(io::Error::other("native symbol audit failed"))}
 let offset=String::from_utf8_lossy(&symbol.stdout).lines().find_map(|line|{
  let fields:Vec<_>=line.split_whitespace().collect();
  if fields.len()==3 && fields[2]=="__module_cu_gfl_deep_MOD_cu_gfl_deep_run" {
   Some(u64::from_str_radix(fields[0],16).unwrap())}else{None}
 }).ok_or_else(||io::Error::other("pristine GFL symbol absent"))?;
 let mut command=Command::new(&executable);
 unsafe{command.pre_exec(||{if ptrace(0,0,std::ptr::null_mut(),std::ptr::null_mut())==-1{
  Err(io::Error::last_os_error())}else{Ok(())}});}
 let child=command.spawn()?;let pid=child.id() as i32;
 let initial=wait(pid)?;if initial&255!=127{return Err(io::Error::other("no initial debugger stop"))}
 // Keep the normal randomization setting and derive this executable's load bias.
 let maps=fs::read_to_string(format!("/proc/{pid}/maps"))?;
 let executable_text=executable.to_string_lossy();
 let bias=maps.lines().find_map(|line|{let v:Vec<_>=line.split_whitespace().collect();
  if v.len()>=6 && v[5]==executable_text && v[2]=="00000000" {
   Some(u64::from_str_radix(v[0].split('-').next().unwrap(),16).unwrap())}else{None}
 }).ok_or_else(||io::Error::other("native executable load map absent"))?;
 let address=bias+offset;let original=word(pid,address)?;
 trace(5,pid,address,(original&!255)|204)?;
 let rank=env::var("PMI_RANK").unwrap_or_else(|_|"0".into());
 if !rank.chars().all(|c|c.is_ascii_digit()){return Err(io::Error::other("invalid MPI rank"))}
 let mut receipt=fs::File::create(output.join(format!("rank{rank}.jsonl")))?;let mut event=0;
 trace(7,pid,0,0)?;
 loop {
  let status=wait(pid)?;
  if status&127==0 {exit((status>>8)&255)}
  if status&255!=127 {exit(128+(status&127))}
  let signal=(status>>8)&255;let mut regs=Registers::default();
  trace(12,pid,0,&mut regs as *mut _ as u64)?;
  if signal==5 && regs.rip==address+1 {
   let itf=integer(pid,argument(pid,&regs,0)?)?;
   let its=integer(pid,argument(pid,&regs,2)?)?;
   let ite=integer(pid,argument(pid,&regs,3)?)?;
   let imid=integer(pid,argument(pid,&regs,13)?)?;
   let ierr=argument(pid,&regs,57)?;
   if its<1 || ite<its || ite-its>10000000 || itf<its || itf>ite || ![0,1].contains(&imid) {
    return Err(io::Error::other("native entry ABI bounds mismatch"))
   }
   let name=format!("rank{rank}-event{event:04}-imid{imid}-ierr.raw");
   let mut data=fs::File::create(output.join(&name))?;
   let mut nonzero=0;let mut first_nonzero=None;
   for cell in 0..=ite-its {let value=integer(pid,ierr+4*cell as u64)?;
    data.write_all(&value.to_le_bytes())?;
    if value!=0 {nonzero+=1;if first_nonzero.is_none(){first_nonzero=Some((cell+its,value));}}
   }
   let first=first_nonzero.map_or_else(||"null".to_string(),|(cell,value)|format!("{{\"cell\":{cell},\"word\":{value}}}"));
   writeln!(receipt,"{{\"event\":{event},\"rank\":{rank},\"imid\":{imid},\"its\":{its},\"ite\":{ite},\"itf\":{itf},\"word_width\":4,\"nonzero_words\":{nonzero},\"first_nonzero\":{first},\"file\":\"{name}\"}}")?;
   receipt.flush()?;event+=1;
   trace(5,pid,address,original)?;regs.rip=address;
   trace(13,pid,0,&mut regs as *mut _ as u64)?;
   trace(9,pid,0,0)?;
   if wait(pid)? != (5<<8|127) {return Err(io::Error::other("native instruction step failed"))}
   trace(5,pid,address,(original&!255)|204)?;
   trace(7,pid,0,0)?;
  }else{trace(7,pid,0,signal as u64)?;}
 }
}
