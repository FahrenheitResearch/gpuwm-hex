//! Observe automatic PRE storage in an unchanged strict CAM executable.
//! MPAS module_ra_cam.F:195-232,364-365,678-679. No injected library or heap call.
//! Instruction breakpoints are confined to this tracer's own child process.
use std::{env, ffi::c_void, fs::{self, File, OpenOptions}, io::{self, Write},
          os::unix::{fs::FileExt, process::CommandExt}, path::Path,
          process::{Child, Command}};
unsafe extern "C" {
    fn ptrace(request: usize, pid: i32, address: *mut c_void, data: *mut c_void) -> i64;
    fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct Registers {
    r15:u64,r14:u64,r13:u64,r12:u64,rbp:u64,rbx:u64,r11:u64,r10:u64,
    r9:u64,r8:u64,rax:u64,rcx:u64,rdx:u64,rsi:u64,rdi:u64,orig_rax:u64,
    rip:u64,cs:u64,eflags:u64,rsp:u64,ss:u64,fs_base:u64,gs_base:u64,
    ds:u64,es:u64,fs:u64,gs:u64,
}
struct Owned(Child, bool);
impl Drop for Owned {
    fn drop(&mut self) { if !self.1 {let _=self.0.kill();let _=self.0.wait();} }
}
fn control(request:usize,pid:i32,address:u64,data:u64) {
    let result=unsafe {ptrace(request,pid,address as *mut c_void,data as *mut c_void)};
    assert_eq!(result,0,"ptrace {request}: {}",io::Error::last_os_error());
}
fn wait(pid:i32)->i32 {
    let mut status=0;assert_eq!(unsafe {waitpid(pid,&mut status,0)},pid);status
}
fn regs(pid:i32)->Registers {
    let mut r=Registers::default();control(12,pid,0,(&mut r as *mut Registers) as u64);r
}
fn set_regs(pid:i32,r:&Registers) {control(13,pid,0,(r as *const Registers) as u64)}
fn read(mem:&File,address:u64,bytes:usize)->Vec<u8> {
    let mut data=vec![0;bytes];mem.read_exact_at(&mut data,address).unwrap();data
}
fn u64_at(mem:&File,address:u64)->u64 {
    u64::from_le_bytes(read(mem,address,8).try_into().unwrap())
}
fn argument(mem:&File,r:&Registers,index:usize)->u64 {
    if index<6 {[r.rdi,r.rsi,r.rdx,r.rcx,r.r8,r.r9][index]}
    else {u64_at(mem,r.rsp+8*(index-5) as u64)}
}
fn scalar(mem:&File,r:&Registers,index:usize)->i32 {
    i32::from_le_bytes(read(mem,argument(mem,r,index),4).try_into().unwrap())
}
fn breakpoint(mem:&File,pid:i32,address:u64)->u64 {
    let old=u64_at(mem,address);control(5,pid,address,(old&!255)|204);old
}
fn compact(file:&mut File,columns:usize,levels:usize,raw:&[u8]) {
    for value in [2,1,columns as i32,0,levels as i32-1] {
        file.write_all(&value.to_be_bytes()).unwrap();
    }
    for word in raw.chunks_exact(8) {
        file.write_all(&u64::from_le_bytes(word.try_into().unwrap()).to_be_bytes()).unwrap();
    }
}
struct Pending {ret:u64,old:u64,columns:usize,levels:usize,seen:[bool;2]}
fn run(binary:&Path,entry:u64,sites:[u64;2],out:&Path,args:&[String])->i32 {
    fs::create_dir_all(out).unwrap();
    let mut streams:[File;2]=["tauxci","tauxcl"].map(|name|
        OpenOptions::new().write(true).create_new(true).open(out.join(format!("entry-{name}.bin"))).unwrap());
    let mut command=Command::new(binary);command.args(args);
    unsafe {command.pre_exec(|| {
        if ptrace(0,0,std::ptr::null_mut(),std::ptr::null_mut())<0 {
            return Err(io::Error::last_os_error());
        } Ok(())
    });}
    let mut child=Owned(command.spawn().unwrap(),false);let pid=child.0.id() as i32;
    let status=wait(pid);assert_eq!(status&255,127,"native did not stop after exec");
    let maps=fs::read_to_string(format!("/proc/{pid}/maps")).unwrap();
    let mapped_binary=fs::canonicalize(binary).unwrap();
    let base=maps.lines().find_map(|line| {
        let cols:Vec<_>=line.split_whitespace().collect();
        if cols.len()>5 && cols[2]=="00000000" && cols[5]==mapped_binary.to_str().unwrap() {
            Some(u64::from_str_radix(cols[0].split('-').next().unwrap(),16).unwrap())
        } else {None}
    }).expect("native PIE load address absent");
    let mem=File::open(format!("/proc/{pid}/mem")).unwrap();
    let address=base+entry;let sites=sites.map(|site|base+site);
    let original=breakpoint(&mem,pid,address);
    let mut allocation_original=[0u64;2];let mut pending:Option<Pending>=None;
    let mut calls=0;control(7,pid,0,0);
    loop {
        let status=wait(pid);
        if status&127==0 {
            child.1=true;
            if let Some(call)=pending {
                // MPAS module_ra_cam_support.F:2651 can STOP inside CAM.
                // Report the actual process exit; terminal-call qualification
                // belongs to the caller. Both PRE allocations must be captured.
                assert_eq!(call.seen,[true;2],"native exited before both PRE allocations");
                println!("CAM_NATIVE_ENTRY_TERMINAL calls_returned={calls} terminal_call={} exit={}",
                         calls+1,(status>>8)&255);
            } else {
                println!("CAM_NATIVE_ENTRY_COMPLETE calls={calls} exit={}",(status>>8)&255);
            }
            return (status>>8)&255;
        }
        if status&255!=127 {child.1=true;return 128+(status&127);}
        let signal=(status>>8)&255;
        if signal!=5 {control(7,pid,0,signal as u64);continue;}
        let mut r=regs(pid);let at=r.rip-1;
        if at==address && pending.is_none() {
            control(5,pid,address,original);r.rip=address;set_regs(pid,&r);
            // The source's exact CAM argument list, including all domain/tile
            // bounds, is asserted by the Python metadata adapter before launch.
            let bounds=[97,98,101,102].map(|index|scalar(&mem,&r,index));
            assert!(bounds[1]>=bounds[0] && bounds[3]>=bounds[2]);
            let columns=(bounds[1]-bounds[0]+1) as usize;
            let levels=(bounds[3]-bounds[2]+2) as usize;
            assert!(columns.checked_mul(levels).unwrap()<100_000_000);
            for index in 0..2 {allocation_original[index]=breakpoint(&mem,pid,sites[index]);}
            let ret=u64_at(&mem,r.rsp);let old=breakpoint(&mem,pid,ret);
            pending=Some(Pending {ret,old,columns,levels,seen:[false;2]});
        } else if let Some(index)=sites.iter().position(|site|*site==at) {
            let call=pending.as_mut().expect("allocation trap outside CAM");
            assert!(!call.seen[index]);call.seen[index]=true;
            control(5,pid,at,allocation_original[index]);r.rip=at;set_regs(pid,&r);
            assert!(r.rax!=0);
            let raw=read(&mem,r.rax,call.columns*call.levels*8);
            compact(&mut streams[index],call.columns,call.levels,&raw);
            println!("CAM_NATIVE_ENTRY call={} field={} columns={} levels={} bytes={} pointer=0x{:x}",
                     calls+1,["tauxci","tauxcl"][index],call.columns,call.levels,raw.len(),r.rax);
        } else if let Some(call)=pending.take() {
            assert_eq!(at,call.ret,"unexpected native trap");assert_eq!(call.seen,[true;2]);
            control(5,pid,at,call.old);r.rip=at;set_regs(pid,&r);calls+=1;
            assert_eq!(breakpoint(&mem,pid,address),original);
        } else {panic!("unexpected native trap at {at:x}");}
        control(7,pid,0,0);
    }
}
fn main() {
    let args:Vec<_>=env::args().collect();assert!(args.len()>=6);
    let hex=|text:&str|u64::from_str_radix(text.trim_start_matches("0x"),16).unwrap();
    std::process::exit(run(Path::new(&args[1]),hex(&args[2]),[hex(&args[3]),hex(&args[4])],
                          Path::new(&args[5]),&args[6..]));
}
