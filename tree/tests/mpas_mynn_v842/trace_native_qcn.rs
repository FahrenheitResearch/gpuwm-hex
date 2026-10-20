//! Read QCn in the unchanged strict executable using parent/child ptrace.
//! Diagnostic only: no writes to weather data, no acceptance or exclusions.
//! The two breakpoint bytes are restored before their original instructions run.
//! MPAS physics_mmm/bl_mynn_subroutines.F90:5941,6166,5815.
use std::{ffi::CString, fs, io::Write, os::raw::{c_int,c_long,c_void}, path::PathBuf, process::Command};

unsafe extern "C" {
    fn fork() -> c_int;
    fn execv(path: *const i8, argv: *const *const i8) -> c_int;
    fn ptrace(request: c_int, pid: c_int, addr: *mut c_void, data: *mut c_void) -> c_long;
    fn waitpid(pid: c_int, status: *mut c_int, options: c_int) -> c_int;
    fn _exit(status: c_int) -> !;
    fn __errno_location() -> *mut c_int;
}
#[repr(C)]
#[derive(Default)]
struct Regs { words: [u64;27] }
unsafe fn trace(request:c_int,pid:c_int,addr:u64,data:u64)->Result<i64,String> {
    unsafe { *__errno_location()=0; }
    let result=unsafe { ptrace(request,pid,addr as *mut c_void,data as *mut c_void) };
    let errno=unsafe { *__errno_location() };
    if result == -1 && errno != 0 {Err(format!("ptrace {request}: {}",std::io::Error::from_raw_os_error(errno)))}else{Ok(result)}
}
fn wait(pid:i32)->Result<i32,String> {
    let mut status=0;
    if unsafe {waitpid(pid,&mut status,0)}!=pid {return Err(std::io::Error::last_os_error().to_string())} Ok(status)
}
fn symbol(exe:&PathBuf,name:&str)->Result<(u64,u64),String> {
    let out=Command::new("nm").args(["-S","--defined-only"]).arg(exe).output().map_err(|e|e.to_string())?;
    if !out.status.success(){return Err("nm failed".into())}
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let fields:Vec<_>=line.split_whitespace().collect();
        if fields.len()==4&&fields[3]==name {return Ok((u64::from_str_radix(fields[0],16).map_err(|e|e.to_string())?,u64::from_str_radix(fields[1],16).map_err(|e|e.to_string())?))}
    } Err(format!("missing exact symbol {name}"))
}
fn run()->Result<i32,String> {
    let args:Vec<_>=std::env::args().skip(1).collect();
    if args.len()<2 {return Err("usage: trace_native_qcn EXE LOG_PREFIX [EXE_ARGS...]".into())}
    let exe=fs::canonicalize(&args[0]).map_err(|e|e.to_string())?;
    let dd=symbol(&exe,"__bl_mynn_subroutines_MOD_ddmf_jpl")?;
    let cond=symbol(&exe,"__bl_mynn_subroutines_MOD_condensation_edmf.isra.0")?;
    let exec_name=CString::new(exe.to_str().ok_or("executable path not UTF8")?).map_err(|e|e.to_string())?;
    let cargs:Vec<CString>=std::iter::once(exe.to_str().unwrap()).chain(args[2..].iter().map(String::as_str)).map(|s|CString::new(s).unwrap()).collect();
    let mut argv:Vec<*const i8>=cargs.iter().map(|s|s.as_ptr()).collect();argv.push(std::ptr::null());
    let pid=unsafe {fork()};
    if pid<0{return Err(std::io::Error::last_os_error().to_string())}
    if pid==0 {unsafe {if trace(0,0,0,0).is_err(){_exit(126)} execv(exec_name.as_ptr(),argv.as_ptr());_exit(127)}}
    let initial=wait(pid)?;
    if initial&0xff!=0x7f{return Err(format!("child did not stop after exec: {initial}"))}
    unsafe {trace(0x4200,pid,0,0x100000)?;} // EXITKILL: never leave our own stopped child behind.
    let maps=fs::read_to_string(format!("/proc/{pid}/maps")).map_err(|e|e.to_string())?;
    let base=maps.lines().find_map(|line|{
        let f:Vec<_>=line.split_whitespace().collect();
        if f.len()>=6&&f[2]=="00000000"&&f[5]==exe.to_str().unwrap(){u64::from_str_radix(f[0].split('-').next()?,16).ok()}else{None}
    }).ok_or("executable base mapping absent")?;
    let points=[base+dd.0,base+cond.0];
    let mut originals=[0i64;2];
    for (i,addr) in points.iter().enumerate(){unsafe {originals[i]=trace(1,pid,*addr,0)?;trace(4,pid,*addr,((originals[i] as u64)&!255)|0xcc)?;}}
    let mut log=fs::File::create(format!("{}.{}.words",args[1],pid)).map_err(|e|e.to_string())?;
    writeln!(log,"# unchanged_executable={} dd_offset={:x} condensation_offset={:x}",exe.display(),dd.0,cond.0).map_err(|e|e.to_string())?;
    let mut dd_call=0u64;let mut condensation_call=0u64;
    unsafe {trace(7,pid,0,0)?;}
    loop {
        let status=wait(pid)?;
        if status&0x7f==0{return Ok((status>>8)&255)}
        if status&0xff!=0x7f{return Err(format!("child terminated: status {status}"))}
        let signal=(status>>8)&255;
        if signal!=5 {unsafe {trace(7,pid,0,signal as u64)?;}continue}
        let mut regs=Regs::default();unsafe {trace(12,pid,0,(&mut regs as *mut Regs)as u64)?;}
        let address=regs.words[16]-1;
        let index=points.iter().position(|x|*x==address).ok_or(format!("unexpected SIGTRAP at {address:x}"))?;
        if index==0 {dd_call+=1;condensation_call=0;} else {
            let caller=unsafe {trace(1,pid,regs.words[19],0)?}as u64;
            if caller>=points[0]&&caller<points[0]+dd.1 {
                condensation_call+=1;
                let qc=unsafe {trace(1,pid,regs.words[12],0)?}as u32;
                writeln!(log,"QCn {dd_call} {condensation_call} {qc:08X}").map_err(|e|e.to_string())?;
            }
        }
        regs.words[16]=address;
        unsafe {trace(4,pid,address,originals[index]as u64)?;trace(13,pid,0,(&mut regs as *mut Regs)as u64)?;trace(9,pid,0,0)?;}
        let stepped=wait(pid)?;if stepped&0xff!=0x7f||(stepped>>8)&255!=5{return Err(format!("single step failed: {stepped}"))}
        unsafe {trace(4,pid,address,((originals[index]as u64)&!255)|0xcc)?;trace(7,pid,0,0)?;}
    }
}
fn main(){match run(){Ok(code)=>std::process::exit(code),Err(e)=>{eprintln!("{e}");std::process::exit(125)}}}
