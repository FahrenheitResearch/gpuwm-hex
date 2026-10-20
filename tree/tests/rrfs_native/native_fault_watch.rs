//! Observe the original child's SIGSEGV registers without changing code/data.
//! No breakpoints, numerical memory writes, or register writes. Every native
//! signal is forwarded unchanged. The exec stop only admits the observer.
//! Used with MPAS-Model-rrfs/src/core_atmosphere/dynamics/
//! mpas_atm_time_integration.F:3496,3574,3783-3834 at 1d6e370e.
//! Register/read mechanism follows r1/r12's read-only native observer ABI.
use std::{env, ffi::c_void, fs, io, os::unix::process::CommandExt,
          path::Path, process::{Command, exit}};

unsafe extern "C" {
    fn ptrace(request: i32, pid: i32, address: *mut c_void, data: *mut c_void) -> i64;
    fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
    fn __errno_location() -> *mut i32;
}
#[repr(C)]
#[derive(Default)]
struct Registers {
    r15:u64,r14:u64,r13:u64,r12:u64,rbp:u64,rbx:u64,r11:u64,r10:u64,
    r9:u64,r8:u64,rax:u64,rcx:u64,rdx:u64,rsi:u64,rdi:u64,orig_rax:u64,
    rip:u64,cs:u64,eflags:u64,rsp:u64,ss:u64,fs_base:u64,gs_base:u64,
    ds:u64,es:u64,fs:u64,gs:u64,
}
fn trace(request:i32,pid:i32,address:u64,data:u64)->io::Result<i64> {
    unsafe {
        *__errno_location()=0;
        let value=ptrace(request,pid,address as *mut _,data as *mut _);
        if value == -1 && *__errno_location()!=0 {Err(io::Error::last_os_error())}
        else {Ok(value)}
    }
}
fn wait(pid:i32)->io::Result<i32> {
    let mut status=0;
    if unsafe {waitpid(pid,&mut status,0)} != pid {return Err(io::Error::last_os_error())}
    Ok(status)
}
fn word(pid:i32,address:u64)->String {
    match trace(2,pid,address,0) {
        Ok(value)=>format!("\"0x{:016x}\"",value as u64),
        Err(_)=>"null".into(),
    }
}
fn observe(executable:&Path,output:&Path,descriptor_offset:u64)->io::Result<()> {
    let executable=fs::canonicalize(executable)?;
    fs::create_dir_all(output)?;
    let mut command=Command::new(&executable);
    unsafe {command.pre_exec(||{trace(0,0,0,0)?;Ok(())});}
    let child=command.spawn()?;
    let pid=child.id() as i32;
    if wait(pid)? != (5<<8|127) {return Err(io::Error::other("missing exec stop"))}
    trace(0x4200,pid,0,0x100000)?; // EXITKILL only; no code breakpoints.
    let maps=fs::read_to_string(format!("/proc/{pid}/maps"))?;
    let bias=maps.lines().find_map(|line| {
        let fields=line.split_whitespace().collect::<Vec<_>>();
        if fields.len()>=6 && fields[5]==executable.to_string_lossy() && fields[2]=="00000000" {
            u64::from_str_radix(fields[0].split('-').next()?,16).ok()
        } else {None}
    }).ok_or_else(||io::Error::other("native ELF load bias missing"))?;
    let rank=env::var("PMI_RANK").unwrap_or_else(|_|"0".into());
    if !rank.bytes().all(|b|b.is_ascii_digit()) {return Err(io::Error::other("invalid MPI rank"))}
    fs::write(output.join(format!("rank-{rank}.maps")),maps)?;
    trace(7,pid,0,0)?;
    let mut faults=0;
    loop {
        let status=wait(pid)?;
        if status&127==0 {exit((status>>8)&255)}
        if status&255!=127 {exit(128+(status&127))}
        let signal=(status>>8)&255;
        if signal==11 {
            let mut registers=Registers::default();
            trace(12,pid,0,&mut registers as *mut _ as u64)?;
            let stack_slot=registers.rsp.checked_add(descriptor_offset)
                .ok_or_else(||io::Error::other("stack offset overflow"))?;
            let descriptor=trace(2,pid,stack_slot,0).ok().map(|v|v as u64);
            let descriptor_words=(0..8).map(|i|descriptor.map(|d|word(pid,d+8*i))
                .unwrap_or_else(||"null".into())).collect::<Vec<_>>().join(",");
            let names=[("rip",registers.rip),("rsp",registers.rsp),
                ("r11",registers.r11),("rdi",registers.rdi),("rsi",registers.rsi),
                ("rcx",registers.rcx),("rdx",registers.rdx),("rax",registers.rax)];
            let values=names.iter().map(|(name,value)|format!("\"{name}\":\"0x{value:016x}\""))
                .collect::<Vec<_>>().join(",");
            let data=format!("{{\"schema\":\"rrfs-original-native-fault/v1\",\"rank\":{rank},\"signal\":{signal},\"load_bias\":\"0x{bias:016x}\",\"elf_rip\":\"0x{:x}\",\"registers\":{{{values}}},\"descriptor_stack_offset\":{descriptor_offset},\"descriptor\":{},\"descriptor_words_u64\":[{descriptor_words}]}}\n",
                registers.rip.wrapping_sub(bias),descriptor.map(|d|format!("\"0x{d:016x}\""))
                    .unwrap_or_else(||"null".into()));
            fs::write(output.join(format!("rank-{rank}-fault-{faults}.json")),data)?;
            faults+=1;
        }
        trace(7,pid,0,signal as u64)?; // Preserve original signal/Fortran handler.
    }
}
fn main()->io::Result<()> {
    let args=env::args().collect::<Vec<_>>();
    if args.len()!=4 {return Err(io::Error::other("native_fault_watch EXECUTABLE OUTPUT_DIRECTORY DESCRIPTOR_STACK_OFFSET_HEX"))}
    let offset=u64::from_str_radix(args[3].trim_start_matches("0x"),16)
        .map_err(|_|io::Error::other("invalid descriptor offset"))?;
    observe(Path::new(&args[1]),Path::new(&args[2]),offset)
}
