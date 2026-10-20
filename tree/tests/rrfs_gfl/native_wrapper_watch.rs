//! Observe all pristine GFL wrapper arguments at entry and return in Rust.
//! No native rebuild or allocation hooks. All full forecast outputs must first
//! pass the observer/no-observer comparison before admitting a capture.
//! Authority: RRFS-MPAS 1d6e370e/GFL bf6a4b0, MPAS/module_cu_gfl_mpas.F90:33-745
//! and module_cu_gfl_deep.F90:85-166. Local cell IDs are unknown, explicitly -1.
#[allow(dead_code)] #[path="native_entry_watch.rs"] mod native;
use native::{trace,word,integer,argument,wait,Registers};
use std::{env,fs,io::{self,Write},os::unix::process::CommandExt,path::{Path,PathBuf},process::{Command,exit}};
use std::ffi::c_void;
include!("wrapper_spec.rs");
#[repr(C)] struct IoVec {base:*mut c_void,len:usize}
unsafe extern "C" {fn process_vm_readv(pid:i32,local:*const IoVec,nl:usize,remote:*const IoVec,nr:usize,flags:usize)->isize;}
fn bad(s:&str)->io::Error{io::Error::other(s)}
fn bytes(pid:i32,address:u64,length:usize)->io::Result<Vec<u8>>{
 let mut out=vec![0u8;length];let mut used=0;
 while used<length {
  let local=IoVec{base:out[used..].as_mut_ptr() as *mut _,len:length-used};
  let remote=IoVec{base:(address+used as u64) as *mut _,len:length-used};
  let got=unsafe{process_vm_readv(pid,&local,1,&remote,1,0)};
  if got<=0{return Err(io::Error::last_os_error())}used+=got as usize;
 }
 Ok(out)
}
struct Frame {pointers:[u64;87],dims:[i32;18],character_length:usize,step:i32,call:usize,return_ip:u64}
impl Frame {
 fn new(pid:i32,regs:&Registers,call:usize)->io::Result<Self>{
  let mut pointers=[0;87];for(i,p)in pointers.iter_mut().enumerate(){*p=argument(pid,regs,i)?;}
  let mut dims=[0;18];for(i,d)in dims.iter_mut().enumerate(){*d=integer(pid,pointers[69+i])?;}
  let character_length=argument(pid,regs,87)? as usize;
  if !(1..=16384).contains(&character_length) {return Err(bad("character ABI extent invalid"))}
  for(a,b)in [(0,1),(2,3),(4,5),(12,13),(14,15),(16,17)]{
   if dims[a]<1 || dims[b]<dims[a] || dims[b]-dims[a]>10000000{return Err(bad("native wrapper extent invalid"))}
  }
  let step=integer(pid,pointers[0])?;
  Ok(Self{pointers,dims,character_length,step,call,return_ip:word(pid,regs.stack())?})
 }
 fn prefix(&self,rank:i32)->String{format!("gfl-passive.step{:06}.call{:08}.rank{rank:04}",self.step,self.call)}
 fn shape(&self,layout:i32,chem:i32)->io::Result<(Vec<i32>,Vec<i32>,i32,i32)>{
  let d=&self.dims;let ni=d[1]-d[0]+1;let nj=d[3]-d[2]+1;let nk=d[5]-d[4]+1;
  Ok(match layout {
   0|4=>(vec![],vec![],0,0),
   1=>(vec![ni,nj],vec![d[0],d[2]],1,0),
   2=>(vec![ni,nk,nj],vec![d[0],d[4],d[2]],1,2),
   3 if chem>=0=>(vec![ni,nk,nj,chem],vec![d[0],d[4],d[2],1],1,2),
   _=>return Err(bad("native wrapper shape invalid")),
  })
 }
 fn capture(&self,pid:i32,root:&Path,rank:i32,phase:&str)->io::Result<()>{
  let chem=integer(pid,self.pointers[55])?;
  for(index,(name,code,layout))in SPEC.iter().enumerate(){
   let present=self.pointers[index]!=0;
   let prefix=self.prefix(rank);
   record(&root.join(format!("{prefix}.{phase}.{name}.present.bin")),4,4,&[],&[],0,0,
    self.step,rank,name,phase,&(present as i32).to_le_bytes())?;
   if !present{continue}
   let(shape,lower,cell,level)=self.shape(*layout,chem)?;
   let width=if *code==5{self.character_length}else{4};
   let elements=shape.iter().try_fold(1usize,|a,d|a.checked_mul(*d as usize)).ok_or_else(||bad("capture size overflow"))?;
   let length=elements.checked_mul(width).ok_or_else(||bad("capture width overflow"))?;
   if length>512*1024*1024{return Err(bad("native argument unexpectedly large"))}
   record(&root.join(format!("{prefix}.{phase}.{name}.bin")),*code,width as i32,&shape,&lower,
    cell,level,self.step,rank,name,phase,&bytes(pid,self.pointers[index],length)?)?;
  }
  Ok(())
 }
}
fn record(path:&Path,code:i32,width:i32,shape:&[i32],lower:&[i32],cell:i32,level:i32,
 step:i32,rank:i32,name:&str,phase:&str,data:&[u8])->io::Result<()>{
 let mut h=[0i32;21];h[0]=code;h[1]=width;h[2]=shape.len() as i32;
 h[3..8].fill(1);h[8..13].fill(1);h[3..3+shape.len()].copy_from_slice(shape);h[8..8+lower.len()].copy_from_slice(lower);
 h[13]=step;h[14]=0;h[15]=rank;h[16]=if cell>0{shape[cell as usize-1]}else{0};h[17]=cell;h[18]=level;h[19]=256;
 let mut file=fs::OpenOptions::new().write(true).create_new(true).open(path)?;
 file.write_all(b"ORCL0001")?;for v in h{file.write_all(&v.to_le_bytes())?;}
 for value in [name,phase,"pristine-external-wrapper-unknown-global-IDs"]{
  let mut text=[b' ';256];let source=value.as_bytes();if source.len()>256{return Err(bad("metadata text length"))}
  text[..source.len()].copy_from_slice(source);file.write_all(&text)?;
 }
 for _ in 0..h[16]{file.write_all(&(-1i32).to_le_bytes())?;}
 file.write_all(data)
}
fn arm(pid:i32,address:u64)->io::Result<u64>{let old=word(pid,address)?;trace(5,pid,address,(old&!255)|204)?;Ok(old)}
fn step(pid:i32,regs:&mut Registers,address:u64,original:u64,rearm:bool)->io::Result<()>{
 trace(5,pid,address,original)?;regs.set_ip(address);trace(13,pid,0,regs as *mut _ as u64)?;
 trace(9,pid,0,0)?;if wait(pid)?!=(5<<8|127){return Err(bad("native instruction step failed"))}
 if rearm{trace(5,pid,address,(original&!255)|204)?;}
 trace(7,pid,0,0)?;Ok(())
}
fn main()->io::Result<()>{
 let args:Vec<_>=env::args_os().collect();if args.len()!=3{return Err(bad("native_wrapper_watch EXEC OUTPUT"))}
 let executable=fs::canonicalize(&args[1])?;let output=PathBuf::from(&args[2]);fs::create_dir_all(&output)?;
 let symbols=Command::new("nm").arg(&executable).output()?;if !symbols.status.success(){return Err(bad("symbol audit failed"))}
 let text=String::from_utf8_lossy(&symbols.stdout);
 let symbol=|name:&str|->io::Result<u64>{text.lines().find_map(|line|{let v:Vec<_>=line.split_whitespace().collect();
  if v.len()==3&&v[2]==name{u64::from_str_radix(v[0],16).ok()}else{None}}).ok_or_else(||bad("native GFL symbol absent"))};
 let wrapper_offset=symbol("__module_cu_gfl_MOD_cu_grell_freitas_li")?;
 let deep_offset=symbol("__module_cu_gfl_deep_MOD_cu_gfl_deep_run")?;
 let mut command=Command::new(&executable);
 unsafe{command.pre_exec(||{trace(0,0,0,0)?;Ok(())});}
 let child=command.spawn()?;let pid=child.id() as i32;
 if wait(pid)?&255!=127{return Err(bad("no debugger initial stop"))}
 trace(0x4200,pid,0,0x100000)?; // PTRACE_O_EXITKILL: never leave our patched child running on observer failure.
 let maps=fs::read_to_string(format!("/proc/{pid}/maps"))?;
 let bias=maps.lines().find_map(|line|{let v:Vec<_>=line.split_whitespace().collect();
  if v.len()>=6&&v[5]==executable.to_string_lossy()&&v[2]=="00000000"{
   u64::from_str_radix(v[0].split('-').next().unwrap(),16).ok()}else{None}}).ok_or_else(||bad("native load map absent"))?;
 let wrapper=bias+wrapper_offset;let deep=bias+deep_offset;
 let wrapper_original=arm(pid,wrapper)?;let deep_original=arm(pid,deep)?;
 let rank:i32=env::var("PMI_RANK").unwrap_or_else(|_|"0".into()).parse().map_err(|_|bad("invalid MPI rank"))?;
 let mut events=fs::File::create(output.join(format!("rank{rank}.jsonl")))?;
 let mut calls=fs::File::create(output.join(format!("rank{rank}-calls.jsonl")))?;
 let mut active:Option<(Frame,u64)>=None;let mut call=0;let mut event=0;
 trace(7,pid,0,0)?;
 loop{
  let status=wait(pid)?;
  if status&127==0{if active.is_some(){return Err(bad("native exited inside wrapper"))}exit((status>>8)&255)}
  if status&255!=127{exit(128+(status&127))}
  let signal=(status>>8)&255;let mut regs=Registers::default();trace(12,pid,0,&mut regs as *mut _ as u64)?;
  if signal==5&&regs.ip()==wrapper+1{
   if active.is_some(){return Err(bad("unexpected recursive native wrapper"))}
   let frame=Frame::new(pid,&regs,call)?;frame.capture(pid,&output,rank,"pre")?;
   let ret_original=arm(pid,frame.return_ip)?;
   writeln!(calls,"{{\"call\":{call},\"step\":{},\"rank\":{rank},\"phase\":\"pre\",\"prefix\":\"{}\",\"global_cell_ids_known\":false,\"hidden_state_complete\":false,\"character_length\":{}}}",frame.step,frame.prefix(rank),frame.character_length)?;
   calls.flush()?;active=Some((frame,ret_original));step(pid,&mut regs,wrapper,wrapper_original,true)?;
  }else if signal==5&&regs.ip()==deep+1{
   let itf=integer(pid,argument(pid,&regs,0)?)?;let its=integer(pid,argument(pid,&regs,2)?)?;
   let ite=integer(pid,argument(pid,&regs,3)?)?;let imid=integer(pid,argument(pid,&regs,13)?)?;
   if its<1||ite<its||ite-its>10000000||![0,1].contains(&imid){return Err(bad("deep ABI bounds"))}
   let data=bytes(pid,argument(pid,&regs,57)?,4*(ite-its+1) as usize)?;
   let file=format!("rank{rank}-event{event:04}-imid{imid}-ierr.raw");fs::write(output.join(&file),&data)?;
   let nonzero=data.chunks_exact(4).filter(|w|i32::from_le_bytes((*w).try_into().unwrap())!=0).count();
   writeln!(events,"{{\"event\":{event},\"rank\":{rank},\"imid\":{imid},\"its\":{its},\"ite\":{ite},\"itf\":{itf},\"word_width\":4,\"nonzero_words\":{nonzero},\"file\":\"{file}\"}}")?;events.flush()?;
   if let Some((frame,_))=&active{
    let field=if imid==1{"local.module_cu_gfl.ierrm"}else{"local.module_cu_gfl.ierr"};
    record(&output.join(format!("{}.pre.{field}.bin",frame.prefix(rank))),3,4,&[ite-its+1],&[its],1,0,frame.step,rank,field,"pre",&data)?;
   }else{return Err(bad("deep entry outside captured wrapper"))}
   event+=1;step(pid,&mut regs,deep,deep_original,true)?;
  }else if signal==5&&active.as_ref().is_some_and(|(f,_)|regs.ip()==f.return_ip+1){
   let(frame,original)=active.take().unwrap();frame.capture(pid,&output,rank,"post")?;
   writeln!(calls,"{{\"call\":{call},\"step\":{},\"rank\":{rank},\"phase\":\"post\",\"prefix\":\"{}\"}}",frame.step,frame.prefix(rank))?;calls.flush()?;
   call+=1;step(pid,&mut regs,frame.return_ip,original,false)?;
  }else{trace(7,pid,0,signal as u64)?;}
 }
}
