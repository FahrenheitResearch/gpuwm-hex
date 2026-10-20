//! Native GNU x86-64 explicit-shape argument observer. Source-derived spec is
//! emitted by the shared generate.py; data decode/encode stays in Rust/Fortran.
//! Debugger stops require a complete observer/ordinary forecast word comparison.
//! No numerical native source is rebuilt. Unknown global cell IDs remain -1.
#[allow(dead_code)] #[path="native_entry_watch.rs"] mod native;
use native::{trace,word,integer,argument,wait,Registers};
use std::{env,fs,io::{self,Write},os::unix::process::CommandExt,path::{Path,PathBuf},process::{Command,exit}};
use std::ffi::c_void;
struct Arg {name:&'static str,code:i32,width:usize,cell:i32,level:i32,optional:bool,descriptor:bool}
struct Probe {symbol:&'static str,nargs:usize,argument:usize,selector:Option<usize>,labels:&'static [(i32,&'static str)],code:i32,width:usize,cell:i32,level:i32}
include!("argument_spec.rs");
#[repr(C)] struct IoVec {base:*mut c_void,len:usize}
unsafe extern "C" {fn process_vm_readv(pid:i32,local:*const IoVec,nl:usize,remote:*const IoVec,nr:usize,flags:usize)->isize;}
fn bad(s:&str)->io::Error{io::Error::other(s)}
fn scalar(pid:i32,p:&[u64],i:usize,width:usize)->io::Result<i64>{
 let address=*p.get(i).ok_or_else(||bad("bound argument ordinal"))?;
 if address==0{return Err(bad("missing bound scalar"))}
 if width==4{Ok(integer(pid,address)? as i64)}else if width==8{Ok(word(pid,address)? as i64)}else{Err(bad("scalar ABI width"))}
}
fn layout(bounds:&[(i64,i64)])->io::Result<(Vec<i32>,Vec<i32>)>{
 let mut shape=Vec::new();let mut lower=Vec::new();
 for &(lo,hi) in bounds{
  let extent=hi.checked_sub(lo).and_then(|v|v.checked_add(1)).ok_or_else(||bad("bound overflow"))?.max(0);
  shape.push(i32::try_from(extent).map_err(|_|bad("extent overflow"))?);
  lower.push(i32::try_from(lo).map_err(|_|bad("lower bound overflow"))?);
 }
 if shape.len()>5{return Err(bad("record rank exceeds native format"))}Ok((shape,lower))
}
fn length(shape:&[i32],width:usize)->io::Result<usize>{
 let n=shape.iter().try_fold(width,|n,d|n.checked_mul(*d as usize)).ok_or_else(||bad("argument extent overflow"))?;
 if n>512*1024*1024{return Err(bad("argument extent exceeds bounded observer"))}Ok(n)
}
fn bytes(pid:i32,address:u64,length:usize)->io::Result<Vec<u8>>{
 let mut out=vec![0u8;length];let mut used=0;
 while used<length {
  let local=IoVec{base:out[used..].as_mut_ptr() as *mut _,len:length-used};
  let remote=IoVec{base:(address+used as u64) as *mut _,len:length-used};
  let got=unsafe{process_vm_readv(pid,&local,1,&remote,1,0)};
  if got<=0{return Err(io::Error::last_os_error())}used+=got as usize;
 }Ok(out)
}
// GNU's x86-64 internal descriptor: data/offset/dtype/span, then stride/lower/upper
// triples. The compiler ABI fixture checks actual metadata and strided payloads.
// This is not the ISO_Fortran_binding C descriptor.
fn descriptor(pid:i32,address:u64,rank:usize,code:i32,width:usize)->io::Result<(u64,Vec<i64>,Vec<i64>)>{
 if rank>5{return Err(bad("descriptor rank exceeds native record format"))}
 let raw=bytes(pid,address,40+24*rank)?;
 let u64at=|i|u64::from_le_bytes(raw[i..i+8].try_into().unwrap());
 let native_type=match code{1|2=>3,3|6=>1,4=>2,5=>6,7|8=>4,_=>return Err(bad("unsupported descriptor element type"))};
 if u64at(16)!=width as u64 || raw[28] as usize!=rank || raw[29]!=native_type || u64at(32)!=width as u64 {
  return Err(bad("GNU descriptor element type, rank or span differs from source"))
 }
 let mut extents=Vec::new();let mut strides=Vec::new();
 for i in 0..rank {
  let stride=u64at(40+24*i) as i64;let lo=u64at(48+24*i) as i64;let hi=u64at(56+24*i) as i64;
  extents.push(hi.checked_sub(lo).and_then(|v|v.checked_add(1)).ok_or_else(||bad("descriptor extent overflow"))?.max(0));
  strides.push(stride.checked_mul(width as i64).ok_or_else(||bad("descriptor stride overflow"))?);
 }Ok((u64at(0),extents,strides))
}
fn descriptor_layout(pid:i32,address:u64,lower:&[i64],code:i32,width:usize)->io::Result<(Vec<i32>,Vec<i32>)>{
 let (_,extents,_)=descriptor(pid,address,lower.len(),code,width)?;
 layout(&lower.iter().zip(extents).map(|(&lo,n)|Ok((lo,lo.checked_add(n).and_then(|v|v.checked_sub(1)).ok_or_else(||bad("descriptor bound overflow"))?)))
  .collect::<io::Result<Vec<_>>>()?)
}
fn array_bytes(pid:i32,address:u64,shape:&[i32],strides:&[i64],width:usize)->io::Result<Vec<u8>>{
 let n=length(shape,width)?;let mut expected=width as i64;let mut contiguous=true;
 for (&extent,&stride) in shape.iter().zip(strides){
  if extent>1&&stride!=expected{contiguous=false}
  expected=expected.checked_mul(extent as i64).ok_or_else(||bad("array stride overflow"))?;
 }
 if contiguous{return bytes(pid,address,n)}
 let mut result=Vec::with_capacity(n);
 // Retain the native dummy's Fortran element order, including reversed sections.
 let block=if strides.first()==Some(&(width as i64)){shape[0] as usize}else{1};
 let block_bytes=block.checked_mul(width).ok_or_else(||bad("array block overflow"))?;
 if block_bytes==0{return Ok(result)}
 for index in (0..n/width).step_by(block){
  let mut remaining=index;let mut offset=0i128;
  for (&extent,&stride) in shape.iter().zip(strides){
   let coordinate=remaining%(extent as usize);remaining/=extent as usize;
   offset+=coordinate as i128*stride as i128;
  }
  let location=u64::try_from(address as i128+offset).map_err(|_|bad("array address overflow"))?;
  result.extend(bytes(pid,location,block_bytes)?);
 }Ok(result)
}
fn record(path:&Path,code:i32,width:usize,shape:&[i32],lower:&[i32],cell:i32,level:i32,
 step:i32,event:i32,rank:i32,name:&str,phase:&str,data:&[u8])->io::Result<()>{
 let mut h=[0i32;21];h[0]=code;h[1]=width as i32;h[2]=shape.len() as i32;
 h[3..8].fill(1);h[8..13].fill(1);h[3..3+shape.len()].copy_from_slice(shape);h[8..8+lower.len()].copy_from_slice(lower);
 if code==5&&shape.is_empty(){h[3]=data.len() as i32;h[1]=1;}
 h[13]=step;h[14]=event;h[15]=rank;h[16]=if cell>0{shape[cell as usize-1]}else{0};h[17]=cell;h[18]=level;h[19]=256;
 let mut file=fs::OpenOptions::new().write(true).create_new(true).open(path)?;
 file.write_all(b"ORCL0001")?;for v in h{file.write_all(&v.to_le_bytes())?;}
 for value in [SYMBOL,phase,name]{
  let mut text=[b' ';256];let source=value.as_bytes();if source.len()>256{return Err(bad("metadata text extent"))}
  text[..source.len()].copy_from_slice(source);file.write_all(&text)?;
 }
 for _ in 0..h[16]{file.write_all(&(-1i32).to_le_bytes())?;}file.write_all(data)
}
struct Frame {pointers:Vec<u64>,locations:Vec<u64>,strides:Vec<Vec<i64>>,shapes:Vec<Vec<i32>>,lower:Vec<Vec<i32>>,widths:Vec<usize>,step:i32,call:i32,return_ip:u64}
impl Frame {
 fn new(pid:i32,regs:&Registers,call:i32)->io::Result<Self>{
  let pointers=(0..NARGS).map(|i|argument(pid,regs,i)).collect::<io::Result<Vec<_>>>()?;
  let mut shapes=Vec::new();let mut lower=Vec::new();let mut locations=Vec::new();let mut strides=Vec::new();let mut widths=ARGS.iter().map(|a|a.width).collect::<Vec<_>>();
  for &(i,slot) in CHARACTERS {
   let len=argument(pid,regs,slot)? as usize;
   if len==0||len>16384{return Err(bad("character ABI extent invalid"))}widths[i]=len;
  }
  for (i,a) in ARGS.iter().enumerate(){
   if pointers[i]==0&&!a.optional{return Err(bad("required argument absent"))}
   let (s,l)=if pointers[i]==0 {(Vec::new(),Vec::new())}else{argument_layout(pid,&pointers,i,widths[i])?};
   length(&s,widths[i])?;
   let (data,stride)=if a.descriptor&&pointers[i]!=0 {
    let(data,_,stride)=descriptor(pid,pointers[i],s.len(),a.code,widths[i])?;(data,stride)
   }else{let mut stride=Vec::new();let mut n=widths[i] as i64;for &extent in &s{stride.push(n);n=n.checked_mul(extent as i64).ok_or_else(||bad("explicit stride overflow"))?;}(pointers[i],stride)};
   locations.push(data);strides.push(stride);shapes.push(s);lower.push(l);
  }
  let step=ARGS.iter().position(|a|a.name=="itimestep").map(|i|integer(pid,pointers[i])).transpose()?.unwrap_or(call);
  Ok(Self{pointers,locations,strides,shapes,lower,widths,step,call,return_ip:word(pid,regs.stack())?})
 }
 fn prefix(&self,rank:i32)->String{format!("external.step{:06}.call{:08}.rank{rank:04}",self.step,self.call)}
 fn capture(&self,pid:i32,root:&Path,rank:i32,phase:&str)->io::Result<()>{
  for (i,a) in ARGS.iter().enumerate(){
   let present=self.pointers[i]!=0;let prefix=self.prefix(rank);
   record(&root.join(format!("{prefix}.{phase}.{}.present.bin",a.name)),4,4,&[],&[],0,0,
    self.step,self.call,rank,&format!("{}.present",a.name),phase,&(present as i32).to_le_bytes())?;
   if !present{continue}
   record(&root.join(format!("{prefix}.{phase}.{}.bin",a.name)),a.code,self.widths[i],&self.shapes[i],&self.lower[i],
    a.cell,a.level,self.step,self.call,rank,a.name,phase,&array_bytes(pid,self.locations[i],&self.shapes[i],&self.strides[i],self.widths[i])?)?;
  }Ok(())
 }
 fn aliases(&self)->String{
  let mut pairs=Vec::new();
  for i in 0..ARGS.len(){
   if self.pointers[i]==0||self.shapes[i].is_empty(){continue}
   let members=(0..ARGS.len()).filter(|&j|self.locations[i]==self.locations[j]&&self.strides[i]==self.strides[j]&&self.shapes[i]==self.shapes[j]
     &&self.lower[i]==self.lower[j]&&self.widths[i]==self.widths[j]&&ARGS[i].code==ARGS[j].code).collect::<Vec<_>>();
   let target=members.iter().find(|&&j|!ARGS[j].optional).copied().unwrap_or(members[0]);
   if i!=target{pairs.push(format!("\"{}\":\"{}\"",ARGS[i].name,ARGS[target].name));}
  }format!("{{{}}}",pairs.join(","))
 }
}
fn arm(pid:i32,address:u64)->io::Result<u64>{let old=word(pid,address)?;trace(5,pid,address,(old&!255)|204)?;Ok(old)}
fn resume(pid:i32,regs:&mut Registers,address:u64,original:u64,rearm:bool)->io::Result<()>{
 trace(5,pid,address,original)?;regs.set_ip(address);trace(13,pid,0,regs as *mut _ as u64)?;
 trace(9,pid,0,0)?;if wait(pid)?!=(5<<8|127){return Err(bad("native instruction step failed"))}
 if rearm{trace(5,pid,address,(original&!255)|204)?;}trace(7,pid,0,0)?;Ok(())
}
fn main()->io::Result<()>{
 let args=env::args_os().collect::<Vec<_>>();if args.len()<3||(args.len()>3&&args[3]!="--"){return Err(bad("observer EXECUTABLE OUTPUT [-- PROGRAM_ARGUMENTS]"))}
 let executable=fs::canonicalize(&args[1])?;let output=PathBuf::from(&args[2]);fs::create_dir_all(&output)?;
 let symbols=Command::new("nm").arg(&executable).output()?;if !symbols.status.success(){return Err(bad("symbol audit failed"))}
 let text=String::from_utf8_lossy(&symbols.stdout);
 let symbol=|name:&str|->io::Result<u64>{text.lines().find_map(|line|{let v=line.split_whitespace().collect::<Vec<_>>();
  if v.len()==3&&v[2]==name{u64::from_str_radix(v[0],16).ok()}else{None}}).ok_or_else(||bad("source-derived native symbol absent"))};
 let offset=symbol(SYMBOL)?;let probes=PROBES.iter().map(|p|symbol(p.symbol)).collect::<io::Result<Vec<_>>>()?;
 let mut command=Command::new(&executable);if args.len()>3{command.args(&args[4..]);}unsafe{command.pre_exec(||{trace(0,0,0,0)?;Ok(())});}
 let child=command.spawn()?;let pid=child.id() as i32;
 if wait(pid)?&255!=127{return Err(bad("no initial debugger stop"))}trace(0x4200,pid,0,0x100000)?;
 let maps=fs::read_to_string(format!("/proc/{pid}/maps"))?;
 let bias=maps.lines().find_map(|line|{let v=line.split_whitespace().collect::<Vec<_>>();
  if v.len()>=6&&v[5]==executable.to_string_lossy()&&v[2]=="00000000"{
   u64::from_str_radix(v[0].split('-').next().unwrap(),16).ok()}else{None}}).ok_or_else(||bad("native load map absent"))?;
 let address=bias+offset;let original=arm(pid,address)?;
 let probe_traps=probes.iter().map(|a|Ok((bias+a,arm(pid,bias+a)?))).collect::<io::Result<Vec<_>>>()?;
 let rank:i32=env::var("PMI_RANK").unwrap_or_else(|_|"0".into()).parse().map_err(|_|bad("MPI rank invalid"))?;
 let mut calls=fs::File::create(output.join(format!("rank{rank}-calls.jsonl")))?;
 let mut active:Option<(Frame,u64)>=None;let mut call=0;
 trace(7,pid,0,0)?;
 loop{
  let status=wait(pid)?;
  if status&127==0 {if active.is_some(){return Err(bad("child exits inside captured routine"))}exit((status>>8)&255)}
  if status&255!=127{exit(128+(status&127))}
  let signal=(status>>8)&255;let mut regs=Registers::default();trace(12,pid,0,&mut regs as *mut _ as u64)?;
  if signal==5&&regs.ip()==address+1{
   if active.is_some(){return Err(bad("recursive observed routine needs nested frame support"))}
   let frame=Frame::new(pid,&regs,call)?;frame.capture(pid,&output,rank,"pre")?;let ret=arm(pid,frame.return_ip)?;
   writeln!(calls,"{{\"call\":{call},\"step\":{},\"rank\":{rank},\"phase\":\"pre\",\"prefix\":\"{}\",\"argument_aliases\":{},\"global_cell_ids_known\":false,\"hidden_state_complete\":false}}",frame.step,frame.prefix(rank),frame.aliases())?;calls.flush()?;
   active=Some((frame,ret));resume(pid,&mut regs,address,original,true)?;
  }else if signal==5&&probe_traps.iter().any(|(a,_)|regs.ip()==a+1){
   let i=probe_traps.iter().position(|(a,_)|regs.ip()==a+1).unwrap();let probe=&PROBES[i];
   let pointers=(0..probe.nargs).map(|j|argument(pid,&regs,j)).collect::<io::Result<Vec<_>>>()?;
   let selector=probe.selector.map(|j|integer(pid,pointers[j])).transpose()?.unwrap_or(0);
   let name=probe.labels.iter().find(|(v,_)|*v==selector).map(|(_,n)|*n).ok_or_else(||bad("native hidden call selector absent"))?;
   let (shape,lower)=probe_layout(pid,&pointers,i)?;
   let frame=&active.as_ref().ok_or_else(||bad("hidden probe outside observed routine"))?.0;
   let n=length(&shape,probe.width)?;
   record(&output.join(format!("{}.pre.{name}.bin",frame.prefix(rank))),probe.code,probe.width,&shape,&lower,probe.cell,probe.level,
    frame.step,frame.call,rank,name,"pre",&bytes(pid,pointers[probe.argument],n)?)?;
   let(a,o)=probe_traps[i];resume(pid,&mut regs,a,o,true)?;
  }else if signal==5&&active.as_ref().is_some_and(|(f,_)|regs.ip()==f.return_ip+1){
   let(frame,ret)=active.take().unwrap();frame.capture(pid,&output,rank,"post")?;
   writeln!(calls,"{{\"call\":{call},\"step\":{},\"rank\":{rank},\"phase\":\"post\",\"prefix\":\"{}\"}}",frame.step,frame.prefix(rank))?;calls.flush()?;
   call+=1;resume(pid,&mut regs,frame.return_ip,ret,false)?;
  }else{trace(7,pid,0,signal as u64)?;}
 }
}
