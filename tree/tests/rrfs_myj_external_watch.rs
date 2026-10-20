//! Read-only external observation of both pristine native MYJ modules.
//! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_bl_myjpbl.F:141-782
//! and module_sf_myjsfc.F:76-366 at 1d6e370e. ABI argument order follows the
//! SUBROUTINE signatures, not the driver keyword order. Only breakpoint bytes
//! change; argument and module SAVE storage is never written. Observer mechanics
//! derive from r4 rrfs_driver_external_watch.rs; MPI1 only.
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
#[derive(Clone)] struct Array{at:u64,code:u32,width:usize,shape:Vec<usize>,lower:Vec<i64>,stride:Vec<i64>}
impl Array{
 fn scalar(at:u64,code:u32,width:usize)->Self{Self{at,code,width,shape:vec![],lower:vec![],stride:vec![]}}
 fn read(&self,p:i32)->io::Result<Vec<u8>>{let n=self.shape.iter().try_fold(self.width,|n,&d|n.checked_mul(d)).ok_or_else(||bad("extent overflow"))?;let mut expected=self.width as i64;let mut contiguous=true;for(&s,&st)in self.shape.iter().zip(&self.stride){if s>1&&expected!=st{contiguous=false}expected*=s as i64;}if contiguous{return bytes(p,self.at,n)}let mut v=Vec::with_capacity(n);for i in 0..n/self.width{let mut j=i;let mut off=0i64;for(&s,&st)in self.shape.iter().zip(&self.stride){off+=(j%s)as i64*st;j/=s;}v.extend(bytes(p,(self.at as i128+off as i128).try_into().map_err(|_|bad("strided address overflow"))?,self.width)?);}Ok(v)}
 fn save(&self,p:i32,path:&Path,name:&str,phase:&str,event:i32,site:&str)->io::Result<()>{
  let payload=self.read(p)?;let mut f=fs::OpenOptions::new().create_new(true).write(true).open(path)?;
  f.write_all(b"ORCL0001")?;let mut h=[0i32;21];h[0]=self.code as i32;h[1]=self.width as i32;h[2]=self.shape.len()as i32;
  h[3..8].fill(1);h[8..13].fill(1);for(i,&v)in self.shape.iter().enumerate(){h[3+i]=i32::try_from(v).map_err(|_|bad("shape exceeds record format"))?;h[8+i]=i32::try_from(self.lower[i]).map_err(|_|bad("lower bound exceeds record format"))?;}h[13]=-1;h[14]=event;h[19]=256;
  for v in h{f.write_all(&v.to_le_bytes())?;}for s in [site,phase,name]{if s.len()>256{return Err(bad("record name overflow"))}let mut text=[b' ';256];text[..s.len()].copy_from_slice(s.as_bytes());f.write_all(&text)?;}f.write_all(&payload)
 }
}


#[derive(Clone)] enum Kind { Entry(usize), Done }
struct Trap { byte:u8, kind:Kind }
fn setbyte(p:i32,a:u64,b:u8)->io::Result<()> { let w=word(p,a)?;trace(5,p,a,(w&!255)|b as u64)?;Ok(()) }
fn arm(p:i32,t:&mut BTreeMap<u64,Trap>,a:u64,k:Kind)->io::Result<()> { if t.contains_key(&a){return Err(bad("overlapping observer trap"))} let b=word(p,a)?as u8;setbyte(p,a,204)?;t.insert(a,Trap{byte:b,kind:k});Ok(()) }
fn step(p:i32,r:&mut Registers,a:u64,b:u8,rearm:bool)->io::Result<()> { setbyte(p,a,b)?;r.rip=a;trace(13,p,0,r as *mut _ as u64)?;trace(9,p,0,0)?;if wait(p)?!=(5<<8|127){return Err(bad("native single step failed"))}if rearm{setbyte(p,a,204)?}Ok(()) }
struct Arg { name:&'static str, rank:usize, code:u32, optional:bool }
const MYJPBL: &[Arg] = &[
Arg{name:"dt",rank:0,code:1,optional:false},
Arg{name:"stepbl",rank:0,code:3,optional:false},
Arg{name:"ht",rank:2,code:1,optional:false},
Arg{name:"dz",rank:3,code:1,optional:false},
Arg{name:"pmid",rank:3,code:1,optional:false},
Arg{name:"pint",rank:3,code:1,optional:false},
Arg{name:"th",rank:3,code:1,optional:false},
Arg{name:"t",rank:3,code:1,optional:false},
Arg{name:"exner",rank:3,code:1,optional:false},
Arg{name:"qv",rank:3,code:1,optional:true},
Arg{name:"qcw",rank:3,code:1,optional:true},
Arg{name:"qci",rank:3,code:1,optional:true},
Arg{name:"qcs",rank:3,code:1,optional:true},
Arg{name:"qcr",rank:3,code:1,optional:true},
Arg{name:"qcg",rank:3,code:1,optional:true},
Arg{name:"u",rank:3,code:1,optional:false},
Arg{name:"v",rank:3,code:1,optional:false},
Arg{name:"rho",rank:3,code:1,optional:false},
Arg{name:"tsk",rank:2,code:1,optional:false},
Arg{name:"qsfc",rank:2,code:1,optional:false},
Arg{name:"chklowq",rank:2,code:1,optional:false},
Arg{name:"thz0",rank:2,code:1,optional:false},
Arg{name:"qz0",rank:2,code:1,optional:false},
Arg{name:"uz0",rank:2,code:1,optional:false},
Arg{name:"vz0",rank:2,code:1,optional:false},
Arg{name:"lowlyr",rank:2,code:3,optional:false},
Arg{name:"xland",rank:2,code:1,optional:false},
Arg{name:"sice",rank:2,code:1,optional:false},
Arg{name:"snow",rank:2,code:1,optional:false},
Arg{name:"tke_myj",rank:3,code:1,optional:false},
Arg{name:"exch_h",rank:3,code:1,optional:false},
Arg{name:"ustar",rank:2,code:1,optional:false},
Arg{name:"znt",rank:2,code:1,optional:false},
Arg{name:"el_myj",rank:3,code:1,optional:false},
Arg{name:"pblh",rank:2,code:1,optional:false},
Arg{name:"kpbl",rank:2,code:3,optional:false},
Arg{name:"ct",rank:2,code:1,optional:false},
Arg{name:"akhs",rank:2,code:1,optional:false},
Arg{name:"akms",rank:2,code:1,optional:false},
Arg{name:"elflx",rank:2,code:1,optional:false},
Arg{name:"mixht",rank:2,code:1,optional:false},
Arg{name:"rublten",rank:3,code:1,optional:false},
Arg{name:"rvblten",rank:3,code:1,optional:false},
Arg{name:"rthblten",rank:3,code:1,optional:false},
Arg{name:"rqvblten",rank:3,code:1,optional:true},
Arg{name:"rqcblten",rank:3,code:1,optional:true},
Arg{name:"rqiblten",rank:3,code:1,optional:true},
Arg{name:"rqsblten",rank:3,code:1,optional:true},
Arg{name:"rqrblten",rank:3,code:1,optional:true},
Arg{name:"rqgblten",rank:3,code:1,optional:true},
Arg{name:"ids",rank:0,code:3,optional:false},
Arg{name:"ide",rank:0,code:3,optional:false},
Arg{name:"jds",rank:0,code:3,optional:false},
Arg{name:"jde",rank:0,code:3,optional:false},
Arg{name:"kds",rank:0,code:3,optional:false},
Arg{name:"kde",rank:0,code:3,optional:false},
Arg{name:"ims",rank:0,code:3,optional:false},
Arg{name:"ime",rank:0,code:3,optional:false},
Arg{name:"jms",rank:0,code:3,optional:false},
Arg{name:"jme",rank:0,code:3,optional:false},
Arg{name:"kms",rank:0,code:3,optional:false},
Arg{name:"kme",rank:0,code:3,optional:false},
Arg{name:"its",rank:0,code:3,optional:false},
Arg{name:"ite",rank:0,code:3,optional:false},
Arg{name:"jts",rank:0,code:3,optional:false},
Arg{name:"jte",rank:0,code:3,optional:false},
Arg{name:"kts",rank:0,code:3,optional:false},
Arg{name:"kte",rank:0,code:3,optional:false}
];
const MYJSFC: &[Arg] = &[
Arg{name:"itimestep",rank:0,code:3,optional:false},
Arg{name:"ht",rank:2,code:1,optional:false},
Arg{name:"dz",rank:3,code:1,optional:false},
Arg{name:"pmid",rank:3,code:1,optional:false},
Arg{name:"pint",rank:3,code:1,optional:false},
Arg{name:"th",rank:3,code:1,optional:false},
Arg{name:"t",rank:3,code:1,optional:false},
Arg{name:"qv",rank:3,code:1,optional:false},
Arg{name:"qc",rank:3,code:1,optional:false},
Arg{name:"u",rank:3,code:1,optional:false},
Arg{name:"v",rank:3,code:1,optional:false},
Arg{name:"q2",rank:3,code:1,optional:false},
Arg{name:"tsk",rank:2,code:1,optional:false},
Arg{name:"qsfc",rank:2,code:1,optional:false},
Arg{name:"thz0",rank:2,code:1,optional:false},
Arg{name:"qz0",rank:2,code:1,optional:false},
Arg{name:"uz0",rank:2,code:1,optional:false},
Arg{name:"vz0",rank:2,code:1,optional:false},
Arg{name:"lowlyr",rank:2,code:3,optional:false},
Arg{name:"xland",rank:2,code:1,optional:false},
Arg{name:"ivgtyp",rank:2,code:3,optional:false},
Arg{name:"isurban",rank:0,code:3,optional:false},
Arg{name:"iz0tlnd",rank:0,code:3,optional:false},
Arg{name:"ustar",rank:2,code:1,optional:false},
Arg{name:"znt",rank:2,code:1,optional:false},
Arg{name:"z0base",rank:2,code:1,optional:false},
Arg{name:"pblh",rank:2,code:1,optional:false},
Arg{name:"mavail",rank:2,code:1,optional:false},
Arg{name:"rmol",rank:2,code:1,optional:false},
Arg{name:"akhs",rank:2,code:1,optional:false},
Arg{name:"akms",rank:2,code:1,optional:false},
Arg{name:"rib",rank:2,code:1,optional:false},
Arg{name:"chs",rank:2,code:1,optional:false},
Arg{name:"chs2",rank:2,code:1,optional:false},
Arg{name:"cqs2",rank:2,code:1,optional:false},
Arg{name:"hfx",rank:2,code:1,optional:false},
Arg{name:"qfx",rank:2,code:1,optional:false},
Arg{name:"flx_lh",rank:2,code:1,optional:false},
Arg{name:"flhc",rank:2,code:1,optional:false},
Arg{name:"flqc",rank:2,code:1,optional:false},
Arg{name:"qgh",rank:2,code:1,optional:false},
Arg{name:"cpm",rank:2,code:1,optional:false},
Arg{name:"ct",rank:2,code:1,optional:false},
Arg{name:"u10",rank:2,code:1,optional:false},
Arg{name:"v10",rank:2,code:1,optional:false},
Arg{name:"t02",rank:2,code:1,optional:false},
Arg{name:"th02",rank:2,code:1,optional:false},
Arg{name:"tshltr",rank:2,code:1,optional:false},
Arg{name:"th10",rank:2,code:1,optional:false},
Arg{name:"q02",rank:2,code:1,optional:false},
Arg{name:"qshltr",rank:2,code:1,optional:false},
Arg{name:"q10",rank:2,code:1,optional:false},
Arg{name:"pshltr",rank:2,code:1,optional:false},
Arg{name:"p1000mb",rank:0,code:1,optional:false},
Arg{name:"ids",rank:0,code:3,optional:false},
Arg{name:"ide",rank:0,code:3,optional:false},
Arg{name:"jds",rank:0,code:3,optional:false},
Arg{name:"jde",rank:0,code:3,optional:false},
Arg{name:"kds",rank:0,code:3,optional:false},
Arg{name:"kde",rank:0,code:3,optional:false},
Arg{name:"ims",rank:0,code:3,optional:false},
Arg{name:"ime",rank:0,code:3,optional:false},
Arg{name:"jms",rank:0,code:3,optional:false},
Arg{name:"jme",rank:0,code:3,optional:false},
Arg{name:"kms",rank:0,code:3,optional:false},
Arg{name:"kme",rank:0,code:3,optional:false},
Arg{name:"its",rank:0,code:3,optional:false},
Arg{name:"ite",rank:0,code:3,optional:false},
Arg{name:"jts",rank:0,code:3,optional:false},
Arg{name:"jte",rank:0,code:3,optional:false},
Arg{name:"kts",rank:0,code:3,optional:false},
Arg{name:"kte",rank:0,code:3,optional:false}
];
const MYJSFCINIT: &[Arg] = &[
Arg{name:"lowlyr",rank:2,code:3,optional:false},
Arg{name:"ustar",rank:2,code:1,optional:false},
Arg{name:"z0",rank:2,code:1,optional:false},
Arg{name:"seamask",rank:2,code:1,optional:false},
Arg{name:"xice",rank:2,code:1,optional:false},
Arg{name:"ivgtyp",rank:2,code:3,optional:false},
Arg{name:"restart",rank:0,code:4,optional:false},
Arg{name:"allowed_to_read",rank:0,code:4,optional:false},
Arg{name:"ids",rank:0,code:3,optional:false},
Arg{name:"ide",rank:0,code:3,optional:false},
Arg{name:"jds",rank:0,code:3,optional:false},
Arg{name:"jde",rank:0,code:3,optional:false},
Arg{name:"kds",rank:0,code:3,optional:false},
Arg{name:"kde",rank:0,code:3,optional:false},
Arg{name:"ims",rank:0,code:3,optional:false},
Arg{name:"ime",rank:0,code:3,optional:false},
Arg{name:"jms",rank:0,code:3,optional:false},
Arg{name:"jme",rank:0,code:3,optional:false},
Arg{name:"kms",rank:0,code:3,optional:false},
Arg{name:"kme",rank:0,code:3,optional:false},
Arg{name:"its",rank:0,code:3,optional:false},
Arg{name:"ite",rank:0,code:3,optional:false},
Arg{name:"jts",rank:0,code:3,optional:false},
Arg{name:"jte",rank:0,code:3,optional:false},
Arg{name:"kts",rank:0,code:3,optional:false},
Arg{name:"kte",rank:0,code:3,optional:false},
];
struct Active { prefix:String, site:String, values:BTreeMap<String,Array>,event:i32,absent:Vec<String>,caller:u64 }
fn contiguous(at:u64,code:u32,shape:Vec<usize>,lower:Vec<i64>)->Array {let mut stride=vec![];let mut n=4i64;for &d in &shape {stride.push(n);n*=d as i64;}Array{at,code,width:4,shape,lower,stride}}
fn observe(executable:&Path,output:&Path,initializers:bool)->io::Result<()> {
 let executable=fs::canonicalize(executable)?;fs::create_dir_all(output)?;
 let nm=Command::new("nm").arg("-S").arg(&executable).output()?;if !nm.status.success(){return Err(bad("nm failed"))}
 let symbols=String::from_utf8_lossy(&nm.stdout).lines().filter_map(|l|{let v=l.split_whitespace().collect::<Vec<_>>();if v.len()==4{Some((v[3].to_owned(),(u64::from_str_radix(v[0],16).ok()?,usize::from_str_radix(v[1],16).ok()?,v[2].to_owned())))}else{None}}).collect::<BTreeMap<_,_>>();
 let mut cmd=Command::new(&executable);unsafe{cmd.pre_exec(||{trace(0,0,0,0)?;Ok(())});}let child=cmd.spawn()?;let pid=child.id()as i32;
 if wait(pid)?&255!=127{return Err(bad("missing exec stop"))}trace(0x4200,pid,0,0x100000)?;
 let maps=fs::read_to_string(format!("/proc/{pid}/maps"))?;
 let bias=maps.lines().find_map(|l|{let v=l.split_whitespace().collect::<Vec<_>>();if v.len()>=6&&v[5]==executable.to_string_lossy()&&v[2]=="00000000"{u64::from_str_radix(v[0].split('-').next()?,16).ok()}else{None}}).ok_or_else(||bad("load bias absent"))?;
 let entries:&[(usize,&str)]=if initializers {&[(1,"__module_sf_myjsfc_MOD_myjsfcinit")]}else{&[(0,"__module_bl_myjpbl_MOD_myjpbl"),(1,"__module_sf_myjsfc_MOD_myjsfc")]};
 let mut traps=BTreeMap::new();for &(i,n) in entries {let a=symbols.get(n).ok_or_else(||bad("missing MYJ symbol"))?.0+bias;arm(pid,&mut traps,a,Kind::Entry(i))?;}
 let mut active:Option<Active>=None;let mut event=0;trace(7,pid,0,0)?;
 loop {let status=wait(pid)?;if status&127==0{if active.is_some(){return Err(bad("native exited inside MYJ"))}exit((status>>8)&255)}if status&255!=127{exit(128+(status&127))}let sig=(status>>8)&255;let mut r=Registers::default();trace(12,pid,0,&mut r as *mut _ as u64)?;let at=r.rip-1;
 let Some(t)=traps.get(&at)else{trace(7,pid,0,sig as u64)?;continue};if sig!=5{trace(7,pid,0,sig as u64)?;continue}let byte=t.byte;let kind=t.kind.clone();let rearm=matches!(kind,Kind::Entry(_));
 match kind {
  Kind::Entry(which)=>{if active.is_some(){return Err(bad("nested MYJ entry"))}let args=if initializers {MYJSFCINIT}else if which==0 {MYJPBL}else{MYJSFC};let module=if which==0{"module_bl_myjpbl"}else{"module_sf_myjsfc"};let site=if initializers {"myjsfcinit__external_entry"}else if which==0 {"myjpbl__external_entry"}else{"myjsfc__external_entry"};
   let ptrs=args.iter().enumerate().map(|(i,_)|arg(pid,&r,i)).collect::<io::Result<Vec<_>>>()?;
   let mut bounds=BTreeMap::new();for(a,&p)in args.iter().zip(&ptrs){if a.rank==0&&a.code==3 {if p==0{return Err(bad("null required scalar"))}bounds.insert(a.name,i32::from_le_bytes(bytes(pid,p,4)?.try_into().unwrap())as i64);}}
   let shape=|rank|->io::Result<(Vec<usize>,Vec<i64>)>{let axes=if rank==3{vec![("ims","ime"),("kms","kme"),("jms","jme")]}else{vec![("ims","ime"),("jms","jme")]};let mut s=vec![];let mut l=vec![];for(lo,hi)in axes{let a=*bounds.get(lo).ok_or_else(||bad("lower bound absent"))?;let b=*bounds.get(hi).ok_or_else(||bad("upper bound absent"))?;if b<a||b-a>10000000{return Err(bad("invalid native bounds"))}s.push((b-a+1)as usize);l.push(a);}Ok((s,l))};
   let mut a=Active{prefix:format!("{site}.sunknown.e{event:010}.r000000"),site:site.into(),values:BTreeMap::new(),event,absent:vec![],caller:word(pid,r.rsp)?-bias};event+=1;
   for(spec,&p)in args.iter().zip(&ptrs){if p==0{if !spec.optional{return Err(bad("null required MYJ argument"))}a.absent.push(spec.name.into());continue}let v=if spec.rank==0{Array::scalar(p,spec.code,4)}else{let(s,l)=shape(spec.rank)?;contiguous(p,spec.code,s,l)};a.values.insert(spec.name.into(),v);}
   // Every ELF-visible numerical variable in the scheme module, including tables.
   let prefix=format!("__{module}_MOD_");for(n,(at,size,ty))in &symbols{if n.starts_with(&prefix)&&["B","D","b","d"].contains(&ty.as_str()){if *size==0||size%4!=0{return Err(bad("unsupported MYJ SAVE width"))}let name=&n[prefix.len()..];let code=if name=="itrmx"{3}else{1};let v=if *size==4{Array::scalar(*at+bias,code,4)}else{contiguous(*at+bias,code,vec![size/4],vec![1])};a.values.insert(format!("state.{module}.{name}"),v);}}
   for(n,v)in &a.values{v.save(pid,&output.join(format!("{}.pre.{n}.bin",a.prefix)),n,"pre",a.event,&a.site)?;}
   arm(pid,&mut traps,word(pid,r.rsp)?,Kind::Done)?;active=Some(a);
  }
  Kind::Done=>{let a=active.take().ok_or_else(||bad("return outside MYJ"))?;for(n,v)in&a.values{v.save(pid,&output.join(format!("{}.post.{n}.bin",a.prefix)),n,"post",a.event,&a.site)?;}
   let mut f=fs::File::create(output.join(format!("{}.scope.txt",a.prefix)))?;writeln!(f,"MPI1; explicit arguments and every ELF-visible scheme SAVE word; numerical memory read-only; native_model_step_unknown=true; caller_elf={:#x}; hidden_framework_state_complete=false",a.caller)?;for n in a.absent{writeln!(f,"absent={n}")?;}
  }
 }
 step(pid,&mut r,at,byte,rearm)?;if !rearm{traps.remove(&at);}trace(7,pid,0,0)?;
 }
}
fn main()->io::Result<()> {let args=env::args_os().collect::<Vec<_>>();if args.len()!=3 && !(args.len()==4&&args[3]=="--initializers"){return Err(bad("rrfs-myj-watch NATIVE_EXECUTABLE CAPTURE_DIRECTORY [--initializers]"))}observe(Path::new(&args[1]),Path::new(&args[2]),args.len()==4)}
