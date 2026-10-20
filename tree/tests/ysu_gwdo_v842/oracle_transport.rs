//! Test-only MPAS ORCL0001 decoder/encoder. All field transport stays in Rust.
//! MPAS module_bl_gwdo.F:158-235: copy only the live tile; retain entry ghosts.
use std::{env, fs, io, path::{Path, PathBuf}};
const FIXED: usize = 860;
struct Record { raw: Vec<u8>, h: [i32;21], offset: usize }
fn bad(s:&str)->io::Error { io::Error::new(io::ErrorKind::InvalidData,s) }
fn read(p:&Path)->io::Result<Record> {
    let raw=fs::read(p)?;
    if raw.len()<FIXED || &raw[..8]!=b"ORCL0001" {return Err(bad("invalid oracle record"));}
    let mut h=[0;21];
    for (i,x) in h.iter_mut().enumerate() {*x=i32::from_le_bytes(raw[8+4*i..12+4*i].try_into().unwrap());}
    if !(0..=5).contains(&h[2]) || h[1]<=0 || h[16]<0 || h[19]!=256 || h[20]!=0 {return Err(bad("invalid record header"));}
    let offset=FIXED+4*h[16] as usize;
    let count=h[3..8].iter().try_fold(1usize,|a,&b|if b>0 {a.checked_mul(b as usize)} else {None}).ok_or_else(||bad("invalid shape"))?;
    if raw.len()!=offset+count*h[1] as usize {return Err(bad("truncated record"));}
    Ok(Record{raw,h,offset})
}
fn file(prefix:&Path,phase:&str,name:&str)->PathBuf {PathBuf::from(format!("{}.{}.{}.bin",prefix.display(),phase,name))}
fn scalar_i(prefix:&Path,name:&str)->io::Result<i32> {
    let r=read(&file(prefix,"pre",name))?;
    if r.h[0]!=3 || r.h[2]!=0 {return Err(bad("expected integer scalar"));}
    Ok(i32::from_le_bytes(r.raw[r.offset..].try_into().unwrap()))
}
fn scalar_f(prefix:&Path,name:&str)->io::Result<f32> {
    let r=read(&file(prefix,"pre",name))?;
    if r.h[0]!=1 || r.h[2]!=0 {return Err(bad("expected RKIND scalar"));}
    Ok(f32::from_le_bytes(r.raw[r.offset..].try_into().unwrap()))
}
fn geometry(prefix:&Path)->io::Result<(usize,usize,usize)> {
    let r=read(&file(prefix,"pre","u3d"))?;
    let n=r.h[3] as usize;let live=scalar_i(prefix,"kte")? as usize;
    if r.h[0]!=1 || r.h[2]!=3 || r.h[5]!=1 || r.h[8..11]!=[1,1,1]
        || scalar_i(prefix,"its")?!=1 || scalar_i(prefix,"ite")?!=n as i32
        || scalar_i(prefix,"kts")?!=1 || scalar_i(prefix,"jts")?!=1 || scalar_i(prefix,"jte")?!=1
        || live<2 || live+1!=r.h[4] as usize {return Err(bad("capture must be complete singleton-J native tile with one top ghost"));}
    Ok((n,live,r.h[4] as usize))
}
const INPUTS: [(&str,&str,bool);21]=[
    ("u3d","u_p",true),("v3d","v_p",true),("t3d","t_p",true),("qv3d","qv_p",true),
    ("p3d","pres_hydd_p",true),("p3di","pres2_hydd_p",true),("pi3d","pi_p",true),
    ("z","zmid_p",true),("rublten","rublten",true),("rvblten","rvblten",true),
    ("dx","dx_p",false),("var2d","var2d",false),("oc12d","con",false),
    ("oa2d1","oa1",false),("oa2d2","oa2",false),("oa2d3","oa3",false),("oa2d4","oa4",false),
    ("ol2d1","ol1",false),("ol2d2","ol2",false),("ol2d3","ol3",false),("ol2d4","ol4",false)];
fn unpack(prefix:&Path,out:&Path)->io::Result<()> {
    fs::create_dir(out)?;let(n,k,storage)=geometry(prefix)?;
    // MPAS driver_gwdo:541-542 fixes rotation to 0,1. Refuse a different driver.
    for (name,value) in [("sina",0f32),("cosa",1f32)] {
        let r=read(&file(prefix,"pre",name))?;
        if r.h[0]!=1 || r.raw[r.offset..].chunks_exact(4).any(|x|x!=value.to_le_bytes()) {return Err(bad("capture uses a different rotation"));}
    }
    let mut entries=vec![];
    for (src,dst,layer) in INPUTS {
        let r=read(&file(prefix,"pre",src))?;
        let levels=if src=="p3di" {k+1} else if layer {k} else {1};
        if r.h[0]!=1 || r.h[1]!=4 || r.h[3]!=n as i32 || r.h[4]!=if layer {storage as i32} else {1}
            || r.raw.len()-r.offset!=n*if layer {storage} else {1}*4 {return Err(bad("input layout mismatch"));}
        fs::write(out.join(format!("{dst}.raw")),&r.raw[r.offset..r.offset+n*levels*4])?;
        entries.push(format!("\"{dst}\":{}",if layer {format!("[{levels},{n}]")} else {format!("[{n}]")}));
    }
    let metadata=format!("{{\"nlev\":{k},\"ncol\":{n},\"dt_seconds\":{},\"constants\":{{\"gravity\":{},\"cp\":{},\"rd\":{},\"fv\":{},\"pi\":{}}},\"inputs\":{{{}}}}}",
        scalar_f(prefix,"dt")?,scalar_f(prefix,"g")?,scalar_f(prefix,"cp")?,scalar_f(prefix,"rd")?,scalar_f(prefix,"ep1")?,scalar_f(prefix,"pi")?,entries.join(","));
    fs::write(out.join("metadata.json"),metadata)
}
fn write_post(prefix:&Path,name:&str,mut r:Record)->io::Result<()> {
    r.raw[348..604].fill(b' ');r.raw[348..352].copy_from_slice(b"post");
    fs::write(file(prefix,"post",name),r.raw)
}
fn pack(input:&Path,output:&Path,work:&Path)->io::Result<()> {
    let(n,k,_)=geometry(input)?;
    let standalone=fs::read(work.join("standalone.raw"))?;
    if standalone.len()!=4*(6*n*k+2*n) {return Err(bad("standalone output size mismatch"));}
    if work.join("candidate.raw").exists() {
        let candidate=fs::read(work.join("candidate.raw"))?;
        if candidate.len()!=standalone.len() {return Err(bad("host candidate size mismatch"));}
        let mut start=0;
        for (name,layer) in [("rublten",true),("rvblten",true),("dtaux3d",true),("dtauy3d",true),
                             ("rubldiff",true),("rvbldiff",true),("dusfcg",false),("dvsfcg",false)] {
            let end=start+4*n*if layer {k} else {1};
            fs::write(work.join(format!("{name}.raw")),&candidate[start..end])?;start=end;
        }
    }
    let mut cursor=0;let mut rows=vec![];let mut mismatched=false;
    for (name,layer) in [("rublten",true),("rvblten",true),("dtaux3d",true),("dtauy3d",true),
                         ("rubldiff",true),("rvbldiff",true),("dusfcg",false),("dvsfcg",false)] {
        let data=fs::read(work.join(format!("{name}.raw")))?;
        let size=4*n*if layer {k} else {1};
        if data.len()!=size {return Err(bad("candidate output size mismatch"));}
        let count=data.chunks_exact(4).zip(standalone[cursor..cursor+size].chunks_exact(4)).filter(|(a,b)|a!=b).count();
        let first=data.chunks_exact(4).zip(standalone[cursor..cursor+size].chunks_exact(4)).position(|(a,b)|a!=b);
        rows.push(format!("\"{name}\":{{\"words\":{},\"different_words\":{count},\"first_word\":{}}}",size/4,first.map_or("null".to_string(),|i|i.to_string())));
        mismatched |= count!=0;cursor+=size;
    }
    fs::write(PathBuf::from(format!("{}.standalone.json",output.display())),format!("{{\"schema\":\"ysu-standalone-real-columns-v1\",\"nlev\":{k},\"ncol\":{n},\"all_outputs_bitwise_equal\":{},\"arrays\":{{{}}}}}",!mismatched,rows.join(",")))?;
    for (name,layer) in [("rublten",true),("rvblten",true),("dtaux3d",true),("dtauy3d",true),("dusfcg",false),("dvsfcg",false)] {
        let mut r=read(&file(input,"pre",name))?;
        let data=fs::read(work.join(format!("{name}.raw")))?;
        if data.len()!=n*if layer {k} else {1}*4 {return Err(bad("output payload size mismatch"));}
        r.raw[r.offset..r.offset+data.len()].copy_from_slice(&data);
        write_post(output,name,r)?;
    }
    // MPAS bl_gwdo.F90:218-219: all-blank message and zero error.
    for name in ["errflg","errmsg"] {
        let mut r=read(&file(input,"pre",name))?;
        if name=="errflg" {r.raw[r.offset..].fill(0);} else {r.raw[r.offset..].fill(b' ');}
        write_post(output,name,r)?;
    }
    // MPAS bl_gwdo.F90:187-188: initialized DATA, read-only throughout the call.
    let name="local.bl_gwdo.bl_gwdo_run.nwdir";
    let r=read(&file(input,"pre",name))?;
    let expected=[6i32,7,5,8,2,3,1,4].into_iter().flat_map(i32::to_le_bytes).collect::<Vec<_>>();
    if r.raw[r.offset..]!=expected {return Err(bad("native saved directional table differs"));}
    write_post(output,name,r)?;
    if mismatched {return Err(bad("CUDA differs from strict pristine standalone MPAS"));}
    Ok(())
}
fn main()->io::Result<()> {
    let a:Vec<_>=env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("unpack") if a.len()==4=>unpack(Path::new(&a[2]),Path::new(&a[3])),
        Some("pack") if a.len()==5=>pack(Path::new(&a[2]),Path::new(&a[3]),Path::new(&a[4])),
        _=>Err(bad("usage: transport unpack INPUT_PREFIX WORK | pack INPUT_PREFIX OUTPUT_PREFIX WORK"))
    }
}
