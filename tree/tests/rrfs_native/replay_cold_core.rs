//! Complete real cold pool replay, including all unchanged allocation words.
//! MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:1268-1309,1481-1584
//! at 1d6e370e. Only PRE provides bridge operands and entry storage.
use std::{collections::BTreeMap,ffi::CString,fs,io::{self,Write},path::{Path,PathBuf},
          process::{Command,Stdio}};
#[path="oracle_record.rs"] mod record;
fn bad(message:impl ToString)->io::Error {io::Error::other(message.to_string())}
struct Field {header:[i32;21], data:Vec<u8>}
fn read(path:&Path)->io::Result<Field> {
    let mut header=[0;21];let mut len=0;
    let filename=CString::new(path.to_str().ok_or_else(||bad("non-UTF8 record path"))?).map_err(bad)?;
    unsafe {
        let pointer=record::oracle_read(filename.as_ptr(),header.as_mut_ptr(),&mut len);
        if pointer.is_null(){return Err(bad("invalid real native record"));}
        let data=std::slice::from_raw_parts(pointer,len).to_vec();record::oracle_free(pointer,len);
        Ok(Field{header,data})
    }
}
fn int(field:&Field)->io::Result<usize> {
    if field.header[0]!=3 || field.header[2]!=0 || field.data.len()!=4 {return Err(bad("native integer scalar required"));}
    usize::try_from(i32::from_le_bytes(field.data[..4].try_into().unwrap())).map_err(bad)
}
fn floats(field:&Field)->io::Result<Vec<f32>> {
    if field.header[0]!=1 || field.header[1]!=4 {return Err(bad("native binary32 required"));}
    Ok(field.data.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect())
}
fn bridge(path:&str,payload:Vec<u8>)->io::Result<Vec<u8>> {
    let mut child=Command::new(path).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
    let mut input=child.stdin.take().ok_or_else(||bad("bridge stdin missing"))?;
    let writer=std::thread::spawn(move||input.write_all(&payload));
    let output=child.wait_with_output()?;writer.join().map_err(|_|bad("bridge writer panic"))??;
    if !output.status.success(){return Err(bad(String::from_utf8_lossy(&output.stderr)));}
    Ok(output.stdout)
}
fn get<'a>(fields:&'a BTreeMap<String,Field>,name:&str)->io::Result<&'a Field> {
    fields.get(&format!("pool.{name}")).ok_or_else(||bad(format!("missing native PRE {name}")))
}
fn connectivity(fields:&BTreeMap<String,Field>,executable:&str)->io::Result<BTreeMap<String,Vec<u8>>> {
    let counts=get(fields,"nedgesoncell")?;let neighbors=get(fields,"cellsoncell")?;
    let entry_counts=get(fields,"ncellsoncellsoncell")?;let entry=get(fields,"cellsoncellsoncell")?;
    let cells=counts.header[3] as usize;let solved=int(get(fields,"ncellssolve")?)?;
    let width=int(get(fields,"maxedges")?)?;let outer_width=int(get(fields,"maxedges2")?)?;
    let physical=int(get(fields,"ncells")?)?;
    if solved>physical || physical>cells || counts.header[0]!=3 || counts.header[2]!=1 ||
       neighbors.header[0]!=3 || neighbors.header[3..5]!=[width as i32,cells as i32] ||
       entry_counts.header[0]!=3 || entry_counts.header[3]!=cells as i32 ||
       entry.header[0]!=3 || entry.header[3..5]!=[outer_width as i32,cells as i32] {
        return Err(bad("native second-ring allocation shape mismatch"));
    }
    let mut payload=b"RRFSN2V2".to_vec();
    for n in [cells,solved,width,outer_width]{payload.extend((n as u64).to_le_bytes());}
    payload.extend(&counts.data);
    // Native pointers are one-based; preserve even unused integer pool words
    // reversibly rather than replace them with invented defaults.
    for field in [neighbors,entry_counts,entry] {
        for raw in field.data.chunks_exact(4) {
            let value=i32::from_le_bytes(raw.try_into().unwrap());
            payload.extend(if std::ptr::eq(field,entry_counts){value}else{value.wrapping_sub(1)}.to_le_bytes());
        }
    }
    let mut output=bridge(executable,payload)?;
    if output.len()!=4*(cells+cells*outer_width){return Err(bad("second-ring bridge output extent changed"));}
    for raw in output[4*cells..].chunks_exact_mut(4){
        let value=i32::from_le_bytes(raw.try_into().unwrap()).wrapping_add(1);
        raw.copy_from_slice(&value.to_le_bytes());
    }
    Ok(BTreeMap::from([("pool.ncellsoncellsoncell".into(),output[..4*cells].to_vec()),
                      ("pool.cellsoncellsoncell".into(),output[4*cells..].to_vec())]))
}
fn damping(fields:&BTreeMap<String,Field>,executable:&str)->io::Result<BTreeMap<String,Vec<u8>>> {
    let nc=int(get(fields,"ncells")?)?;let nl=int(get(fields,"nvertlevels")?)?;
    let zfield=get(fields,"zgrid")?;let entry=get(fields,"dss")?;let density=get(fields,"meshdensity")?;
    let storage=entry.header[4] as usize;
    if nc>storage || entry.header[3]!=nl as i32 || zfield.header[3..5]!=[(nl+1) as i32,storage as i32] ||
       density.header[3]!=storage as i32 {return Err(bad("native damping allocation shape mismatch"));}
    let z=floats(zfield)?;let rho=floats(density)?;
    let xnutr=floats(get(fields,"config_xnutr")?)?;let zd=floats(get(fields,"config_zd")?)?;
    if xnutr.len()!=1 || zd.len()!=1{return Err(bad("native damping scalar config required"));}
    let mut payload=b"RRFSDSS1".to_vec();for n in [nl,nc]{payload.extend((n as u64).to_le_bytes());}
    payload.extend(xnutr[0].to_le_bytes());payload.extend(zd[0].to_le_bytes());
    for k in 0..=nl{for c in 0..nc{payload.extend(z[c*(nl+1)+k].to_le_bytes());}}
    for value in &rho[..nc]{payload.extend(value.to_le_bytes());}
    let output=bridge(executable,payload)?;
    if output.len()!=4*nc*nl{return Err(bad("damping bridge output extent changed"));}
    let mut native=vec![0u8;entry.data.len()]; // core.F:1297 clears ALL storage.
    for c in 0..nc{for k in 0..nl{
        native[4*(c*nl+k)..4*(c*nl+k+1)].copy_from_slice(&output[4*(k*nc+c)..4*(k*nc+c+1)]);
    }}
    Ok(BTreeMap::from([("pool.dss".into(),native)]))
}
fn main()->io::Result<()> {
    let args:Vec<_>=std::env::args().collect();
    if args.len()!=5{return Err(bad("BANK CONNECTIVITY_BRIDGE DAMPING_BRIDGE OUT"));}
    let bank=Path::new(&args[1]);let mut calls:BTreeMap<String,Vec<(String,PathBuf)>>=BTreeMap::new();
    for path in fs::read_dir(bank)? {
        let path=path?.path();let name=path.file_name().and_then(|n|n.to_str()).ok_or_else(||bad("bad record path"))?;
        if let Some((event,field))=name.split_once(".post.") {
            if !event.starts_with("core__") || !field.ends_with(".bin"){return Err(bad("unrecognized cold POST record"));}
            calls.entry(event.into()).or_default().push((field.trim_end_matches(".bin").into(),path));
        }
    }
    if calls.len()!=2{return Err(bad("expected both complete cold initialization boundaries"));}
    let mut rows=Vec::new();let mut words=0;let mut differences=0;let mut post_records=0;
    for (event,outputs) in &calls {
        let mut pre=BTreeMap::new();
        for (field,_) in outputs {pre.insert(field.clone(),read(&bank.join(format!("{event}.pre.{field}.bin")))?);}
        let changed=if event.starts_with("core__atm_compute_cellsoncellsoncell.") {
            connectivity(&pre,&args[2])?
        }else if event.starts_with("core__atm_compute_damping_coefs.") {
            damping(&pre,&args[3])?
        }else{return Err(bad("unknown real cold core boundary"));};
        for (field,path) in outputs {
            let expected=read(path)?;let entry=&pre[field];
            if expected.header!=entry.header{return Err(bad("PRE/POST metadata changed"));}
            let actual=changed.get(field).unwrap_or(&entry.data);
            if expected.data.len()!=actual.len() || actual.len()%4!=0{return Err(bad("output word extent mismatch"));}
            let different=actual.chunks_exact(4).zip(expected.data.chunks_exact(4)).filter(|(a,b)|a!=b).count();
            let first=actual.chunks_exact(4).zip(expected.data.chunks_exact(4)).position(|(a,b)|a!=b);
            let count=actual.len()/4;words+=count;differences+=different;post_records+=1;
            rows.push(format!("{{\"call\":\"{event}\",\"field\":\"{field}\",\"words\":{count},\"different_words\":{different},\"first_difference\":{}}}",first.map_or("null".into(),|v|v.to_string())));
        }
    }
    fs::write(&args[4],format!("{{\"schema\":\"rrfs-real-cold-core-replay-v1\",\"native_ranks\":1,\"complete_calls\":{},\"post_records\":{post_records},\"compared_words\":{words},\"different_words\":{differences},\"passed\":{},\"comparisons\":[{}]}}\n",calls.len(),differences==0,rows.join(",")))?;
    if differences!=0{return Err(bad(format!("{differences} different real cold output words")));}Ok(())
}
