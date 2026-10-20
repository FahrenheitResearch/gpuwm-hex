//! Audit captured unwritten storage against native outputs, never replay inputs.
//! MPAS module_ra_cam.F:364-365,678-679 reverses the native tau level index.
use std::{env,fs,path::{Path,PathBuf}};

struct Record { kind:i32, dims:[usize;5], low:[i32;5], data:Vec<u8> }
fn read(path:&Path)->Record {
    let bytes=fs::read(path).unwrap();
    assert!(bytes.len()>=860); assert_eq!(&bytes[..8],b"ORCL0001");
    let int=|i:usize|i32::from_le_bytes(bytes[8+i*4..12+i*4].try_into().unwrap());
    assert!(int(16)>=0); let offset=860+int(16) as usize*4;
    let dims=std::array::from_fn(|i|usize::try_from(int(3+i)).unwrap());
    let low=std::array::from_fn(|i|int(8+i));
    Record {kind:int(0),dims,low,data:bytes[offset..].to_vec()}
}
fn main() {
    let root=PathBuf::from(env::args().nth(1).expect("complete capture directory"));
    let mut files:Vec<_>=fs::read_dir(&root).unwrap().map(|v|v.unwrap().path())
        .filter(|p|p.file_name().unwrap().to_str().unwrap().ends_with(".pre.dosw.bin"))
        .collect(); files.sort(); assert!(!files.is_empty());
    let mut checked=0; let mut total=0; let mut differences=0;
    println!("{{\"scope\":\"automatic PRE/native POST storage audit; no CUDA or forecast claim\",\"process_count\":1,\"excluded_words\":0,\"captures\":{},\"records\":[",files.len());
    for file in files {
        let name=file.file_name().unwrap().to_str().unwrap();
        let prefix=name.strip_suffix(".pre.dosw.bin").unwrap();
        let flag=read(&file); assert_eq!(flag.kind,4); assert_eq!(flag.data.len(),4);
        if u32::from_le_bytes(flag.data[..4].try_into().unwrap())!=0 {continue;}
        for (entry,output) in [("tauxcl","taucldc"),("tauxci","taucldi")] {
            let automatic:Vec<_>=fs::read_dir(&root).unwrap().map(|v|v.unwrap().path())
                .filter(|p| {
                    let name=p.file_name().unwrap().to_str().unwrap();
                    let stem=format!("{}.pre.allocation.module_ra_cam.camrad.{}.",prefix,entry);
                    let Some(suffix)=name.strip_prefix(&stem).and_then(|s|s.strip_suffix(".bin")) else {return false};
                    let ordinal=if let Some(line)=suffix.strip_prefix('L') {
                        let Some((line,ordinal))=line.split_once('.') else {return false};
                        if line.is_empty() || !line.bytes().all(|b|b.is_ascii_digit()) {return false;}
                        ordinal
                    } else {suffix};
                    ordinal.strip_prefix('n').is_some_and(|n|!n.is_empty() && n.bytes().all(|b|b.is_ascii_digit()))
                })
                .collect(); assert_eq!(automatic.len(),1);
            let before=read(&automatic[0]); let after=read(&root.join(format!("{}.post.{}.bin",prefix,output)));
            assert_eq!(before.kind,2); assert_eq!(after.kind,1);
            let ncol=before.dims[0]; let nlev=before.dims[1]-1;
            assert_eq!(before.low[..2],[1,0]); assert_eq!(after.low[..3],[1,1,1]);
            assert_eq!(before.data.len(),ncol*(nlev+1)*8); assert_eq!(after.dims[..3],[ncol,nlev+1,1]);
            assert_eq!(after.data.len(),ncol*(nlev+1)*4);
            let mut different=0;
            for level in 0..nlev { for column in 0..ncol {
                let offset=(column+ncol*(nlev-level))*8;
                let value=f64::from_bits(u64::from_le_bytes(before.data[offset..offset+8].try_into().unwrap()));
                let offset=(column+ncol*level)*4;
                let expected=u32::from_le_bytes(after.data[offset..offset+4].try_into().unwrap());
                if (value as f32).to_bits()!=expected {different+=1;}
            }}
            if checked>0 {println!(",");} checked+=1; total+=ncol*nlev; differences+=different;
            print!("{{\"capture\":\"{}\",\"field\":\"{}\",\"words\":{},\"different_words\":{}}}",prefix,output,ncol*nlev,different);
        }
    }
    println!("],\"checked_records\":{},\"words\":{},\"different_words\":{}}}",checked,total,differences);
    assert!(checked>0); if differences!=0 {std::process::exit(1);}
}
