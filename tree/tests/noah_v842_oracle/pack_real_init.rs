//! Pack already Rust-decoded MPAS initial surface/soil fields for the native ABI.
//! No interpolation or substitute numerical forcing. Four restart/albedo flags.
use std::{env,fs,io::Write,path::Path};
fn field(dir:&Path,index:usize)->Vec<f64>{
    let bytes=fs::read(dir.join(format!("{index:04}.f64"))).unwrap();
    assert_eq!(bytes.len()%8,0);
    bytes.chunks_exact(8).map(|x|f64::from_le_bytes(x.try_into().unwrap())).collect()
}
fn main(){
    let a:Vec<_>=env::args().collect();assert_eq!(a.len(),3,"DECODED_DIRECTORY OUTPUT_PREFIX");
    let dir=Path::new(&a[1]);let prefix=&a[2];
    let names=["tslb","smois","sh2o","snoalb","snow","snowh","ivgtyp","isltyp"];
    let fields:Vec<_>=names.iter().enumerate().map(|(k,_)|field(dir,k)).collect();let n=fields[6].len();
    for (k,f) in fields.iter().enumerate(){assert_eq!(f.len(),n*if k<3{4}else{1});}
    let mut floats=fs::File::create(format!("{prefix}.f32")).unwrap();
    let mut ints=fs::File::create(format!("{prefix}.i32")).unwrap();
    for flags in 0..4{for c in 0..n{
        for f in &fields[..3]{for s in 0..4{floats.write_all(&(f[c*4+s] as f32).to_le_bytes()).unwrap();}}
        for f in &fields[3..6]{floats.write_all(&(f[c] as f32).to_le_bytes()).unwrap();}
        for v in [fields[6][c] as i32,fields[7][c] as i32,flags&1,(flags>>1)&1]{ints.write_all(&v.to_le_bytes()).unwrap();}
    }}
    println!("{} physical cells, {} replay columns",n,n*4);
}
