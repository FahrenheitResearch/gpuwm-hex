//! Decode real f1 PRE pool captures; no weather-file processing in Python.
//! MPAS mpas_atmphys_driver_sfclayer.F:352-546, RKIND surface inputs.
use std::{fs, io::{self, Write}, path::Path};
const FIXED: usize = 8 + 21*4 + 3*256;
fn record(path: &Path, ty: i32) -> (Vec<i32>, Vec<i32>, Vec<u8>) {
    let b=fs::read(path).unwrap();
    assert!(b.len()>=FIXED && &b[..8]==b"ORCL0001");
    let h:Vec<i32>=b[8..92].chunks_exact(4).map(|v|i32::from_le_bytes(v.try_into().unwrap())).collect();
    assert_eq!(h[0],ty); assert_eq!(h[1],4); assert_eq!(h[19],256); assert_eq!(h[20],0);
    assert!((0..=1).contains(&h[2])); assert!(h[3..8].iter().all(|v|*v>=1)); assert!(h[16]>=0);
    let off=FIXED+4*h[16] as usize;
    assert_eq!(b.len(),off+4*h[3..8].iter().map(|v|*v as usize).product::<usize>());
    let ids=b[FIXED..off].chunks_exact(4).map(|v|i32::from_le_bytes(v.try_into().unwrap())).collect();
    (h,ids,b[off..].to_vec())
}
fn main() {
    let args:Vec<String>=std::env::args().collect(); let dir=Path::new(&args[1]);
    let (_,_,d)=record(&dir.join("block.dimensions.nCellsSolve.bin"),3);
    let n=i32::from_le_bytes(d[..4].try_into().unwrap()) as usize; assert!(n>0);
    let mut reference=None; let mut stdout=io::stdout().lock();
    stdout.write_all(&(n as u64).to_le_bytes()).unwrap();
    for name in ["skintemp","sst","xice"] {
        let (h,ids,v)=record(&dir.join(format!("sfc_input.{name}.bin")),1);
        assert_eq!(h[2],1); assert!(n<=h[3] as usize); assert_eq!(ids.len(),h[3] as usize);
        if let Some(ref r)=reference {assert_eq!(&ids,r);} else {reference=Some(ids);}
        stdout.write_all(&v[..4*n]).unwrap();
    }
}
