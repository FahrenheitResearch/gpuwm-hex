use std::{env,fs};
fn main() {
    let root = env::args().nth(1).expect("array directory");
    let names = fs::read_to_string(format!("{root}/fields.txt")).unwrap();
    let mut total=0u64; let mut bad=0u64; let mut first=Vec::new();
    for name in names.lines() {
        let a=fs::read(format!("{root}/{name}.native.bin")).unwrap();
        let b=fs::read(format!("{root}/{name}.cuda.bin")).unwrap();
        assert_eq!(a.len(),b.len()); assert_eq!(a.len()%4,0);
        let mut n=0u64; let mut samples=Vec::new();
        for (i,(x,y)) in a.chunks_exact(4).zip(b.chunks_exact(4)).enumerate() {
            total+=1;
            if x!=y {
                n+=1; bad+=1;
                if samples.len()<8 { samples.push(format!("{{\"index\":{i},\"native_bits\":{},\"cuda_bits\":{}}}",u32::from_le_bytes(x.try_into().unwrap()),u32::from_le_bytes(y.try_into().unwrap()))); }
            }
        }
        if n>0 { first.push(format!("{{\"field\":\"{name}\",\"differing_words\":{n},\"samples\":[{}]}}",samples.join(","))); }
    }
    println!("{{\"words\":{total},\"differing_words\":{bad},\"fields\":[{}]}}",first.join(","));
}
