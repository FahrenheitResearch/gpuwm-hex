//! Compare authenticated native changed-word masks; JSON metadata maps bit positions to fields.
use std::{env,fs,io};
fn relation(left:&[u8],right:&[u8])->io::Result<(u64,u64,u64)>{
    if left.len()!=right.len() || left.len()<8 || &left[..8]!=b"NCMASK01" || &right[..8]!=b"NCMASK01" {
        return Err(io::Error::other("native mask header or extent differs"))
    }
    let mut common=0;let mut left_only=0;let mut right_only=0;
    for (&a,&b) in left[8..].iter().zip(&right[8..]) {
        common+=(a&b).count_ones() as u64;
        left_only+=(a&!b).count_ones() as u64;
        right_only+=(b&!a).count_ones() as u64;
    }
    Ok((common,left_only,right_only))
}
fn main()->io::Result<()>{
    let args:Vec<_>=env::args().collect();if args.len()!=3{return Err(io::Error::other("compare_word_masks LEFT.mask RIGHT.mask"))}
    let (common,left_only,right_only)=relation(&fs::read(&args[1])?,&fs::read(&args[2])?)?;
    println!("{{\"common_words\":{common},\"left_only_words\":{left_only},\"right_only_words\":{right_only},\"same_word_set\":{}}}",left_only==0&&right_only==0);
    Ok(())
}
#[cfg(test)]mod tests{
 use super::*;
 #[test]fn exact_positions_matter(){
   let a=[b"NCMASK01".as_slice(),&[0b01010101,0b10000000]].concat();
   let b=[b"NCMASK01".as_slice(),&[0b10101010,0b10000000]].concat();
   assert_eq!(relation(&a,&b).unwrap(),(1,4,4));assert_eq!(relation(&a,&a).unwrap(),(5,0,0));
 }
 #[test]fn malformed_masks_have_no_grade(){assert!(relation(b"",b"").is_err());assert!(relation(b"NCMASK01",b"NCMASK02").is_err());assert!(relation(b"NCMASK01",b"NCMASK01x").is_err());}
}
