//! MPAS mpas_atmphys_rrtmg_swinit.F:70-1206 table-read boundary.
use std::path::Path;
#[path = "../rrtmg_sw_tables.rs"]
mod rrtmg_sw_tables;
use rrtmg_sw_tables::decode_file;
fn main() -> std::io::Result<()> {
    let args:Vec<String>=std::env::args().collect();
    if args.len()!=3 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput,
            "usage: rw_mpas_rrtmg_sw_tables RRTMG_SW_DATA NEW_OUTPUT_DIRECTORY"));
    }
    decode_file(Path::new(&args[1]),Path::new(&args[2]))
}
