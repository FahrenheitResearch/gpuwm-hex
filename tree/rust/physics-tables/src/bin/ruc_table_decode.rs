//! Rust input bridge for the RRFS fork's RUC land-surface tables (no native library).
//! MPAS-Model-rrfs 1d6e370e mpas_atmphys_lsm_rucinit.F:148-174,248-520 (RUCLSM_SOILVEGPARM), reading the
//! pinned VEGPARM.TBL, SOILPARM.TBL and GENPARM.TBL with Fortran list-directed semantics: the first token(s)
//! of each record, `character*8` dataset names, rows stored in file order. Real words are the correctly
//! rounded binary32 of each decimal token (as libgfortran's list-directed REAL(4) read).
//!
//! Output: the 955 little-endian words of module_ruc_land in hexcore.mpasphys.rrfs.ruc_lsm_8c9f1a8's
//! TABLE_LAYOUT order (RRFS-MPAS RUCLSM 8c9f1a8 MPAS/module_ruc_land.F90:35-58): unread entries keep the
//! module's static zero, `targetcell` its initial 0, and the two character slots carry the port's tokens
//! (lutype 4 for USGS-RUC else 5, sltype 6).
use std::{env, fs, path::Path};

const NLUS: usize = 50;
const NSLTYPE: usize = 30;
const NSLOPE: usize = 30;
const WORDS: usize = 955;

fn tokens(line: &str) -> Vec<&str> {
    line.split(['\'', '"']).next().unwrap().split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty()).collect()
}
fn first_token(line: &str) -> String {
    // A list-directed character read: the first value, apostrophe- or quote-delimited or blank/comma-ended.
    let s = line.trim_start();
    if let Some(rest) = s.strip_prefix('\'').or_else(|| s.strip_prefix('"')) {
        return rest.split(['\'', '"']).next().unwrap().to_owned();
    }
    s.split(|c: char| c == ',' || c.is_whitespace()).next().unwrap_or("").to_owned()
}
fn char8(value: &str) -> String {
    let v: String = value.chars().take(8).collect();
    v.trim_end().to_owned()
}
fn read_lines(path: &Path) -> Vec<String> {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines().map(str::to_owned).collect()
}
fn real(value: &str) -> u32 {
    value.replace(['d', 'D'], "E").parse::<f32>()
        .unwrap_or_else(|e| panic!("bad real {value}: {e}")).to_bits()
}
fn int(value: &str) -> u32 {
    value.parse::<i32>().unwrap_or_else(|e| panic!("bad integer {value}: {e}")) as u32
}

struct Layout { words: Vec<u32> }
impl Layout {
    fn put(&mut self, offset: usize, value: u32) { self.words[offset] = value; }
}

// Offsets of TABLE_LAYOUT (ruc_lsm_8c9f1a8.py), checked against it by the Python loader.
const LUCATS: usize = 0; const BARE: usize = 1; const NATURAL: usize = 2; const CROP: usize = 3;
const URBAN: usize = 4; const LUTYPE: usize = 5; const IFORTBL: usize = 6; const SNUPTBL: usize = 56;
const RSTBL: usize = 106; const RGLTBL: usize = 156; const HSTBL: usize = 206; const LAITBL: usize = 256;
const ALBTBL: usize = 306; const Z0TBL: usize = 356; const LEMITBL: usize = 406; const PCTBL: usize = 456;
const SHDTBL: usize = 506; const MAXALB: usize = 556; const TOPT: usize = 606; const CMCMAX: usize = 607;
const CFACTR: usize = 608; const RSMAX: usize = 609; const SLCATS: usize = 610; const SLTYPE: usize = 611;
const BB: usize = 612; const DRYSMC: usize = 642; const HC: usize = 672; const MAXSMC: usize = 702;
const REFSMC: usize = 732; const SATPSI: usize = 762; const SATDK: usize = 792; const SATDW: usize = 822;
const WLTSMC: usize = 852; const QTZ: usize = 882; const SLPCATS: usize = 912; const SLOPE: usize = 913;
const SBETA: usize = 943; const FXEXP: usize = 944; const CSOIL: usize = 945; const SALP: usize = 946;
const REFDK: usize = 947; const REFKDT: usize = 948; const FRZK: usize = 949; const ZBOT: usize = 950;
const SMLOW: usize = 951; const SMHIGH: usize = 952; const CZIL: usize = 953;

// rucinit.F:260-345 (VEGPARM.TBL).
fn vegetation(dir: &Path, wanted: &str, out: &mut Layout) {
    let lines = read_lines(&dir.join("VEGPARM.TBL"));
    let mut pos = 1; // read(16,*) a_string
    let mut loops = 0;
    loop {
        assert!(pos + 1 < lines.len(), "land use dataset '{wanted}' not found in VEGPARM.TBL");
        let lutype = char8(&first_token(&lines[pos]));
        let head = tokens(&lines[pos + 1]);
        let lucats: usize = head[0].parse().unwrap();
        pos += 2;
        if lutype == wanted {
            assert!(lucats <= NLUS, "VEGPARM exceeds RUC NLUS");
            out.put(LUCATS, lucats as u32);
            for lc in 0..lucats {
                let row = tokens(&lines[pos]); pos += 1;
                assert!(row.len() >= 13, "short VEGPARM row: {}", lines[pos - 1]);
                for (column, base) in [(1, ALBTBL), (2, Z0TBL), (3, LEMITBL), (4, PCTBL), (5, SHDTBL)] {
                    out.put(base + lc, real(row[column]));
                }
                out.put(IFORTBL + lc, int(row[6]));
                for (column, base) in [(7, RSTBL), (8, RGLTBL), (9, HSTBL), (10, SNUPTBL), (11, LAITBL),
                                       (12, MAXALB)] {
                    out.put(base + lc, real(row[column]));
                }
            }
            for (offset, integer) in [(TOPT, false), (CMCMAX, false), (CFACTR, false), (RSMAX, false),
                                      (BARE, true), (NATURAL, true), (CROP, true), (URBAN, true)] {
                let value = tokens(&lines[pos + 1])[0]; pos += 2;
                out.put(offset, if integer { int(value) } else { real(value) });
            }
            out.put(LUTYPE, if wanted == "USGS-RUC" { 4 } else { 5 });
            return;
        }
        loops += 1;
        loop {
            assert!(pos < lines.len(), "land use dataset '{wanted}' not found in VEGPARM.TBL");
            let line = &lines[pos]; pos += 1;
            if line.len() >= 21 && &line[..21] == "Vegetation Parameters" { break; }
            assert!(loops < 10, "Too many loops in VEGPARM.TBL");
        }
    }
}

// rucinit.F:380-430 (SOILPARM.TBL).
fn soil(dir: &Path, wanted: &str, out: &mut Layout) {
    let lines = read_lines(&dir.join("SOILPARM.TBL"));
    let mut pos = 0;
    loop {
        assert!(pos + 2 < lines.len(), "inconsistent or missing soilparm file");
        pos += 1; // read(16,*)
        let sltype = char8(&first_token(&lines[pos])); pos += 1;
        let slcats: usize = tokens(&lines[pos])[0].parse().unwrap(); pos += 1;
        if sltype == wanted {
            assert!(slcats <= NSLTYPE, "SOILPARM exceeds RUC NSLTYPE");
            out.put(SLCATS, slcats as u32);
            out.put(SLTYPE, 6);
            for lc in 0..slcats {
                let row = tokens(&lines[pos]); pos += 1;
                assert!(row.len() >= 11, "short SOILPARM row");
                for (column, base) in [(1, BB), (2, DRYSMC), (3, HC), (4, MAXSMC), (5, REFSMC), (6, SATPSI),
                                       (7, SATDK), (8, SATDW), (9, WLTSMC), (10, QTZ)] {
                    out.put(base + lc, real(row[column]));
                }
            }
            return;
        }
        pos += slcats;
    }
}

// rucinit.F:460-500 (GENPARM.TBL).
fn general(dir: &Path, out: &mut Layout) {
    let lines = read_lines(&dir.join("GENPARM.TBL"));
    let count: usize = tokens(&lines[2])[0].parse().unwrap();
    assert!(count <= NSLOPE, "NUM_SLOPE too large for slope_data array");
    out.put(SLPCATS, count as u32);
    for lc in 0..count { out.put(SLOPE + lc, real(tokens(&lines[3 + lc])[0])); }
    let mut pos = 3 + count;
    for offset in [SBETA, FXEXP, CSOIL, SALP, REFDK, REFKDT, FRZK, ZBOT, CZIL, SMLOW, SMHIGH] {
        out.put(offset, real(tokens(&lines[pos + 1])[0])); pos += 2;
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 4, "usage: ruc_table_decode TABLE_DIRECTORY LANDUSE_RUC SOIL_RUC");
    let dir = Path::new(&args[1]);
    let mut out = Layout { words: vec![0; WORDS] };
    vegetation(dir, &args[2], &mut out);
    soil(dir, &args[3], &mut out);
    general(dir, &mut out);
    let bytes: Vec<u8> = out.words.iter().flat_map(|w| w.to_le_bytes()).collect();
    use std::io::Write;
    std::io::stdout().lock().write_all(&bytes).unwrap();
}
