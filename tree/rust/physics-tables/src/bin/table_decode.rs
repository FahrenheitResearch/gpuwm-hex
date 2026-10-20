//! Rust input bridge for MPAS Noah's four pinned text tables.
//! MPAS v8.4.2 mpas_atmphys_lsm_noahinit.F:210-597 and mpas_atmphys_landuse.F:177-227.
//! Output is the little-endian NoahTables ABI followed by LANDUSE's seven
//! category-major arrays. Numeric file decoding belongs to Rust.
use std::{env, fs, path::Path};
include!("table_schema.rs");

fn tokens(line: &str) -> Vec<&str> {
    line.split(['\'', '"']).next().unwrap().split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty()).collect()
}
fn read_lines(path: &Path) -> Vec<String> {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines().map(str::to_owned).collect()
}
fn store(data: &mut [u8], name: &str, index: usize, value: &str) {
    let (offset, count, kind) = field(name);
    assert!(index < count, "{name} table capacity exceeded");
    let bytes = if kind == 'i' {
        value.parse::<i32>().unwrap().to_le_bytes()
    } else {
        assert_eq!(kind, 'f');
        value.replace(['d','D'], "E").parse::<f32>().unwrap().to_le_bytes()
    };
    data[offset + index*4..offset + index*4 + 4].copy_from_slice(&bytes);
}
fn text(data: &mut [u8], name: &str, value: &str) {
    let (offset, count, kind) = field(name);
    assert_eq!(kind, 'c'); assert!(value.len() <= count);
    data[offset..offset+count].fill(b' ');
    data[offset..offset+value.len()].copy_from_slice(value.as_bytes());
}

// MPAS v8.4.2 mpas_atmphys_lsm_noahinit.F:256-419.
fn vegetation(dir: &Path, wanted: &str, data: &mut [u8]) {
    let lines = read_lines(&dir.join("VEGPARM.TBL"));
    let at = lines.windows(2).position(|w| w[0].starts_with("Vegetation Parameters")
        && w[1].trim() == wanted).expect("land use dataset absent from VEGPARM.TBL");
    let count: usize = tokens(&lines[at+2])[0].parse().unwrap();
    assert!(count <= 50, "VEGPARM exceeds MPAS NLUS");
    text(data, "LUTYPE", wanted); store(data, "LUCATS", 0, &count.to_string());
    let names = ["SHDTBL","NROTBL","RSTBL","RGLTBL","HSTBL","SNUPTBL","MAXALB",
        "LAIMINTBL","LAIMAXTBL","EMISSMINTBL","EMISSMAXTBL","ALBEDOMINTBL",
        "ALBEDOMAXTBL","Z0MINTBL","Z0MAXTBL","ZTOPVTBL","ZBOTVTBL"];
    for i in 0..count {
        let row = tokens(&lines[at+3+i]); assert!(row.len() >= 18);
        // MPAS reads the row index, but stores rows in file order.
        for (j, name) in names.iter().enumerate() {store(data, name, i, row[j+1]);}
    }
    let mut pos = at+3+count;
    for name in ["TOPT_DATA","CMCMAX_DATA","CFACTR_DATA","RSMAX_DATA","BARE","NATURAL"] {
        store(data, name, 0, tokens(&lines[pos+1])[0]); pos += 2;
    }
    pos += 2; // CROP's label and value are discarded by the native decoder.
    for i in 1..=11 {store(data, &format!("LCZ_{i}"), 0, tokens(&lines[pos+1])[0]); pos += 2;}
}

// MPAS v8.4.2 mpas_atmphys_lsm_noahinit.F:445-485.
fn soil(dir: &Path, wanted: &str, data: &mut [u8]) {
    let lines = read_lines(&dir.join("SOILPARM.TBL"));
    // Native uses format(A4) on the FIRST soil block; it does not search blocks.
    let soil_type: String = lines[1].chars().take(4).collect();
    assert_eq!(soil_type.trim(), wanted, "soilparm classification differs from native A4 block");
    let count: usize = tokens(&lines[2])[0].parse().unwrap(); assert!(count <= 30);
    text(data, "SLTYPE", soil_type.trim()); store(data, "SLCATS", 0, &count.to_string());
    let names = ["BB","DRYSMC","F11","MAXSMC","REFSMC","SATPSI","SATDK","SATDW","WLTSMC","QTZ"];
    for i in 0..count {
        let row = tokens(&lines[3+i]); assert!(row.len() >= 11);
        for (j, name) in names.iter().enumerate() {store(data, name, i, row[j+1]);}
    }
}

// MPAS v8.4.2 mpas_atmphys_lsm_noahinit.F:535-584.
fn general(dir: &Path, data: &mut [u8]) {
    let lines = read_lines(&dir.join("GENPARM.TBL"));
    let count: usize = tokens(&lines[2])[0].parse().unwrap(); assert!(count <= 30);
    store(data, "SLPCATS", 0, &count.to_string());
    for i in 0..count {store(data, "SLOPE_DATA", i, tokens(&lines[3+i])[0]);}
    let mut pos = 3+count;
    for name in ["SBETA_DATA","FXEXP_DATA","CSOIL_DATA","SALP_DATA","REFDK_DATA",
        "REFKDT_DATA","FRZK_DATA","ZBOT_DATA","CZIL_DATA","SMLOW_DATA","SMHIGH_DATA","LVCOEF_DATA"] {
        store(data, name, 0, tokens(&lines[pos+1])[0]); pos += 2;
    }
}

// MPAS v8.4.2 mpas_atmphys_landuse.F:190-227.
fn landuse(dir: &Path, wanted: &str) -> Vec<u8> {
    let lines = read_lines(&dir.join("LANDUSE.TBL")); let mut pos = 0;
    let mut output = vec![0; 7*12*100*4];
    while pos < lines.len() {
        let kind: String = lines[pos].chars().take(35).collect();
        let header = tokens(&lines[pos+1]);
        let cats: usize = header[0].parse().unwrap(); let seasons: usize = header[1].parse().unwrap();
        pos += 2;
        if kind.trim() != wanted {pos += seasons*(cats+1); continue;}
        assert!(cats <= 100 && seasons <= 12, "LANDUSE exceeds native array capacities");
        for season in 0..seasons {
            pos += 1;
            for cat in 0..cats {
                let row = tokens(&lines[pos]); assert!(row.len() >= 8); pos += 1;
                for var in 0..7 {
                    let offset = ((var*12+season)*100+cat)*4;
                    output[offset..offset+4].copy_from_slice(&row[var+1].replace(['d','D'], "E")
                        .parse::<f32>().unwrap().to_le_bytes());
                }
            }
        }
        return output;
    }
    panic!("land use dataset absent from LANDUSE.TBL");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 4, "usage: noah-table-decode TABLE_DIRECTORY LANDUSE SOIL_TYPE");
    let dir = Path::new(&args[1]); let mut data = defaults();
    vegetation(dir, &args[2], &mut data); soil(dir, &args[3], &mut data); general(dir, &mut data);
    data.extend(landuse(dir, &args[2]));
    use std::io::Write;
    std::io::stdout().lock().write_all(&data).unwrap();
}
