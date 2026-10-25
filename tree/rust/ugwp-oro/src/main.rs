//! rw_hex_ugwp_oro: the GSL orographic drag statistics on any hex mesh.
//!
//! THE BREAKAGE THIS PREVENTS: the RRFS-shape suite (rrfs_v2_dev, and every
//! selection of config_gwdo_scheme='bl_ugwp_gwdo') reads twenty per-cell
//! statistics from `<mesh>.ugwp_oro_data.nc`. RRFS publishes that file only
//! for its own meshes (conus3km, conus12km), so on every other hex mesh -- a
//! global, a zoom globe, a region, a point mesh, a storm-following member --
//! the forecast stopped before its first step ("--orographic-statistics names
//! a missing file", 3 km region from GFS 2026-10-23 18Z f348, box L,
//! 2026-10-24).
//!
//! Written from the published method, in our own code:
//!
//! * Kim & Arakawa (1995, J. Atmos. Sci. 52, 1875-1902): the sub-grid
//!   orography statistics of a grid box -- standard deviation, convexity
//!   OC = mean((h-hbar)^4)/sigma^4, asymmetry OA = (Nu-Nd)/(Nu+Nd) from the
//!   points above the box mean in the upstream and downstream halves, and the
//!   effective length OL as the fraction of points above the box mean -- in
//!   four flow directions (from the W, S, SW and NW).
//! * Kim & Doyle (2005, QJRMS 131, 1893-1921): the effective length and the
//!   low-level blocking the meso-scale ("ls") drag uses.
//! * Beljaars, Brown & Wood (2004, QJRMS 130, 1327-1347): the turbulent
//!   orographic form drag reads the standard deviation of a FILTERED
//!   orography that keeps only the small scales (below k_flt = 0.00035 rad/m,
//!   about 18 km, down to the ~2 km the 30-arc-second terrain resolves).
//! * Steeneveld et al. (2008) and Tsiringakis et al. (2017): the small-scale
//!   gravity-wave drag reads the same small-scale ("ss") statistics.
//! * The NOAA GSL drag suite documentation (CCPP scientific documentation):
//!   "ss" statistics from the filtered 30-arc-second terrain, "ls" statistics
//!   from 2.5-arc-minute terrain, both over the grid box; where the grid box
//!   is too small for the 2.5-minute terrain to describe, the "ls" fields are
//!   the "ss" fields (RRFS's conus3km file carries identical ls and ss fields).
//!
//! Concretely, per cell (identical rule on a uniform globe, a zoom globe and a
//! 3 km region):
//!
//! 1. The box: a lat/lon box centred on the cell, of side `side_m` (the cell's
//!    nominal spacing, MPAS's len_disp / meshDensity^(1/4)), as the nearest
//!    whole number of source pixels in latitude and in longitude.
//! 2. Filtered terrain: the 30" terrain minus its running mean over 20 km
//!    (Beljaars' k_flt scale); then, for "ss", a 1.5 km running mean applied
//!    twice (the scales the 30" data resolves without pixel noise); for "ls",
//!    a 3.5 km running mean applied twice, averaged into 2.5-arc-minute
//!    blocks.
//! 3. The ten statistics of each box (`statistics`).
//!
//! Validated against RRFS's own conus12km and conus3km files (see the hex
//! docs, physics-suites.md); the filter widths above are the ones that match
//! them.
//!
//! Inputs: `--cells FILE` (little-endian u64 n, then f64 latitude[n] and
//! longitude[n] in radians and f64 box side[n] in metres), `--topo DIR` (a
//! WPS_GEOG 30-arc-second terrain directory: `index` plus tiles). Output:
//! `--out FILE`, little-endian f32 [20][n] in the file's variable order (see
//! [`VARIABLES`]). A JSON receipt goes to stdout. Every stage runs on every
//! core (rayon).

use rayon::prelude::*;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Earth radius MPAS carries (m).
const RE: f64 = 6_371_229.0;
/// The cap the GSL statistics place on convexity (RRFS files top out at it).
const MAX_CONVEXITY: f64 = 10.0;
/// Source pixels per 2.5-arc-minute block of the 30-arc-second terrain.
const LS_BLOCK: i64 = 5;

/// Output order: the RRFS file's variable order.
pub const VARIABLES: [&str; 20] = [
    "var2dls", "conls", "oa1ls", "oa2ls", "oa3ls", "oa4ls", "ol1ls", "ol2ls", "ol3ls", "ol4ls",
    "var2dss", "conss", "oa1ss", "oa2ss", "oa3ss", "oa4ss", "ol1ss", "ol2ss", "ol3ss", "ol4ss",
];

fn die(msg: impl AsRef<str>) -> ! {
    eprintln!("rw_hex_ugwp_oro: {}", msg.as_ref());
    std::process::exit(2);
}

// ---------------------------------------------------------------------------
// WPS_GEOG terrain
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct GeogIndex {
    dx: f64,
    dy: f64,
    known_x: f64,
    known_y: f64,
    known_lat: f64,
    known_lon: f64,
    wordsize: usize,
    signed: bool,
    big_endian: bool,
    tile_x: usize,
    tile_y: usize,
    tile_bdr: usize,
    scale: f64,
    missing: Option<i64>,
}

fn parse_index(dir: &Path) -> GeogIndex {
    let text = fs::read_to_string(dir.join("index"))
        .unwrap_or_else(|e| die(format!("{}: cannot read index: {e}", dir.display())));
    let mut get = std::collections::HashMap::new();
    for line in text.lines() {
        if let Some((k, v)) = line.split_once('=') {
            get.insert(k.trim().to_ascii_lowercase(), v.trim().trim_matches('"').to_string());
        }
    }
    let num = |k: &str, d: Option<f64>| -> f64 {
        match get.get(k) {
            Some(v) => v.parse::<f64>().unwrap_or_else(|_| die(format!("index {k}={v} is not a number"))),
            None => d.unwrap_or_else(|| die(format!("{}/index has no {k}", dir.display()))),
        }
    };
    if get.get("projection").map(|s| s.as_str()) != Some("regular_ll") {
        die(format!("{}/index is not projection=regular_ll; the box arithmetic is lat/lon", dir.display()));
    }
    let idx = GeogIndex {
        dx: num("dx", None),
        dy: num("dy", None),
        known_x: num("known_x", Some(1.0)),
        known_y: num("known_y", Some(1.0)),
        known_lat: num("known_lat", None),
        known_lon: num("known_lon", None),
        wordsize: num("wordsize", None) as usize,
        signed: get.get("signed").map(|s| s == "yes").unwrap_or(false),
        big_endian: get.get("endian").map(|s| s != "little").unwrap_or(true),
        tile_x: num("tile_x", None) as usize,
        tile_y: num("tile_y", None) as usize,
        tile_bdr: num("tile_bdr", Some(0.0)) as usize,
        scale: num("scale_factor", Some(1.0)),
        missing: get.get("missing_value").and_then(|v| v.parse::<f64>().ok()).map(|v| v as i64),
    };
    // The method's filter widths and its 2.5' blocks are stated in 30" pixels.
    if (idx.dx * 120.0 - 1.0).abs() > 1e-3 || (idx.dy * 120.0 - 1.0).abs() > 1e-3 {
        die(format!(
            "{} is {} x {} degrees; the statistics are defined on 30-arc-second terrain \
             (topo_gmted2010_30s)",
            dir.display(),
            idx.dx,
            idx.dy
        ));
    }
    idx
}

/// A latitude band and circular longitude span of the 30" terrain in memory
/// (metres, f32). Global column `gx` (0-based) sits at window column
/// `(gx - x0) mod nx` when that is below `ncols`.
struct Terrain {
    idx: GeogIndex,
    nx: i64,
    y0: i64,
    x0: i64,
    nrows: i64,
    ncols: i64,
    h: Vec<f32>,
}

impl Terrain {
    /// Fractional 0-based global (x, y) of a point (degrees); integers are pixel centres.
    fn xy(idx: &GeogIndex, lat: f64, lon: f64) -> (f64, f64) {
        let x = (lon - idx.known_lon) / idx.dx + (idx.known_x - 1.0);
        let y = (lat - idx.known_lat) / idx.dy + (idx.known_y - 1.0);
        (x, y)
    }

    fn lat_of_row(&self, gy: i64) -> f64 {
        self.idx.known_lat + (gy as f64 - (self.idx.known_y - 1.0)) * self.idx.dy
    }

    fn with(&self, h: Vec<f32>) -> Terrain {
        Terrain { idx: self.idx.clone(), nx: self.nx, y0: self.y0, x0: self.x0, nrows: self.nrows, ncols: self.ncols, h }
    }

    #[inline]
    fn at(&self, gx: i64, gy: i64) -> f32 {
        let gy = gy.clamp(self.y0, self.y0 + self.nrows - 1);
        let wx = (gx - self.x0).rem_euclid(self.nx);
        if wx >= self.ncols {
            // Unreachable when the window was sized from the same boxes; a zero
            // here would silently flatten terrain, so it is loud.
            die(format!("terrain window miss at global column {gx}"));
        }
        self.h[((gy - self.y0) * self.ncols + wx) as usize]
    }

    fn load(dir: &Path, idx: GeogIndex, rows: (i64, i64), cols: Option<(i64, i64)>) -> (Terrain, usize) {
        let nx = (360.0 / idx.dx).round() as i64;
        let ny = (180.0 / idx.dy).round() as i64;
        let (y0, y1) = (rows.0.max(0), rows.1.min(ny - 1));
        let (x0, ncols) = match cols {
            Some((a, n)) => (a.rem_euclid(nx), n.min(nx)),
            None => (0, nx),
        };
        let nrows = y1 - y0 + 1;
        let mut h = vec![0.0f32; (nrows * ncols) as usize];
        let (tx, ty, b, ws) = (idx.tile_x as i64, idx.tile_y as i64, idx.tile_bdr as i64, idx.wordsize);
        let tiles_x = (nx + tx - 1) / tx;
        // One band of tile rows per task: each writes only its own rows.
        let mut chunks: Vec<(i64, &mut [f32])> = Vec::new();
        {
            let mut rest: &mut [f32] = &mut h;
            let mut gy = y0;
            while gy <= y1 {
                let end = (((gy / ty) + 1) * ty - 1).min(y1);
                let (a, r) = rest.split_at_mut(((end - gy + 1) * ncols) as usize);
                chunks.push((gy, a));
                rest = r;
                gy = end + 1;
            }
        }
        let read: usize = chunks
            .into_par_iter()
            .map(|(gy_start, buf)| {
                let tile_row = gy_start / ty;
                let rows_here = buf.len() as i64 / ncols;
                let mut raw = Vec::new();
                let mut tiles = 0usize;
                for tcol in 0..tiles_x {
                    let gx_a = tcol * tx;
                    let gx_b = (gx_a + tx).min(nx) - 1;
                    let hits = (gx_a - x0).rem_euclid(nx) < ncols || (x0 - gx_a).rem_euclid(nx) <= gx_b - gx_a;
                    if !hits {
                        continue;
                    }
                    let name = format!("{:05}-{:05}.{:05}-{:05}", gx_a + 1, gx_a + tx, tile_row * ty + 1, tile_row * ty + ty);
                    let path = dir.join(&name);
                    raw.clear();
                    match fs::File::open(&path) {
                        Ok(mut f) => {
                            f.read_to_end(&mut raw).unwrap_or_else(|e| die(format!("{}: {e}", path.display())));
                        }
                        Err(_) => continue, // WPS leaves all-ocean tiles out: zero terrain
                    }
                    tiles += 1;
                    let wtx = (tx + 2 * b) as usize;
                    let need = wtx * (ty + 2 * b) as usize * ws;
                    if raw.len() < need {
                        die(format!("{}: {} bytes, expected {need}", path.display(), raw.len()));
                    }
                    for r in 0..rows_here {
                        let ty_in = (gy_start + r - tile_row * ty + b) as usize;
                        for gx in gx_a..=gx_b {
                            let wx = (gx - x0).rem_euclid(nx);
                            if wx >= ncols {
                                continue;
                            }
                            let k = (ty_in * wtx + (gx - gx_a + b) as usize) * ws;
                            let v = decode(&raw[k..k + ws], idx.signed, idx.big_endian);
                            let m = if Some(v) == idx.missing { 0.0 } else { v as f64 * idx.scale };
                            buf[(r * ncols + wx) as usize] = m as f32;
                        }
                    }
                }
                tiles
            })
            .sum();
        (Terrain { idx, nx, y0, x0, nrows, ncols, h }, read)
    }
}

#[inline]
fn decode(b: &[u8], signed: bool, big: bool) -> i64 {
    let mut v: u64 = 0;
    if big {
        for &x in b {
            v = (v << 8) | x as u64;
        }
    } else {
        for &x in b.iter().rev() {
            v = (v << 8) | x as u64;
        }
    }
    let bits = 8 * b.len() as u32;
    if signed && bits < 64 && (v >> (bits - 1)) & 1 == 1 {
        (v as i64) - (1i64 << bits)
    } else {
        v as i64
    }
}

/// Running mean of `t` over a `width_km` square, `passes` times (repeated
/// box means tend to a Gaussian). Rows, then columns, each on its own core;
/// the longitude half-width widens with 1/cos(latitude).
fn smooth(t: &Terrain, width_km: f64, passes: usize) -> Vec<f32> {
    let (nr, nc) = (t.nrows as usize, t.ncols as usize);
    let step_m = t.idx.dy.to_radians() * RE;
    let half_m = width_km * 1000.0 / 2.0;
    let ky = (half_m / step_m).round() as i64;
    let wrap = nc as i64 == t.nx;
    let mut s: Vec<f32> = t.h.clone();
    let mut col_buf: Vec<f32> = vec![0.0; nr * nc];
    for _ in 0..passes.max(1) {
        s.par_chunks_mut(nc).enumerate().for_each(|(r, row)| {
            let lat = t.lat_of_row(t.y0 + r as i64).to_radians();
            let kx = ((half_m / (step_m * lat.cos().max(0.02))).round() as i64).min((nc as i64 - 1) / 2);
            let mut pre = vec![0.0f64; nc + 1];
            for i in 0..nc {
                pre[i + 1] = pre[i] + row[i] as f64;
            }
            let total = pre[nc];
            let n = nc as i64;
            for i in 0..n {
                let (a, b) = (i - kx, i + kx);
                row[i as usize] = if wrap {
                    let (lo, hi) = (a.rem_euclid(n), b.rem_euclid(n));
                    let sum = if lo <= hi {
                        pre[(hi + 1) as usize] - pre[lo as usize]
                    } else {
                        (total - pre[lo as usize]) + pre[(hi + 1) as usize]
                    };
                    (sum / (b - a + 1) as f64) as f32
                } else {
                    let (lo, hi) = (a.max(0) as usize, b.min(n - 1) as usize);
                    ((pre[hi + 1] - pre[lo]) / (hi - lo + 1) as f64) as f32
                };
            }
        });
        // Columns: each task owns a strip of columns and writes it transposed
        // into col_buf (column-major); rows are then filled back in parallel.
        let strip = 64usize;
        col_buf.par_chunks_mut(strip * nr).enumerate().for_each(|(k, out)| {
            let c0 = k * strip;
            let w = out.len() / nr;
            let mut pre = vec![0.0f64; nr + 1];
            for dc in 0..w {
                let i = c0 + dc;
                for r in 0..nr {
                    pre[r + 1] = pre[r] + s[r * nc + i] as f64;
                }
                for r in 0..nr as i64 {
                    let lo = (r - ky).max(0) as usize;
                    let hi = (r + ky).min(nr as i64 - 1) as usize;
                    out[dc * nr + r as usize] = ((pre[hi + 1] - pre[lo]) / (hi - lo + 1) as f64) as f32;
                }
            }
        });
        s.par_chunks_mut(nc).enumerate().for_each(|(r, row)| {
            for (i, v) in row.iter_mut().enumerate() {
                *v = col_buf[i * nr + r];
            }
        });
    }
    s
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// The ten statistics of one box of `nx` x `ny` values (row-major from the
/// south-west): sigma, convexity, OA for flow from W/S/SW/NW, OL for the same.
fn statistics(nx: i64, ny: i64, h: &[f64]) -> [f64; 10] {
    let n = (nx * ny) as f64;
    let mean = h.iter().sum::<f64>() / n;
    let (mut s2, mut s4) = (0.0f64, 0.0f64);
    for &v in h {
        let d = v - mean;
        s2 += d * d;
        s4 += d * d * d * d;
    }
    let var = s2 / n;
    let mut out = [0.0f64; 10];
    out[0] = var.sqrt();
    if !(var > 0.0) {
        return out; // a flat box (open water): every statistic is zero
    }
    out[1] = (s4 / (n * var * var)).min(MAX_CONVEXITY);
    let get = |i: i64, j: i64| h[((j - 1) * nx + (i - 1)) as usize];
    // Upstream (+1) / downstream (-1) halves for flow from W, S, SW, NW,
    // split through the box centre in coordinates normalised by the box's own
    // sides, so a box of any aspect splits into equal halves; a pixel on the
    // split line belongs to neither.
    let side = |i: i64, j: i64, k: usize| -> i32 {
        let u = (2 * i - (nx + 1)) as f64 / nx as f64;
        let v = (2 * j - (ny + 1)) as f64 / ny as f64;
        let d = match k {
            0 => -u,
            1 => -v,
            2 => -(u + v),
            _ => v - u,
        };
        if d > 1e-12 {
            1
        } else if d < -1e-12 {
            -1
        } else {
            0
        }
    };
    for k in 0..4 {
        let (mut nu, mut nd) = (0i64, 0i64);
        for j in 1..=ny {
            for i in 1..=nx {
                if get(i, j) > mean {
                    match side(i, j, k) {
                        1 => nu += 1,
                        -1 => nd += 1,
                        _ => {}
                    }
                }
            }
        }
        out[2 + k] = if nu + nd > 0 { (nu - nd) as f64 / (nu + nd) as f64 } else { 0.0 };
    }
    // Effective length: fraction above the box mean in the middle half of the
    // rows (W), of the columns (S), and in the two quarter boxes each diagonal
    // flow crosses (SW: SW+NE, NW: NW+SE). Integer halving as written, which
    // is what the point counts of RRFS's files show (7 of 13 rows, 9 of 17
    // columns at 12 km; 2 of 3 rows at 3 km).
    let band = |i: i64, j: i64, k: usize| -> bool {
        match k {
            0 => j >= ny / 4 && j <= 3 * ny / 4,
            1 => i >= nx / 4 && i <= 3 * nx / 4,
            2 => (j <= ny / 2 && i <= nx / 2) || (j > ny / 2 && i > nx / 2),
            _ => (j > ny / 2 && i <= nx / 2) || (j <= ny / 2 && i > nx / 2),
        }
    };
    for k in 0..4 {
        let (mut nw, mut nt) = (0i64, 0i64);
        for j in 1..=ny {
            for i in 1..=nx {
                if band(i, j, k) {
                    nt += 1;
                    if get(i, j) > mean {
                        nw += 1;
                    }
                }
            }
        }
        out[6 + k] = if nt > 0 { nw as f64 / nt as f64 } else { 0.0 };
    }
    out
}

/// Box size in values for a box of `side_m` at latitude `lat` (rad) on a grid
/// of `step_deg`: the nearest whole number, at least one, never past a whole
/// circle of longitude.
fn box_size(lat: f64, side_m: f64, step_deg: f64, nx_global: i64) -> (i64, i64) {
    let deg = (side_m / RE).to_degrees();
    let ny = (deg / step_deg).round().max(1.0) as i64;
    let nx = ((deg / lat.cos().max(1e-6)) / step_deg).round().max(1.0) as i64;
    (nx.min(nx_global), ny)
}

/// Gather one cell's box from `t`, averaging `block` x `block` pixels per
/// value (1 = the 30" pixels), centred on the block holding the cell centre.
fn gather(t: &Terrain, lat: f64, lon: f64, side_m: f64, block: i64, out: &mut Vec<f64>) -> (i64, i64) {
    let (nx, ny) = box_size(lat, side_m, t.idx.dy * block as f64, t.nx / block);
    let (x, y) = Terrain::xy(&t.idx, lat.to_degrees(), lon.to_degrees());
    let bx = ((x + 0.5) / block as f64).floor() as i64;
    let by = ((y + 0.5) / block as f64).floor() as i64;
    let (i0, j0) = (bx - (nx - 1) / 2, by - (ny - 1) / 2);
    out.clear();
    let inv = 1.0 / (block * block) as f64;
    for j in 0..ny {
        for i in 0..nx {
            let (cx, cy) = (i0 + i, j0 + j);
            let mut s = 0.0f64;
            for q in 0..block {
                for p in 0..block {
                    s += t.at(cx * block + p, cy * block + q) as f64;
                }
            }
            out.push(s * inv);
        }
    }
    (nx, ny)
}

// ---------------------------------------------------------------------------

fn read_cells(path: &Path) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let raw = fs::read(path).unwrap_or_else(|e| die(format!("{}: {e}", path.display())));
    if raw.len() < 8 {
        die("cells file shorter than its header");
    }
    let n = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
    if raw.len() != 8 + 24 * n {
        die(format!("cells file is {} bytes; {n} cells need {}", raw.len(), 8 + 24 * n));
    }
    let f = |k: usize| -> Vec<f64> {
        raw[8 + k * n * 8..8 + (k + 1) * n * 8]
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    (f(0), f(1), f(2))
}

struct Method {
    highpass_km: f64,
    ss_smooth_km: f64,
    ls_smooth_km: f64,
    smooth_passes: usize,
}

fn main() {
    let t_start = Instant::now();
    let (mut cells, mut topo, mut out): (Option<PathBuf>, Option<PathBuf>, Option<PathBuf>) = (None, None, None);
    let mut m = Method { highpass_km: 20.0, ss_smooth_km: 1.5, ls_smooth_km: 3.5, smooth_passes: 2 };
    let mut threads: Option<usize> = None;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut k = 0;
    while k < args.len() {
        let a = args[k].clone();
        let mut val = || {
            k += 1;
            args.get(k).cloned().unwrap_or_else(|| die(format!("{a} needs a value")))
        };
        let num = |s: String| -> f64 { s.parse().unwrap_or_else(|_| die(format!("{s} is not a number"))) };
        match a.as_str() {
            "--cells" => cells = Some(val().into()),
            "--topo" => topo = Some(val().into()),
            "--out" => out = Some(val().into()),
            "--threads" => threads = Some(num(val()) as usize),
            "--highpass-km" => m.highpass_km = num(val()),
            "--ss-smooth-km" => m.ss_smooth_km = num(val()),
            "--ls-smooth-km" => m.ls_smooth_km = num(val()),
            "--smooth-passes" => m.smooth_passes = num(val()) as usize,
            "--help" | "-h" => {
                println!(
                    "rw_hex_ugwp_oro --cells FILE --topo WPS_GEOG/topo_gmted2010_30s --out FILE [--threads N]\n\
                     cells: u64 n, f64 lat[n] (rad), f64 lon[n] (rad), f64 side[n] (m), little-endian\n\
                     out:   f32 [20][n] in the order {}",
                    VARIABLES.join(",")
                );
                return;
            }
            other => die(format!("unknown argument {other}")),
        }
        k += 1;
    }
    let cells = cells.unwrap_or_else(|| die("--cells is required"));
    let topo = topo.unwrap_or_else(|| die("--topo is required"));
    let out = out.unwrap_or_else(|| die("--out is required"));
    if let Some(n) = threads {
        rayon::ThreadPoolBuilder::new().num_threads(n).build_global().ok();
    }
    let (lat, lon, side) = read_cells(&cells);
    let n = lat.len();
    if n == 0 {
        die("no cells");
    }
    if let Some(c) = (0..n).find(|&c| !(side[c] > 0.0) || !lat[c].is_finite() || !lon[c].is_finite()) {
        die(format!("cell {c} has latitude {} longitude {} side {}", lat[c], lon[c], side[c]));
    }

    // The window every box reads, plus the filters' reach.
    let idx = parse_index(&topo);
    let nxg = (360.0 / idx.dx).round() as i64;
    let step_m = idx.dy.to_radians() * RE;
    let reach_px = ((m.highpass_km + 2.0 * m.smooth_passes as f64 * m.ls_smooth_km.max(m.ss_smooth_km)) * 1000.0
        / 2.0
        / step_m)
        .ceil() as i64
        + 2;
    let mut need_col = vec![false; nxg as usize];
    let (mut ylo, mut yhi) = (i64::MAX, i64::MIN);
    for c in 0..n {
        let (x, y) = Terrain::xy(&idx, lat[c].to_degrees(), lon[c].to_degrees());
        let (nxb, nyb) = box_size(lat[c], side[c], idx.dy * LS_BLOCK as f64, nxg / LS_BLOCK);
        let ry = (nyb / 2 + 2) * LS_BLOCK + reach_px;
        let rx = (nxb / 2 + 2) * LS_BLOCK + (reach_px as f64 / lat[c].cos().max(0.02)) as i64;
        ylo = ylo.min(y.floor() as i64 - ry);
        yhi = yhi.max(y.ceil() as i64 + ry);
        if 2 * rx + 1 >= nxg {
            need_col.iter_mut().for_each(|v| *v = true);
        } else {
            for gx in (x.floor() as i64 - rx)..=(x.ceil() as i64 + rx) {
                need_col[gx.rem_euclid(nxg) as usize] = true;
            }
        }
    }
    // Smallest circular span holding every needed column: drop the longest gap.
    let cols = if need_col.iter().all(|&v| v) {
        None
    } else {
        let (mut best_len, mut best_end, mut run) = (0i64, 0i64, 0i64);
        for k in 0..(2 * nxg) {
            if !need_col[(k % nxg) as usize] {
                run += 1;
                if run > best_len && run <= nxg {
                    best_len = run;
                    best_end = k;
                }
            } else {
                run = 0;
            }
        }
        Some(((best_end + 1).rem_euclid(nxg), nxg - best_len))
    };
    let t_load = Instant::now();
    let (raw, tiles) = Terrain::load(&topo, idx, (ylo, yhi), cols);
    let load_s = t_load.elapsed().as_secs_f64();

    // Filtered terrain: the raw field minus its 20 km running mean, then the
    // ss and ls smoothings. The raw field is dropped as soon as it is used.
    let t_filter = Instant::now();
    let large = smooth(&raw, m.highpass_km, 1);
    let hp: Vec<f32> = raw.h.par_iter().zip(large.par_iter()).map(|(a, b)| a - b).collect();
    drop(large);
    let hp = raw.with(hp);
    drop(raw);
    let ss = hp.with(smooth(&hp, m.ss_smooth_km, m.smooth_passes));
    let ls = hp.with(smooth(&hp, m.ls_smooth_km, m.smooth_passes));
    drop(hp);
    let filter_s = t_filter.elapsed().as_secs_f64();

    // ls is ss where the box is under two 2.5' blocks tall.
    let ls_min_side_m = 1.5 * LS_BLOCK as f64 * step_m;
    let t_stats = Instant::now();
    let results: Vec<([f64; 10], [f64; 10], bool)> = (0..n)
        .into_par_iter()
        .map_init(Vec::new, |buf, c| {
            let (nx, ny) = gather(&ss, lat[c], lon[c], side[c], 1, buf);
            let s = statistics(nx, ny, buf);
            if side[c] < ls_min_side_m {
                return (s, s, true);
            }
            let (nx, ny) = gather(&ls, lat[c], lon[c], side[c], LS_BLOCK, buf);
            (statistics(nx, ny, buf), s, false)
        })
        .collect();
    let stats_s = t_stats.elapsed().as_secs_f64();

    let mut bytes = vec![0u8; 20 * n * 4];
    bytes.par_chunks_mut(n * 4).enumerate().for_each(|(v, chunk)| {
        for (c, r) in results.iter().enumerate() {
            let x = if v < 10 { r.0[v] } else { r.1[v - 10] };
            chunk[c * 4..c * 4 + 4].copy_from_slice(&(x as f32).to_le_bytes());
        }
    });
    fs::write(&out, &bytes).unwrap_or_else(|e| die(format!("{}: {e}", out.display())));
    let dup = results.iter().filter(|r| r.2).count();
    let receipt = serde_json::json!({
        "schema": "gpuwm-hex.ugwp-oro-stats/v1",
        "cells": n,
        "variables": VARIABLES,
        "terrain": topo.file_name().map(|s| s.to_string_lossy().into_owned()),
        "terrain_window": {"rows": ss.nrows, "cols": ss.ncols, "first_row": ss.y0, "first_col": ss.x0, "tiles_read": tiles},
        "method": {
            "box": "lat/lon box centred on the cell, side = the cell's nominal spacing, nearest whole pixels",
            "filtered_terrain": format!("30-arc-second terrain minus its {} km running mean", m.highpass_km),
            "ss": format!("filtered terrain, {} km running mean x{}, 30-arc-second pixels", m.ss_smooth_km, m.smooth_passes),
            "ls": format!("filtered terrain, {} km running mean x{}, 2.5-arc-minute block means", m.ls_smooth_km, m.smooth_passes),
            "ls_is_ss_below_side_m": ls_min_side_m,
            "reference_height": "box mean",
            "max_convexity": MAX_CONVEXITY,
        },
        "cells_ls_from_ss": dup,
        "threads": rayon::current_num_threads(),
        "seconds": {"terrain_load": load_s, "filter": filter_s, "statistics": stats_s, "total": t_start.elapsed().as_secs_f64()},
    });
    println!("{receipt}");
}
