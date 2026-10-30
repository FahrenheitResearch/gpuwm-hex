//! TEMPO v3 module words tempo_init derives from its constants, without a native library.
//! TEMPO 9adf9ef module_mp_tempo_params.F90: initialize_graupel_vars (410-425), initialize_parameters (428-608),
//! initialize_bins_for_tables (611-656), initialize_bins_for_hail_size (659-669), initialize_bins_for_radar
//! (672-690), create_bins (693-720). Kinds and operation order follow the strict gfortran build. Expressions the
//! compiler folds from PARAMETERs alone are evaluated as it folds them (correctly rounded: sc**(1/3), d0s**bm_s,
//! d0g**bm_g, (2.5e-6)**3); expressions with a module variable run as binary32 or binary64 IEEE operations and
//! libm calls (powf; log, exp, sqrt for the bins). Gamma is the project's own correctly rounded tgammaf
//! (../own_gamma.rs), never the host C library's: glibc 2.39's differs from the reference host's on
//! gamma(4), gamma(2), ... and moved every table built from them.
//!
//! Input (stdin, little-endian): i32 flags hail melt hail_size; f32 scalars (SCALARS below, in order);
//! f32 arrays rho_g[9] av_g[9] bv_g[9] (declaration values) r_c[37] r_i[64] r_r[37] r_s[37] r_g[37] nt_i[55]
//! n0r_exp[37] n0g_exp[37] nt_in[55].
//! Output (stdout, little-endian, each table in its native kind, Fortran order): the OUTPUT list below.
use std::io::{Read, Write};

#[path = "../own_gamma.rs"]
mod own_gamma;

extern "C" {
    fn powf(x: f32, y: f32) -> f32;
    fn log(x: f64) -> f64;
    fn exp(x: f64) -> f64;
}

const NRHG: usize = 9;

struct Reader { bytes: Vec<u8>, at: usize }
impl Reader {
    fn i32(&mut self) -> i32 { let v = i32::from_le_bytes(self.bytes[self.at..self.at+4].try_into().unwrap()); self.at += 4; v }
    fn f32(&mut self) -> f32 { let v = f32::from_le_bytes(self.bytes[self.at..self.at+4].try_into().unwrap()); self.at += 4; v }
    fn f32s(&mut self, n: usize) -> Vec<f32> { (0..n).map(|_| self.f32()).collect() }
}

struct Out { bytes: Vec<u8> }
impl Out {
    fn f32(&mut self, v: f32) { self.bytes.extend_from_slice(&v.to_le_bytes()); }
    fn f32s(&mut self, v: &[f32]) { for x in v { self.f32(*x); } }
    fn i32(&mut self, v: i32) { self.bytes.extend_from_slice(&v.to_le_bytes()); }
    fn f64s(&mut self, v: &[f64]) { for x in v { self.bytes.extend_from_slice(&x.to_le_bytes()); } }
}

fn gamma(x: f32) -> f32 { own_gamma::tgammaf(x) }
fn pw(x: f32, y: f32) -> f32 { unsafe { powf(x, y) } }
// A PARAMETER-only power, folded by the compiler with MPFR: the correctly rounded binary32 result.
fn folded_pow(x: f32, y: f32) -> f32 { (x as f64).powf(y as f64) as f32 }
fn folded_powi(x: f32, n: i32) -> f32 { (x as f64).powi(n) as f32 }

// params.F90:693-720.
fn create_bins(numbins: usize, lowbin: f64, highbin: f64) -> (Vec<f64>, Vec<f64>) {
    let mut xdx = vec![0f64; numbins + 1];
    xdx[0] = lowbin;
    xdx[numbins] = highbin;
    for n in 2..=numbins {
        let a = (n - 1) as f64 / numbins as f64;
        xdx[n - 1] = unsafe { exp(a * log(xdx[numbins] / xdx[0]) + log(xdx[0])) };
    }
    let bins = (0..numbins).map(|n| (xdx[n] * xdx[n + 1]).sqrt()).collect();
    let deltas = (0..numbins).map(|n| xdx[n + 1] - xdx[n]).collect();
    (bins, deltas)
}

fn nint_log10(x: f32) -> i32 { (x as f64).log10().round() as i32 }

fn main() {
    let mut bytes = Vec::new();
    std::io::stdin().read_to_end(&mut bytes).unwrap();
    let mut r = Reader { bytes, at: 0 };
    let (hail, melt, hail_size) = (r.i32() != 0, r.i32() != 0, r.i32() != 0);
    let [av_r, bv_r, av_s, bv_s, bv_c, bv_i, av_g_old, bv_g_old, av_g_new, bv_g_new, am_s, bm_s, bm_g, bm_i, bm_r,
         rho_i, xm0i, d0c, d0r, d0s, d0g, c_cube, c_sqrd, mu_r, mu_s, mu_g, mu_i, rho_w, sc, pi, lsub, lvap0, rv,
         rdry, t0]: [f32; 35] = r.f32s(35).try_into().unwrap();
    let rho_g = r.f32s(NRHG);
    let mut av_g = r.f32s(NRHG);
    let mut bv_g = r.f32s(NRHG);
    let r_c = r.f32s(37); let r_i = r.f32s(64); let r_r = r.f32s(37); let r_s = r.f32s(37); let r_g = r.f32s(37);
    let nt_i = r.f32s(55); let n0r_exp = r.f32s(37); let n0g_exp = r.f32s(37); let nt_in = r.f32s(55);
    assert_eq!(r.at, r.bytes.len(), "input words differ from the declared constants");
    const IDX_BG1: usize = 6;

    // initialize_graupel_vars.
    let dim_nrhg: i32;
    if hail { av_g[IDX_BG1 - 1] = av_g_new; bv_g[IDX_BG1 - 1] = bv_g_new; dim_nrhg = NRHG as i32; }
    else { av_g[IDX_BG1 - 1] = av_g_old; bv_g[IDX_BG1 - 1] = bv_g_old; dim_nrhg = 1; }

    // initialize_parameters (pi, lsub, lvap0, rv, rdry are module variables: run-time operations).
    let am_i = pi * rho_i / 6.0;
    let am_r = pi * rho_w / 6.0;
    let am_g: Vec<f32> = rho_g.iter().map(|rg| pi * rg / 6.0).collect();
    let ar_volume = 4.0f32 / 3.0f32 * pi * folded_powi(2.5e-6, 3);
    let lfus = lsub - lvap0;
    let olfus = 1.0f32 / lfus;
    let orv = 1.0f32 / rv;
    let rho_not = 101325.0f32 / (rdry * 298.0);
    let rho_not0 = 101325.0f32 / (rdry * t0);
    let sc3 = folded_pow(sc, 1.0f32 / 3.0f32);
    let d0i = pw(xm0i / am_i, 1.0f32 / bm_i);
    let xm0s = am_s * folded_pow(d0s, bm_s);
    let xm0g = am_g[NRHG - 1] * folded_pow(d0g, bm_g);
    let obmi = 1.0f32 / bm_i;
    let obmr = 1.0f32 / bm_r;
    let oams = 1.0f32 / am_s;
    let obms = 1.0f32 / bm_s;
    let ocms = pw(oams, obms);
    let obmg = 1.0f32 / bm_g;
    let oamg: Vec<f32> = am_g.iter().map(|a| 1.0f32 / a).collect();
    let ocmg: Vec<f32> = oamg.iter().map(|o| pw(*o, obmg)).collect();
    let mut cce = vec![0f32; 5 * 15]; let mut ccg = vec![0f32; 5 * 15];
    let mut ocg1 = vec![0f32; 15]; let mut ocg2 = vec![0f32; 15];
    for n in 1..=15usize {
        let nf = n as f32;
        let col = (n - 1) * 5;
        cce[col] = nf + 1.0;
        cce[col + 1] = bm_r + nf + 1.0;
        cce[col + 2] = bm_r + nf + 4.0;
        cce[col + 3] = nf + bv_c + 1.0;
        cce[col + 4] = bm_r + nf + bv_c + 1.0;
        for m in 0..5 { ccg[col + m] = gamma(cce[col + m]); }
        ocg1[n - 1] = 1.0f32 / ccg[col];
        ocg2[n - 1] = 1.0f32 / ccg[col + 1];
    }
    let cie = [mu_i + 1.0, bm_i + mu_i + 1.0, bm_i + mu_i + bv_i + 1.0, mu_i + bv_i + 1.0, mu_i + 2.0,
               bm_i * 0.5 + mu_i + bv_i + 1.0, bm_i * 0.5 + mu_i + 1.0];
    let cig: Vec<f32> = cie.iter().map(|x| gamma(*x)).collect();
    let oig1 = 1.0f32 / cig[0];
    let oig2 = 1.0f32 / cig[1];
    let cre = [bm_r + 1.0, mu_r + 1.0, bm_r + mu_r + 1.0, bm_r * 2.0 + mu_r + 1.0, mu_r + bv_r + 1.0,
               bm_r + mu_r + bv_r + 1.0, bm_r * 0.5 + mu_r + bv_r + 1.0, bm_r + mu_r + bv_r + 3.0,
               mu_r + bv_r + 3.0, mu_r + 2.0, 0.5 * (bv_r + 5.0 + 2.0 * mu_r), bm_r * 0.5 + mu_r + 1.0,
               bm_r * 2.0 + mu_r + bv_r + 1.0];
    let crg: Vec<f32> = cre.iter().map(|x| gamma(*x)).collect();
    let ore1 = 1.0f32 / cre[0];
    let (org1, org2, org3) = (1.0f32 / crg[0], 1.0f32 / crg[1], 1.0f32 / crg[2]);
    let cse = [bm_s + 1.0, bm_s + 2.0, bm_s * 2.0, bm_s + bv_s + 1.0, bm_s * 2.0 + bv_s + 1.0, bm_s * 2.0 + 1.0,
               bm_s + mu_s + 1.0, bm_s + mu_s + 2.0, bm_s + mu_s + 3.0, bm_s + mu_s + bv_s + 1.0,
               bm_s * 2.0 + mu_s + bv_s + 1.0, bm_s * 2.0 + mu_s + 1.0, bv_s + 2.0, bm_s + bv_s, mu_s + 1.0,
               1.0 + (1.0 + bv_s) / 2.0, bm_s + bv_s + 2.0];
    let csg: Vec<f32> = cse.iter().map(|x| gamma(*x)).collect();
    let mut cge = vec![0f32; 12 * NRHG];
    for m in 0..NRHG {
        let c = &mut cge[m * 12..(m + 1) * 12];
        c[0] = bm_g + 1.0; c[1] = mu_g + 1.0; c[2] = bm_g + mu_g + 1.0; c[3] = bm_g * 2.0 + mu_g + 1.0;
        c[9] = mu_g + 2.0; c[11] = bm_g * 0.5 + mu_g + 1.0;
    }
    for m in 0..NRHG {
        let c = &mut cge[m * 12..(m + 1) * 12];
        c[4] = bm_g * 2.0 + mu_g + bv_g[m] + 1.0;
        c[5] = bm_g + mu_g + bv_g[m] + 1.0;
        c[6] = bm_g * 0.5 + mu_g + bv_g[m] + 1.0;
        c[7] = mu_g + bv_g[m] + 1.0;
        c[8] = mu_g + bv_g[m] + 3.0;
        c[10] = 0.5 * (bv_g[m] + 5.0 + 2.0 * mu_g);
    }
    let cgg: Vec<f32> = cge.iter().map(|x| gamma(*x)).collect();
    let oge1 = 1.0f32 / cge[0];
    let (ogg1, ogg2, ogg3) = (1.0f32 / cgg[0], 1.0f32 / cgg[1], 1.0f32 / cgg[2]);
    let sqrt_av_r = av_r.sqrt();
    let sqrt_av_s = av_s.sqrt();
    let t1_qr_qc = pi * 0.25 * av_r * crg[8];
    let t1_qr_qi = pi * 0.25 * av_r * crg[8];
    let t2_qr_qi = pi * 0.25 * am_r * av_r * crg[7];
    let t1_qs_qc = pi * 0.25 * av_s;
    let t1_qs_qi = pi * 0.25 * av_s;
    let t1_qr_ev = 0.78f32 * crg[9];
    let t2_qr_ev = 0.308f32 * sc3 * sqrt_av_r * crg[10];
    let t1_qs_sd = 0.86f32;
    let t2_qs_sd = 0.28f32 * sc3 * sqrt_av_s;
    let t1_qs_me = pi * 4.0 * c_sqrd * olfus * 0.86;
    let t2_qs_me = pi * 4.0 * c_sqrd * olfus * 0.28 * sc3 * sqrt_av_s;
    let t1_qg_sd = 0.86f32 * cgg[9];
    let t1_qg_me = pi * 4.0 * c_cube * olfus * 0.86 * cgg[9];

    // initialize_bins_for_tables.
    let ints = [nint_log10(r_c[0]), nint_log10(r_i[0]), nint_log10(r_r[0]), nint_log10(r_s[0]),
                nint_log10(r_g[0]), nint_log10(nt_i[0]), nint_log10(n0r_exp[0]), nint_log10(n0g_exp[0]),
                nint_log10(nt_in[0])];
    let nbins = 100usize;
    let mut dc = vec![0f64; nbins]; let mut dtc = vec![0f64; nbins];
    dc[0] = d0c as f64; dtc[0] = d0c as f64;
    for n in 1..nbins { dc[n] = dc[n - 1] + 1.0e-6; dtc[n] = dc[n] - dc[n - 1]; }
    let (di, dti) = create_bins(nbins, d0i as f64, 2.0 * d0s as f64);
    let (dr, dtr) = create_bins(nbins, d0r as f64, 0.005);
    let (ds, dts) = create_bins(nbins, d0s as f64, 0.02);
    let (dg, dtg) = create_bins(nbins, d0g as f64, 0.05);
    let (mut t_nc, _) = create_bins(nbins, 1.0, 3000.0);
    for v in t_nc.iter_mut() { *v = *v * 1.0e6; }
    let nic1 = unsafe { log(t_nc[nbins - 1] / t_nc[0]) };
    let hail_bins = if hail_size { Some(create_bins(50, 500.0e-6, 0.075)) } else { None };
    let radar = if melt { Some((create_bins(50, 100.0e-6, 0.02), create_bins(50, 100.0e-6, 0.05))) } else { None };

    // Output in the Python side's TEMPO_DERIVED order.
    let mut o = Out { bytes: Vec::new() };
    o.f32s(&av_g); o.f32s(&bv_g); o.i32(dim_nrhg);
    for v in [am_i, am_r] { o.f32(v); }
    o.f32s(&am_g);
    for v in [ar_volume, lfus, olfus, orv, rho_not, rho_not0, sc3, d0i, xm0s, xm0g, obmi, obmr, oams, obms, ocms,
              obmg] { o.f32(v); }
    o.f32s(&oamg); o.f32s(&ocmg); o.f32s(&cce); o.f32s(&ccg); o.f32s(&ocg1); o.f32s(&ocg2);
    o.f32s(&cie); o.f32s(&cig); o.f32(oig1); o.f32(oig2); o.f32s(&cre); o.f32s(&crg);
    for v in [ore1, org1, org2, org3] { o.f32(v); }
    o.f32s(&cse); o.f32s(&csg); o.f32s(&cge); o.f32s(&cgg);
    for v in [oge1, ogg1, ogg2, ogg3, t1_qr_qc, t1_qr_qi, t2_qr_qi, t1_qs_qc, t1_qs_qi, t1_qr_ev, t2_qr_ev,
              t1_qs_sd, t2_qs_sd, t1_qs_me, t2_qs_me, t1_qg_sd, t1_qg_me] { o.f32(v); }
    for v in ints { o.i32(v); }
    for v in [&dc, &dtc, &di, &dti, &dr, &dtr, &ds, &dts, &dg, &dtg, &t_nc] { o.f64s(v); }
    o.f64s(&[nic1]);
    if let Some((h, dh)) = &hail_bins { o.f64s(h); o.f64s(dh); }
    if let Some(((s, dsr), (g, dgr))) = &radar { o.f64s(s); o.f64s(dsr); o.f64s(g); o.f64s(dgr); }
    std::io::stdout().lock().write_all(&o.bytes).unwrap();
}
