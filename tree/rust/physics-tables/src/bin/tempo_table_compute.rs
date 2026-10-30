//! TEMPO v3 lookup tables tempo_init computes at run time, without a native library.
//! TEMPO 9adf9ef module_mp_tempo_utils.F90: compute_efrw (322-374), compute_efsw (376-414), qi_aut_qs (416-464),
//! compute_drop_evap (466-500), calc_gamma_p/series/cf (27-119), get_nuc (307-320). Operation order, kinds and
//! the libm calls follow the strict gfortran build (no contraction, no reassociation): binary32 work stays
//! binary32 (gamma, powf, expf, logf), binary64 work binary64 (pow, exp, log, atan), integer powers are
//! libgcc's __powidf2 square-and-multiply. Cells of the large drop-evaporation table run on every core.
//!
//! Input (stdin, little-endian, in this order): f32 scalars am_r bm_r obmr nu_c_scale rho_w pi av_s bv_s fv_s am_s
//! bm_s d0s am_i oig1 obmi bm_i mu_i d0i; f32 arrays r_c[37] cce[5x15] ccg[5x15] ocg1[15] cie[7] cig[7] nt_i[55]
//! r_i[64]; f64 arrays dc dtc t_nc dr ds di dti [100 each].
//! Output (stdout, little-endian binary64, Fortran order): tpc_wev, tnc_wev [100,37,100], t_efrw, t_efsw
//! [100,100], tps_iaus, tni_iaus, tpi_ide [64,55].
use std::io::{Read, Write};

// Gamma is the project's own correctly rounded tgammaf, never the host C library's (../own_gamma.rs says what
// broke): calc_gamma_p's series and continued fraction divide by it.
#[path = "../own_gamma.rs"]
mod own_gamma;

extern "C" {
    fn powf(x: f32, y: f32) -> f32;
    fn expf(x: f32) -> f32;
    fn logf(x: f32) -> f32;
    fn pow(x: f64, y: f64) -> f64;
    fn exp(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn atan(x: f64) -> f64;
}

const NBC: usize = 100; const NBR: usize = 100; const NBS: usize = 100; const NBI: usize = 100;
const NTB_C: usize = 37; const NTB_I: usize = 64; const NTB_I1: usize = 55;

struct Reader { bytes: Vec<u8>, at: usize }
impl Reader {
    fn f32(&mut self) -> f32 { let v = f32::from_le_bytes(self.bytes[self.at..self.at+4].try_into().unwrap()); self.at += 4; v }
    fn f64(&mut self) -> f64 { let v = f64::from_le_bytes(self.bytes[self.at..self.at+8].try_into().unwrap()); self.at += 8; v }
    fn f32s(&mut self, n: usize) -> Vec<f32> { (0..n).map(|_| self.f32()).collect() }
    fn f64s(&mut self, n: usize) -> Vec<f64> { (0..n).map(|_| self.f64()).collect() }
}

// libgcc __powidf2 (GCC's expansion of real(dp)**integer for a non-constant integer exponent).
fn powi(mut x: f64, m: i32) -> f64 {
    let mut n = m.unsigned_abs();
    let mut y = if n % 2 == 1 { x } else { 1.0 };
    loop {
        n >>= 1;
        if n == 0 { break; }
        x = x * x;
        if n % 2 == 1 { y = y * x; }
    }
    if m < 0 { 1.0 / y } else { y }
}

// Fortran NINT for binary32: round half away from zero.
fn nint(x: f32) -> i32 { x.round() as i32 }

// utils.F90:47-79 (binary32).
fn gamma_series(a: f32, x: f32) -> f32 {
    let mut ap1 = a;
    let mut term = 1.0f32 / own_gamma::tgammaf(ap1 + 1.0);
    let mut sum = term;
    for _ in 1..=100 {
        ap1 = ap1 + 1.0;
        term = term * x / ap1;
        sum = sum + term;
        if term.abs() < sum.abs() * 1.0e-7f32 { break; }
    }
    sum * unsafe { powf(x, a) } * unsafe { expf(-x) }
}

// utils.F90:81-119 (binary32).
fn gamma_cf(a: f32, x: f32) -> f32 {
    let offset = 1.0e-30f32;
    let mut b = 1.0f32 - a + x;
    let mut d = 1.0f32 / b;
    let h0 = offset;
    let mut c = b + 1.0f32 / offset;
    let mut delta = c * d;
    let mut h = h0 * delta;
    for k in 1..=100 {
        let kf = k as f32;
        let aj = kf * (a - kf);
        b = b + 2.0;
        d = b + aj * d;
        if d.abs() < offset { d = offset; }
        c = b + aj / c;
        if c.abs() < offset { c = offset; }
        d = 1.0 / d;
        delta = c * d;
        h = h * delta;
        if (delta - 1.0).abs() < 1.0e-7f32 { break; }
    }
    (unsafe { expf(-x + a * logf(x)) }) * h / own_gamma::tgammaf(a)
}

fn gamma_p(a: f32, x: f32) -> f32 {
    if x < a + 1.0 { gamma_series(a, x) } else { 1.0 - gamma_cf(a, x) }
}

fn main() {
    let mut bytes = Vec::new();
    std::io::stdin().read_to_end(&mut bytes).unwrap();
    let mut r = Reader { bytes, at: 0 };
    let [am_r, bm_r, obmr, nu_c_scale, rho_w, pi, av_s, bv_s, fv_s, am_s, bm_s, d0s, am_i, oig1, obmi, bm_i, mu_i, d0i]:
        [f32; 18] = r.f32s(18).try_into().unwrap();
    let r_c = r.f32s(NTB_C);
    let cce = r.f32s(5 * 15); let ccg = r.f32s(5 * 15); let ocg1 = r.f32s(15);
    let cie = r.f32s(7); let cig = r.f32s(7);
    let nt_i = r.f32s(NTB_I1); let r_i = r.f32s(NTB_I);
    let dc = r.f64s(NBC); let dtc = r.f64s(NBC); let t_nc = r.f64s(NBC);
    let dr = r.f64s(NBR); let ds = r.f64s(NBS); let di = r.f64s(NBI); let dti = r.f64s(NBI);
    assert_eq!(r.at, r.bytes.len(), "input words differ from the declared tables");
    let (rho_w, pi64) = (rho_w as f64, pi as f64);
    let two_over_pi = 2.0f64 / pi64;
    let ccg_ = |m: usize, n: usize| ccg[(n - 1) * 5 + (m - 1)];
    let cce_ = |m: usize, n: usize| cce[(n - 1) * 5 + (m - 1)];

    // compute_drop_evap: tpc_wev(i,j,k), tnc_wev(i,j,k); every k on its own thread chunk.
    let massc: Vec<f64> = (0..NBC).map(|n| am_r as f64 * unsafe { pow(dc[n], bm_r as f64) }).collect();
    let plane = NBC * NTB_C;
    let mut tpc = vec![0f64; NBC * NTB_C * NBC];
    let mut tnc = vec![0f64; NBC * NTB_C * NBC];
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(NBC);
    let per = (NBC + workers - 1) / workers;
    std::thread::scope(|scope| {
        for (chunk, (pc, nc)) in tpc.chunks_mut(plane * per).zip(tnc.chunks_mut(plane * per)).enumerate() {
            let (massc, dc, dtc, t_nc, r_c, ocg1) = (&massc, &dc, &dtc, &t_nc, &r_c, &ocg1);
            let ccg_ = &ccg_; let cce_ = &cce_;
            scope.spawn(move || {
                for (local, (pk, nk)) in pc.chunks_mut(plane).zip(nc.chunks_mut(plane)).enumerate() {
                    let k = chunk * per + local;
                    let ncf = t_nc[k] as f32;
                    let nu_c = if nu_c_scale / ncf >= 12.5f32 { 15 } else { 15.min(nint(nu_c_scale / ncf) + 2) } as usize;
                    for j in 0..NTB_C {
                        let lamc = unsafe { pow(t_nc[k] * am_r as f64 * ccg_(2, nu_c) as f64 * ocg1[nu_c - 1] as f64
                                                / r_c[j] as f64, obmr as f64) };
                        let n0_c = t_nc[k] * ocg1[nu_c - 1] as f64 * unsafe { pow(lamc, cce_(1, nu_c) as f64) };
                        let (mut summ, mut summ2) = (0f64, 0f64);
                        for i in 0..NBC {
                            let n_c = n0_c * powi(dc[i], nu_c as i32) * unsafe { exp(-(lamc * dc[i])) } * dtc[i];
                            summ = summ + massc[i] * n_c;
                            summ2 = summ2 + n_c;
                            pk[j * NBC + i] = summ;
                            nk[j * NBC + i] = summ2;
                        }
                    }
                }
            });
        }
    });

    // compute_efrw: t_efrw(i,j), i over rain bins, j over cloud bins.
    let mut efrw = vec![0f64; NBR * NBC];
    for j in 0..NBC {
        for i in 0..NBR {
            let ef_rw;
            let p = dc[j] / dr[i];
            if dr[i] < 50.0e-6 || dc[j] < 3.0e-6 {
                efrw[j * NBR + i] = 0.0;
                continue;
            } else if p > 0.25 {
                let x = dc[j] * 1.0e6;
                ef_rw = if dr[i] < 75.0e-6 {
                    0.026794 * x - 0.20604
                } else if dr[i] < 125.0e-6 {
                    -0.00066842 * x * x + 0.061542 * x - 0.37089
                } else if dr[i] < 175.0e-6 {
                    4.091e-06 * x * x * x * x - 0.00030908 * x * x * x + 0.0066237 * x * x - 0.0013687 * x - 0.073022
                } else if dr[i] < 250.0e-6 {
                    9.6719e-5 * x * x * x - 0.0068901 * x * x + 0.17305 * x - 0.65988
                } else if dr[i] < 350.0e-6 {
                    9.0488e-5 * x * x * x - 0.006585 * x * x + 0.16606 * x - 0.56125
                } else {
                    0.00010721 * x * x * x - 0.0072962 * x * x + 0.1704 * x - 0.46929
                };
            } else {
                let d = dr[i];
                let vtr = -0.1021 + 4.932e3 * d - 0.9551e6 * d * d + 0.07934e9 * d * d * d - 0.002362e12 * d * d * d * d;
                let stokes = dc[j] * dc[j] * vtr * rho_w / (9.0 * 1.718e-5 * d);
                let reynolds = 9.0 * stokes / (p * p * rho_w);
                let f = unsafe { log(reynolds) };
                let g = -0.1007 - 0.358 * f + 0.0261 * f * f;
                let k0 = unsafe { exp(g) };
                let z = unsafe { log(stokes / (k0 + 1.0e-15)) };
                let h = 0.1465 + 1.302 * z - 0.607 * z * z + 0.293 * z * z * z;
                let yc0 = two_over_pi * unsafe { atan(h) };
                ef_rw = (yc0 + p) * (yc0 + p) / ((1.0 + p) * (1.0 + p));
            }
            efrw[j * NBR + i] = 0f64.max(ef_rw.min(0.95));
        }
    }

    // compute_efsw: t_efsw(i,j), i over snow bins, j over cloud bins.
    let mut efsw = vec![0f64; NBS * NBC];
    for j in 0..NBC {
        let vtc = 1.19e4 * (1.0e4 * dc[j] * dc[j] * 0.25);
        for i in 0..NBS {
            let vts = av_s as f64 * unsafe { pow(ds[i], bv_s as f64) } * unsafe { exp(-(fv_s as f64 * ds[i])) } - vtc;
            let ds_m = unsafe { pow(am_s as f64 * pow(ds[i], bm_s as f64) / am_r as f64, obmr as f64) };
            let p = dc[j] / ds_m;
            if p > 0.25 || ds[i] < d0s as f64 || dc[j] < 6.0e-6 || vts < 1.0e-3 {
                efsw[j * NBS + i] = 0.0;
            } else {
                let stokes = dc[j] * dc[j] * vts * rho_w / (9.0 * 1.718e-5 * ds_m);
                let reynolds = 9.0 * stokes / (p * p * rho_w);
                let f = unsafe { log(reynolds) };
                let g = -0.1007 - 0.358 * f + 0.0261 * f * f;
                let k0 = unsafe { exp(g) };
                let z = unsafe { log(stokes / (k0 + 1.0e-15)) };
                let h = 0.1465 + 1.302 * z - 0.607 * z * z + 0.293 * z * z * z;
                let yc0 = two_over_pi * unsafe { atan(h) };
                let ef_sw = (yc0 + p) * (yc0 + p) / ((1.0 + p) * (1.0 + p));
                efsw[j * NBS + i] = 0f64.max(ef_sw.min(0.95));
            }
        }
    }

    // qi_aut_qs: tps_iaus, tni_iaus, tpi_ide (i over ice mass, j over ice number).
    let (mut tps, mut tni, mut tpi) = (vec![0f64; NTB_I * NTB_I1], vec![0f64; NTB_I * NTB_I1], vec![0f64; NTB_I * NTB_I1]);
    for j in 0..NTB_I1 {
        for i in 0..NTB_I {
            let at = j * NTB_I + i;
            let lami = unsafe { powf(am_i * cig[1] * oig1 * nt_i[j] / r_i[i], obmi) } as f64;
            let di_mean = (bm_i + mu_i + 1.0f32) as f64 / lami;
            let n0_i = (nt_i[j] * oig1) as f64 * unsafe { pow(lami, cie[0] as f64) };
            let (mut t1, mut t2) = (0f64, 0f64);
            if (di_mean as f32) > 5.0f32 * d0s {
                t1 = r_i[i] as f64;
                t2 = nt_i[j] as f64;
                tpi[at] = 0.0;
            } else if (di_mean as f32) < d0i {
                tpi[at] = 1.0;
            } else {
                let xlimit = (lami * d0s as f64) as f32;
                tpi[at] = gamma_p(mu_i + 2.0f32, xlimit) as f64;
                for n2 in 0..NBI {
                    let n_i = n0_i * unsafe { pow(di[n2], mu_i as f64) } * unsafe { exp(-(lami * di[n2])) } * dti[n2];
                    if di[n2] >= d0s as f64 {
                        t1 = t1 + n_i * am_i as f64 * unsafe { pow(di[n2], bm_i as f64) };
                        t2 = t2 + n_i;
                    }
                }
            }
            tps[at] = t1;
            tni[at] = t2;
        }
    }

    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for table in [&tpc, &tnc, &efrw, &efsw, &tps, &tni, &tpi] {
        for v in table.iter() { out.write_all(&v.to_le_bytes()).unwrap(); }
    }
}
