//! The project's own correctly rounded binary32 gamma, for the TEMPO table programs.
//!
//! THE BREAKAGE THIS PREVENTS: tempo_table_init and tempo_table_compute called the host C library's tgammaf, so
//! the TEMPO tables followed whatever libm the box had. glibc 2.43 (the native reference host) returns the
//! correctly rounded gamma; glibc 2.39 (Ubuntu 24.04, e.g. the V5090 box) does not (tgammaf(4) = 6.0000005), so
//! 268,926 of 772,327 table words (ccg, cig, crg, csg, cgg and everything computed from them) changed with the
//! host and a strict hrrrv5 run differed from native in 1,957,713 of 7,901,197 words after 1 h on a case that is
//! bitwise on the reference host.
//!
//! This is gfk_tgamma (src/hexcore/data/mpasphys-maths/glibc_flt32.cuh), ArWen's own Apache-2.0 gamma that the
//! CUDA ports already evaluate, in Rust: Gamma on [1,2) as a degree-9 polynomial on 16 equal segments, the
//! recurrence Gamma(z+1) = z Gamma(z) elsewhere, reflection for x < 0, all in binary64 rounded once to binary32.
//! Correctly rounded on every binary32 argument of [0.25, 36] (glibc_flt32.cuh records the 113-bit measurement).
//! MEASURED for this Rust code over all 59,768,833 binary32 arguments of [0.25, 36]: 0 different from the
//! reference host's glibc 2.43 tgammaf, 23,575,230 different from glibc 2.39's (the count the CUDA code
//! records). Rust never fuses a multiply-add on its own, so every binary64 operation below rounds exactly as
//! the CUDA code's pinned ones.

const TG_C: [[u64; 10]; 16] = [
    [0x3fef73ed01940522, 0xbfe092fd20dd784c, 0x3fed1a2ea66d2aef, 0xbfe966a1d7f2b9d9, 0x3feaf057a1fa6b33, 0xbfea114e80ce35b1, 0x3fe99b28f9ddcee8, 0xbfe8e73dca5f8820, 0x3fe843285f62ee7d, 0xbfe78e83c533d8b1],
    [0x3fee865a5b755fb9, 0xbfda6b50f60b5c6e, 0x3fe8e9a9675e8147, 0xbfe39251063e4e66, 0x3fe41a834d107f5e, 0xbfe237aaf45da8d8, 0x3fe0f0e4f41772ea, 0xbfdf11209307f1c0, 0x3fdc8cbfdb620ef9, 0xbfda24012d73ac3c],
    [0x3fedcac35f2a7419, 0xbfd49cf184c91f8e, 0x3fe5ac61acea578e, 0xbfde5d43093961e1, 0x3fde91903e412f2a, 0xbfd9eedaece7d029, 0x3fd6f07e745bddde, 0xbfd3e2e7165ba686, 0x3fd14c7245554af4, 0xbfcdf8a55df66ee0],
    [0x3fed3aa3cecb6cd0, 0xbfcf0b8c2384c507, 0x3fe327dd5130ef72, 0xbfd7a2070b357f21, 0x3fd7a7cfacdf879d, 0xbfd2c07bb9dae965, 0x3fcfb753b1f31c65, 0xbfca0d97e4269587, 0x3fc58589453b97bb, 0xbfc1b0d3d42d932b],
    [0x3fecd0ebb0c4e488, 0xbfc5fa4609a59d2c, 0x3fe13236e09cf181, 0xbfd2616a66cb748a, 0x3fd29fb9224d7007, 0xbfcb7c019bda7132, 0x3fc658f76dab2a86, 0xbfc16bef3649d39b, 0x3fbb6aee3baf8284, 0xbfb570897c5886eb],
    [0x3fec89aaab6c10fd, 0xbfbb8d4972a0d9c0, 0x3fdf59ee44fdc796, 0xbfcc6c163713c6cb, 0x3fcdd53c7e7fbca8, 0xbfc45d61e648b785, 0x3fc00706690f6116, 0xbfb7bbc7df8eac4c, 0x3fb1d809aa6ec9f0, 0xbfaa9b3bfeb56d76],
    [0x3fec61d286fe74ed, 0xbfa8f960eacc3e79, 0x3fdd035f977bf7ff, 0xbfc5afe653a01c68, 0x3fc850c11888a092, 0xbfbe6f81933b909d, 0x3fb7607930824288, 0xbfb06f53c3fa90a8, 0x3fa7afbb95743636, 0xbfa0de3bdb94b909],
    [0x3fec5709f063f61e, 0x3f78e787a2ac1f9e, 0x3fdb3f656a414190, 0xbfc026537ebbdf28, 0x3fc42e2dcb386c76, 0xbfb6dedd3608424a, 0x3fb1533b414d61e0, 0xbfa718dbd05eec3d, 0x3fa001e8923ca168, 0xbf95cf9f00355d61],
    [0x3fec678adaa16db2, 0x3fadade0522ce28d, 0x3fd9f502b04aa917, 0xbfb7061698453ed0, 0x3fc111adaec147d2, 0xbfb138a886268f36, 0x3faa17ae1c392bd6, 0xbfa0707922c8c43e, 0x3f960077de104a49, 0xbf8cb5c7571211dd],
    [0x3fec920953aa5d66, 0x3fbb9496a6874861, 0x3fd9116c2e5de8d1, 0xbfae346a34c78d6e, 0x3fbd74c383999e27, 0xbfa9de756ede0cd1, 0x3fa3f8131690fd1a, 0xbf97a888b14ec653, 0x3f8eb8b5d6cd0026, 0xbf8334a276679165],
    [0x3fecd5a098928442, 0x3fc3fb8f1d0abf2c, 0x3fd8867f0f1d270a, 0xbfa0654798b0415f, 0x3fb9f3d067f9d585, 0xbfa3412ca3fa8323, 0x3f9f164f1fbed563, 0xbf912caa9ef9524e, 0x3f85c618b73bc8a7, 0xbf7a14084fcb5d75],
    [0x3fed31c4db6ff586, 0x3fca140aba605e6a, 0x3fd849a7d111fe3e, 0xbf80665ee5c25bec, 0x3fb75d82f748db66, 0xbf9c1f8dbdb02d95, 0x3f98a422b1844434, 0xbf8917ecef6748cd, 0x3f7f5201988eff97, 0xbf71f187b1036fac],
    [0x3feda6389f09f623, 0x3fd0131e5b57cf1a, 0x3fd853176b1f3b73, 0x3f8c5c9000d6e1ce, 0x3fb581177702069b, 0xbf93d027a323721f, 0x3f93ec6f9466489f, 0xbf8263da5167e690, 0x3f76dbe71bbe090c, 0xbf68fd143df66799],
    [0x3fee3304db941633, 0x3fd3217c4a579d7c, 0x3fd89d2fbae9b259, 0x3fa1807b533d9092, 0x3fb43ae1d6087604, 0xbf8a0c1119bc32a0, 0x3f9077745c6a4217, 0xbf7aecbc52480399, 0x3f70f00dd6253d47, 0xbf6194b848e9be9e],
    [0x3feed87357995087, 0x3fd63cf26c2a3f66, 0x3fd924179f06fc8c, 0x3fab66aa0070a6b4, 0x3fb370c0b9271813, 0xbf7d632bc804cea1, 0x3f8be2e3854ef50b, 0xbf7391015cd2774d, 0x3f69823259db4371, 0xbf58f35002269e0f],
    [0x3fef970ac84d0a49, 0x3fd96cf0f1b4e4d0, 0x3fd9e56bf5311f1a, 0x3fb2815a841a3351, 0x3fb30fa25285e4fe, 0xbf63d45167c1dcba, 0x3f883e8c0f79bf95, 0xbf6bf53a20fad880, 0x3f638d7f76fe48b2, 0xbf51d0e83e2ca312],
];
const TG_SP: [u64; 13] = [0x400921fb54442d18, 0xc014abbce625be53, 0x400466bc6775aae2, 0xbfe32d2cce62bd86, 0x3fb50783487ee782, 0xbf7e3074fde8871f, 0x3f3e8f434d018d63, 0xbef6fadb9f155744, 0x3eaaaec32af93359, 0xbe58a404211f9547, 0x3e02877020d52cf0, 0xbda7215f879e1ac9, 0x3d4859c594ba4573];
const TG_PI: u64 = 0x400921fb54442d18;

#[inline]
fn coefficient(j: usize, k: usize) -> f64 { f64::from_bits(TG_C[j][k]) }

// Gamma(r) for r in [1,2). The two subtractions are exact (Sterbenz), so j and u are exact.
fn poly(r: f64) -> f64 {
    let t = (r - 1.0) * 16.0;
    let j = (t as i32).min(15) as usize;
    let u = r - (33 + 2 * j) as f64 * 0.03125;
    let mut p = coefficient(j, 9);
    for k in (0..9).rev() {
        p = p * u + coefficient(j, k);
    }
    p
}

// Gamma(v) in binary64 for 0 < v < 43.
fn positive(v: f64) -> f64 {
    if v >= 1.0 {
        let n = v as i32;
        let mut p = poly(v - (n - 1) as f64);
        for i in 1..n {
            p = p * (v - i as f64);
        }
        return p;
    }
    poly(v + 1.0) / v
}

// sin(pi x) for |x| < 2^23; k and the subtraction are exact.
fn sinpi(x: f64) -> f64 {
    let k = (if x < 0.0 { x - 0.5 } else { x + 0.5 }) as i64;
    let r = x - k as f64;
    let r2 = r * r;
    let mut s = f64::from_bits(TG_SP[12]);
    for i in (0..12).rev() {
        s = s * r2 + f64::from_bits(TG_SP[i]);
    }
    s = s * r;
    if k & 1 != 0 { -s } else { s }
}

/// The correctly rounded binary32 gamma (gfk_tgamma).
pub fn tgammaf(x: f32) -> f32 {
    let quiet_nan = f32::from_bits(0x7fc0_0000);
    let ix = x.to_bits();
    let ax = ix & 0x7fff_ffff;
    if ax >= 0x7f80_0000 {
        if ix == 0xff80_0000 { return quiet_nan; }      // -inf
        return x + x;                                   // +inf, NaN quieted
    }
    if ix < 0x8000_0000 {
        if ix == 0 { return 1.0f32 / x; }               // +0 -> +inf
        if x >= 36.0f32 {
            let big = f32::from_bits(0x7f00_0000);      // 2^127
            return x * big * big;                       // +inf
        }
        return positive(x as f64) as f32;
    }
    if ix == 0x8000_0000 { return 1.0f32 / x; }         // -0 -> -inf
    if ax >= 0x4b00_0000 { return quiet_nan; }          // every |x| >= 2^23 is an integer: a pole
    let s = sinpi(x as f64);
    if s == 0.0 { return quiet_nan; }                   // negative integer: pole
    if x <= -42.0f32 { return if s < 0.0 { -0.0 } else { 0.0 }; }
    let g1 = positive(1.0 - x as f64);
    (f64::from_bits(TG_PI) / (s * g1)) as f32
}
