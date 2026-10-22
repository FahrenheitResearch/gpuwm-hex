// GOCART source formulas seeded from WRF v4.7.1, public domain.
// Reference: gpuwm/core/kernels/chem_dust.cu and chem_seasalt.cu.
pub fn dust(
    density: f64,
    radius: f64,
    fraction: f64,
    ch: f64,
    erod: f64,
    soil: usize,
    moisture: f64,
    rho: f64,
    wind: f64,
) -> Result<f64, String> {
    if !(density > 0.
        && radius > 0.
        && fraction >= 0.
        && ch >= 0.
        && erod >= 0.
        && erod <= 1.
        && moisture >= 0.
        && rho > 0.
        && wind >= 0.)
    {
        return Err("invalid dust density/radius/fraction/coefficient/erodibility/moisture/air density/wind".into());
    }
    const POROSITY: [f64; 19] = [
        0.339, 0.421, 0.434, 0.476, 0.476, 0.439, 0.404, 0.464, 0.465, 0.406, 0.468, 0.468, 0.439,
        1., 0.2, 0.421, 0.468, 0.2, 0.339,
    ];
    if !(1..=19).contains(&soil) {
        return Err("soil category outside GOCART porosity table 1..19".into());
    }
    let den = density * 1e-3;
    let diameter = 2. * radius * 1e2;
    let gravity = 981.;
    let airden = rho * 1e-3;
    let threshold = 0.13e-2
        * (den * gravity * diameter / airden).sqrt()
        * (1. + 0.006 / den / gravity / diameter.powf(2.5)).sqrt()
        / (1.928 * (1331. * diameter.powf(1.56) + 0.38).powf(0.092) - 1.).sqrt();
    let wet = moisture / POROSITY[soil - 1];
    let threshold = if wet < 0.5 {
        (threshold * (1.2 + 0.2 * wet.max(1e-3).log10())).max(0.)
    } else {
        100.
    };
    Ok((ch * fraction * erod * wind * wind * (wind - threshold)).max(0.))
}
pub fn salt(lo: f64, hi: f64, density: f64, wind: f64) -> Result<f64, String> {
    if !(lo > 0. && hi > lo && density > 0. && wind >= 0.) {
        return Err("invalid sea salt bin/density/wind".into());
    }
    let count = ((hi - lo) * 2. / 0.05 + 0.001) as usize;
    if count > 100_000 {
        return Err("sea salt bin needs over 100000 quadrature points".into());
    }
    let mut flux = 0.;
    for ir in 0..count {
        let rw = 2. * lo + (ir as f64 + 0.5) * 0.05;
        let b = (0.38 - rw.log10()) / 0.65;
        let dfn =
            1.373 / rw.powi(3) * (1. + 0.057 * rw.powf(1.05)) * 10f64.powf(1.19 * (-b * b).exp());
        let dry = rw / 2. * 1e-6;
        flux += (4. / 3.) * std::f64::consts::PI * dry.powi(3) * density * 2. * dfn * 0.05;
    }
    Ok(flux * wind.powf(3.2))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wind_and_moisture() {
        assert_eq!(
            dust(2500., 1e-6, 0.1, 1e-9, 0.4, 1, 0.4, 1.2, 15.).unwrap(),
            0.
        );
        assert!(dust(2500., 1e-6, 0.1, 1e-9, 0.4, 1, 0., 1.2, 15.).unwrap() > 0.);
    }
    #[test]
    fn salt_wind_law() {
        let a = salt(0.1, 0.5, 2200., 10.).unwrap();
        let b = salt(0.1, 0.5, 2200., 20.).unwrap();
        assert!((b / a - 2f64.powf(3.2)).abs() < 1e-12);
        assert_eq!(salt(0.1, 0.5, 2200., 0.).unwrap(), 0.);
    }
}
