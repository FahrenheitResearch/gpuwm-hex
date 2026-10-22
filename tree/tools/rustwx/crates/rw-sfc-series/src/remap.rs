//! Seeded from the case-8 port of MPAS-A v8.4.2 (BSD-3-Clause).
//! The interpolation arithmetic is unchanged; independent cell loops use rayon.
use crate::surface_met::{Map, SurfaceSlab, read_surface_file};
use rayon::prelude::*;
use rw_mpas::init::hinterp::{InterpContext, Method, Slab, Underflow, interp_sequence};
use rw_mpas::{MpasError, MpasResult};
use std::path::Path;
const DEG_PER_RAD: f32 = 180.0_f32 / 3.14159265358979323846_f32;
const MISSING: f32 = -1.0e30;
/// `mpas_init_atm_surface.F:108-170`: read only the three mesh arrays case 8
/// needs; the native max(landmask)==0 refusal distinguishes grid from static.
pub(crate) fn read_mesh(path: &Path) -> MpasResult<(Vec<f32>, Vec<f32>, Vec<i32>)> {
    let file = netcrust::File::open(path)?;
    let n = file
        .dimension("nCells")
        .ok_or_else(|| MpasError::Refusal("surface input has no nCells dimension".into()))?
        .len();
    let read = |name: &str| -> MpasResult<Vec<f64>> {
        let values = file.read_array_f64_first_record_or_all(name)?.into_values();
        if values.len() != n || n == 0 {
            return Err(MpasError::Refusal(format!(
                "surface {name} has {} values for {n} cells; a different shape samples the wrong mesh",
                values.len()
            )));
        }
        Ok(values)
    };
    let lat = read("latCell")?.into_iter().map(|v| v as f32).collect();
    let lon = read("lonCell")?.into_iter().map(|v| v as f32).collect();
    let landmask: Vec<i32> = read("landmask")?.into_iter().map(|v| v as i32).collect();
    if landmask.iter().copied().max() == Some(0) {
        return Err(MpasError::Refusal("the global maximum of landmask is zero; case 8 needs a static or initial conditions file to mask surface interpolation (mpas_init_atm_surface.F:156-169)".into()));
    }
    Ok((lat, lon, landmask))
}

/// `mpas_init_atm_surface.F:289-394`. The native surface routine clamps y
/// BEFORE wrapping x. Reprojection overwrites y, even at a pole; case 7's
/// locate helper clamps AFTER wrapping and cannot be substituted here.
fn interp_to_mpas(
    source: &SurfaceSlab,
    lat: &[f32],
    lon: &[f32],
    landmask: &[i32],
    mask: Option<&Slab>,
    ice: bool,
    dest: &mut [f32],
) -> MpasResult<()> {
    let field = &source.met;
    if field.nx < 3 || field.ny == 0 || field.values.len() != field.nx * field.ny {
        return Err(MpasError::Refusal(
            "surface source needs at least three columns for the native periodic halo".into(),
        ));
    }
    if let Some(m) = mask {
        if m.end_x != field.nx as i32 + 3 || m.end_y != field.ny as i32 {
            return Err(MpasError::Refusal("LANDSEA and surface slabs differ in shape; the native shared mask would index the wrong source points".into()));
        }
    }
    let proj = Map::new(source)?;
    let array = Slab::from_met(&field.values, field.nx, field.ny);
    let ctx = InterpContext {
        array: &array,
        msgval: MISSING,
        maskval: Some(if ice { 1.0 } else { -1.0 }),
        mask,
        underflow: Underflow::Preserve,
        underflow_hits: None,
    };
    let methods: &[Method] = if ice {
        &[Method::FourPoint, Method::WtAverage4, Method::Search]
    } else {
        &[Method::FourPoint, Method::Search]
    };
    dest.par_iter_mut().enumerate().for_each(|(i, output)| {
        if landmask[i] == if ice { 1 } else { -1 } {
            *output = 0.0;
            return;
        }
        let lat = lat[i] * DEG_PER_RAD;
        let lon = lon[i] * DEG_PER_RAD;
        let (mut x, mut y) = proj.ij(lat, lon);
        if y <= 0.5 {
            y = 1.0;
        } else if y >= field.ny as f32 + 0.5 {
            y = field.ny as f32;
        }
        if x < 0.5 {
            (x, y) = proj.ij(lat, lon + 360.0);
        } else if x >= field.nx as f32 + 0.5 {
            (x, y) = proj.ij(lat, lon - 360.0);
        }
        *output = interp_sequence(x, y, &ctx, methods, 0);
    });
    Ok(())
}

/// `mpas_init_atm_surface.F:104-286`. Scan the entire file for LANDSEA
/// first, then process fields in their original order. Last LANDSEA and last
/// SST/SKINTEMP win. A field absent at this time retains the prior time's data.
pub(crate) fn interp_sfc(
    slabs: &[SurfaceSlab],
    lat: &[f32],
    lon: &[f32],
    landmask: &[i32],
    sst: &mut [f32],
    xice: &mut [f32],
) -> MpasResult<(Vec<String>, bool)> {
    let landsea = slabs
        .iter()
        .rev()
        .map(|s| &s.met)
        .find(|s| s.field == "LANDSEA");
    let mask = match landsea {
        Some(s) if s.nx >= 3 => Some(Slab::from_met(&s.values, s.nx, s.ny)),
        Some(_) => {
            return Err(MpasError::Refusal(
                "LANDSEA needs three columns for the native periodic halo".into(),
            ));
        }
        None => None,
    };
    let mut used = Vec::new();
    for s in slabs {
        match s.met.field.as_str() {
            "SST" | "SKINTEMP" => {
                sst.fill(0.0);
                interp_to_mpas(s, lat, lon, landmask, mask.as_ref(), false, sst)?;
            }
            "SEAICE" => {
                xice.fill(0.0);
                interp_to_mpas(s, lat, lon, landmask, mask.as_ref(), true, xice)?;
            }
            _ => continue,
        }
        used.push(s.met.field.clone());
    }
    xice.par_iter_mut().enumerate().for_each(|(i, value)| {
        if sst[i] < 271.0 && landmask[i] == 0 {
            *value = 1.0;
        }
        // Two comparisons, preserving native NaN behaviour.
        if *value < 0.0 {
            *value = 0.0;
        }
        if *value > 1.0 {
            *value = 1.0;
        }
    });
    Ok((used, mask.is_some()))
}

pub(crate) fn remap(
    path: &Path,
    lat: &[f32],
    lon: &[f32],
    land: &[i32],
) -> MpasResult<(Vec<f32>, Vec<f32>, Option<Vec<f32>>)> {
    remap_slabs(&read_surface_file(path)?, lat, lon, land)
}

pub(crate) fn remap_slabs(
    slabs: &[SurfaceSlab],
    lat: &[f32],
    lon: &[f32],
    land: &[i32],
) -> MpasResult<(Vec<f32>, Vec<f32>, Option<Vec<f32>>)> {
    let present = |field: &str| slabs.iter().any(|s| s.met.field == field);
    if !(present("SST") || present("SKINTEMP")) || !present("SEAICE") || !present("LANDSEA") {
        return Err(MpasError::Refusal("each surface record needs SST/SKINTEMP, SEAICE and LANDSEA; missing records cannot silently retain stale ocean forcing".into()));
    }
    let mut sst = vec![0.; land.len()];
    let mut ice = sst.clone();
    interp_sfc(&slabs, lat, lon, land, &mut sst, &mut ice)?;
    let snow = slabs
        .iter()
        .rev()
        .find(|s| s.met.field == "SNOWH")
        .map(|s| {
            let mut output = vec![0.; land.len()];
            let array = Slab::from_met(&s.met.values, s.met.nx, s.met.ny);
            let context = InterpContext {
                array: &array,
                msgval: MISSING,
                maskval: None,
                mask: None,
                underflow: Underflow::Preserve,
                underflow_hits: None,
            };
            let projection = Map::new(s)?;
            output.par_iter_mut().enumerate().for_each(|(i, value)| {
                if land[i] == 1 {
                    let (x, y) = projection.ij(lat[i] * DEG_PER_RAD, lon[i] * DEG_PER_RAD);
                    *value =
                        interp_sequence(x, y, &context, &[Method::FourPoint, Method::Search], 0)
                            .max(0.);
                }
            });
            Ok::<_, MpasError>(output)
        })
        .transpose()?;
    if sst.par_iter().zip(&ice).zip(land).any(|((t, i), l)| {
        !t.is_finite() || (*l == 0 && *t <= 0.) || !i.is_finite() || *i < 0. || *i > 1.
    }) {
        return Err(MpasError::Refusal(
            "surface remap produced invalid ocean forcing".into(),
        ));
    }
    Ok((sst, ice, snow))
}
