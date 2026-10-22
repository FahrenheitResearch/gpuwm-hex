//! RMM/BSISO phase and Hovmoller charts in the production rw_wrfbatch renderer.
//! No latitude/longitude map is fabricated from a time axis.
use image::{Rgba, RgbaImage};
use rayon::prelude::*;
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

pub const ABI: &str = "rw-wrfbatch-s2s-diagnostics/v1 phase longitude-time latitude-time fixed-eof";
#[derive(Deserialize)]
struct Receipt {
    schema: String,
    status: String,
    reference_source: String,
    qualification: String,
    series: Vec<Row>,
}
#[derive(Deserialize)]
struct Row {
    member: String,
    diagnostic: String,
    date: String,
    pc: [f64; 2],
    phase_order: Vec<usize>,
    fields: Vec<String>,
    domain: [f64; 4],
    longitude_bins: usize,
    latitude_bins: usize,
    hovmoller: Vec<Vec<f64>>,
    latitude_hovmoller: Vec<Vec<f64>>,
}
fn label(img: &mut RgbaImage, text: &str, x: i32, y: i32) {
    rustwx_render::draw_text(img, text, x, y, rustwx_render::Rgba::BLACK, 1);
}
fn line(img: &mut RgbaImage, a: (i32, i32), b: (i32, i32), color: Rgba<u8>) {
    let steps = (b.0 - a.0).abs().max((b.1 - a.1).abs()).max(1);
    for i in 0..=steps {
        let x = a.0 + (b.0 - a.0) * i / steps;
        let y = a.1 + (b.1 - a.1) * i / steps;
        if x >= 0 && y >= 0 && x < img.width() as i32 && y < img.height() as i32 {
            img.put_pixel(x as u32, y as u32, color);
        }
    }
}
fn color(v: f64, range: f64) -> Rgba<u8> {
    let t = (v / range).clamp(-1., 1.);
    let s = (255. * (1. - t.abs())) as u8;
    if t < 0. {
        Rgba([s, s, 255, 255])
    } else {
        Rgba([255, s, s, 255])
    }
}
fn slug(s: &str) -> Result<&str, String> {
    if !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        Ok(s)
    } else {
        Err("unsafe S2S figure component".into())
    }
}
fn write(img: &RgbaImage, path: &Path) -> Result<(), String> {
    let part = path.with_extension("png.part");
    img.save_with_format(&part, image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    std::fs::rename(part, path).map_err(|e| e.to_string())?;
    println!("RENDERED\t{}", path.display());
    Ok(())
}
fn hov(
    rows: &[&Row],
    field: usize,
    latitude: bool,
    out: &Path,
    source: &str,
    qualification: &str,
) -> Result<(), String> {
    let first = rows[0];
    let bins = if latitude {
        first.latitude_bins
    } else {
        first.longitude_bins
    };
    let (w, h) = (1100u32, 800u32);
    let mut img = RgbaImage::from_pixel(w, h, Rgba([255, 255, 255, 255]));
    label(
        &mut img,
        &format!(
            "{} {} {} daily anomaly",
            first.diagnostic, first.member, first.fields[field]
        ),
        90,
        20,
    );
    label(
        &mut img,
        &source.chars().take(120).collect::<String>(),
        90,
        45,
    );
    label(
        &mut img,
        "Fixed reference background; preceding 120 days removed; daily endpoint trapezoids",
        90,
        65,
    );
    let (left, top, width, height) = (140usize, 100usize, 850usize, 570usize);
    let range = if first.fields[field] == "olr" {
        40.
    } else {
        10.
    };
    for (t, row) in rows.iter().enumerate() {
        let values = if latitude {
            &row.latitude_hovmoller[field]
        } else {
            &row.hovmoller[field]
        };
        for (x, &value) in values.iter().enumerate() {
            for iy in top + height * t / rows.len()..top + height * (t + 1) / rows.len() {
                for ix in left + width * x / bins..left + width * (x + 1) / bins {
                    img.put_pixel(ix as u32, iy as u32, color(value, range));
                }
            }
        }
        if t % ((rows.len() / 10).max(1)) == 0 || t + 1 == rows.len() {
            label(
                &mut img,
                &row.date,
                10,
                (top + height * t / rows.len()) as i32,
            );
        }
    }
    let low = if latitude {
        first.domain[0]
    } else {
        first.domain[2]
    };
    let high = if latitude {
        first.domain[1]
    } else {
        first.domain[3]
    };
    for k in 0..=6 {
        label(
            &mut img,
            &format!("{:.0}", low + (high - low) * k as f64 / 6.),
            (left + width * k / 6 - 8) as i32,
            680,
        );
    }
    label(
        &mut img,
        if latitude {
            "Latitude (degrees north); time increases downward"
        } else {
            "Longitude (degrees east); time increases downward"
        },
        240,
        710,
    );
    let units = if first.fields[field] == "olr" {
        "W m-2"
    } else {
        "m s-1"
    };
    for x in 0..450 {
        let value = (x as f64 / 449. * 2. - 1.) * range;
        for y in 742..760 {
            img.put_pixel(300 + x, y, color(value, range));
        }
    }
    label(&mut img, &format!("-{range}"), 295, 770);
    label(&mut img, "0", 522, 770);
    label(&mut img, &format!("+{range}"), 740, 770);
    label(&mut img, units, 790, 770);
    label(&mut img, &qualification.chars().take(160).collect::<String>(), 30, 790);
    let axis = if latitude { "latitude" } else { "longitude" };
    write(
        &img,
        &out.join(format!(
            "{}-{}-{}-{axis}-time.png",
            slug(&first.diagnostic)?,
            slug(&first.member)?,
            slug(&first.fields[field])?
        )),
    )
}
fn phase(rows: &[&Row], out: &Path, source: &str, qualification: &str) -> Result<(), String> {
    let first = rows[0];
    let mut img = RgbaImage::from_pixel(900, 900, Rgba([255, 255, 255, 255]));
    label(
        &mut img,
        &format!(
            "{} {} fixed EOF phase trajectory",
            first.diagnostic, first.member
        ),
        90,
        20,
    );
    label(
        &mut img,
        &source.chars().take(110).collect::<String>(),
        90,
        45,
    );
    label(
        &mut img,
        &format!(
            "{} to {}; unit circle marks amplitude 1",
            rows[0].date,
            rows.last().unwrap().date
        ),
        90,
        70,
    );
    let bound = rows
        .iter()
        .flat_map(|r| r.pc)
        .map(f64::abs)
        .fold(3f64, f64::max)
        .ceil();
    let map = |p: [f64; 2]| {
        (
            (450. + p[0] / bound * 340.) as i32,
            (450. - p[1] / bound * 340.) as i32,
        )
    };
    let ink = Rgba([80, 80, 80, 255]);
    for k in 0..8 {
        let angle = k as f64 * std::f64::consts::FRAC_PI_4;
        line(
            &mut img,
            map([0., 0.]),
            map([bound * angle.cos(), bound * angle.sin()]),
            Rgba([210, 210, 210, 255]),
        );
    }
    for k in 0..8 {
        let angle = (k as f64 + 0.5) * std::f64::consts::FRAC_PI_4;
        let (x, y) = map([bound * 0.88 * angle.cos(), bound * 0.88 * angle.sin()]);
        label(
            &mut img,
            &format!("Phase {}", first.phase_order[k]),
            x - 25,
            y,
        );
    }
    for k in 0..360 {
        let a = k as f64 * std::f64::consts::PI / 180.;
        let b = (k + 1) as f64 * std::f64::consts::PI / 180.;
        line(
            &mut img,
            map([a.cos(), a.sin()]),
            map([b.cos(), b.sin()]),
            ink,
        );
    }
    for k in -5..=5 {
        let value = bound * k as f64 / 5.;
        let (x, y) = map([value, value]);
        label(&mut img, &format!("{value:.1}"), x, 800);
        label(&mut img, &format!("{value:.1}"), 70, y);
    }
    for (at, pair) in rows.windows(2).enumerate() {
        let t = at as f64 / (rows.len() - 1).max(1) as f64;
        let shade = Rgba([(230. * t) as u8, 80, (230. * (1. - t)) as u8, 255]);
        line(&mut img, map(pair[0].pc), map(pair[1].pc), shade);
    }
    for (at, row) in rows.iter().enumerate() {
        let (x, y) = map(row.pc);
        for dx in -2..=2 {
            for dy in -2..=2 {
                line(
                    &mut img,
                    (x + dx, y + dy),
                    (x + dx, y + dy),
                    Rgba([20, 20, 20, 255]),
                );
            }
        }
        if at == 0 || at + 1 == rows.len() || at % 7 == 0 {
            label(&mut img, &row.date[5..], x + 5, y + 5);
        }
    }
    label(&mut img, "PC1 / training PC standard deviation", 290, 840);
    label(
        &mut img,
        "Vertical: PC2 / training PC standard deviation; blue start, red end",
        150,
        865,
    );
    label(&mut img, &qualification.chars().take(136).collect::<String>(), 30, 885);
    write(
        &img,
        &out.join(format!(
            "{}-{}-phase.png",
            slug(&first.diagnostic)?,
            slug(&first.member)?
        )),
    )
}
fn render(input: &str, out: &Path) -> Result<(), String> {
    let receipt: Receipt =
        serde_json::from_slice(&std::fs::read(input).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if receipt.schema != "gpuwm-hex.s2s-mjo/v1"
        || receipt.status != "complete"
        || receipt.qualification.is_empty()
    {
        return Err("unqualified S2S diagnostic receipt".into());
    }
    let mut groups: BTreeMap<(&str, &str), Vec<&Row>> = BTreeMap::new();
    for row in &receipt.series {
        if row.pc.iter().any(|v| !v.is_finite())
            || chrono::NaiveDate::parse_from_str(&row.date, "%Y-%m-%d").is_err()
            || row.date.len() != 10
            || !row.date.is_ascii()
            || row.domain.iter().any(|v| !v.is_finite())
            || row.longitude_bins == 0
            || row.latitude_bins == 0
            || row.phase_order.len() != 8
            || row.hovmoller.len() != row.fields.len()
            || row.latitude_hovmoller.len() != row.fields.len()
            || row
                .hovmoller
                .iter()
                .any(|v| v.len() != row.longitude_bins || v.iter().any(|x| !x.is_finite()))
            || row
                .latitude_hovmoller
                .iter()
                .any(|v| v.len() != row.latitude_bins || v.iter().any(|x| !x.is_finite()))
        {
            return Err("invalid S2S chart dimensions/values".into());
        }
        groups
            .entry((&row.diagnostic, &row.member))
            .or_default()
            .push(row);
    }
    if groups.is_empty() {
        return Err("no S2S diagnostic series".into());
    }
    for rows in groups.values_mut() {
        rows.sort_by(|a, b| a.date.cmp(&b.date));
        let first = rows[0];
        if rows.windows(2).any(|w| w[0].date == w[1].date)
            || rows.iter().any(|r| {
                r.phase_order != first.phase_order
                    || r.fields != first.fields
                    || r.domain != first.domain
                    || r.longitude_bins != first.longitude_bins
                    || r.latitude_bins != first.latitude_bins
            })
        {
            return Err("inconsistent or duplicate S2S chart rows".into());
        }
    }
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    groups.par_iter().try_for_each(|(_, rows)| {
        phase(rows, out, &receipt.reference_source, &receipt.qualification)?;
        for field in 0..rows[0].fields.len() {
            hov(rows, field, false, out, &receipt.reference_source, &receipt.qualification)?;
            if rows[0].latitude_bins > 1 {
                hov(rows, field, true, out, &receipt.reference_source, &receipt.qualification)?;
            }
        }
        Ok(())
    })
}
pub fn try_cli(args: &[String]) -> Option<Result<(), String>> {
    if args.first().map(String::as_str) == Some("--s2s-review-sheet") {
        return Some(match args {
            [_, plan] => review_sheet(plan),
            _ => Err("rw_wrfbatch --s2s-review-sheet PLAN.json".into()),
        });
    }
    if args.first().map(String::as_str) == Some("--s2s-diagnostics-abi") {
        println!("{ABI}");
        return Some(Ok(()));
    }
    if args.first().map(String::as_str) != Some("--s2s-diagnostics") {
        return None;
    }
    Some(match args {
        [_, input, flag, out] if flag == "--out-dir" => render(input, &PathBuf::from(out)),
        _ => Err("rw_wrfbatch --s2s-diagnostics RECEIPT.json --out-dir DIR".into()),
    })
}

/// Native contact sheets retain the generated image names for visual review.
/// Input pictures have already passed the normal production renderer.
fn review_sheet(plan: &str) -> Result<(), String> {
    #[derive(Deserialize)]
    struct Plan { images: Vec<PathBuf>, out_dir: PathBuf }
    let p: Plan = serde_json::from_slice(&std::fs::read(plan).map_err(|e|e.to_string())?)
        .map_err(|e|e.to_string())?;
    if p.images.is_empty() { return Err("no review images".into()); }
    std::fs::create_dir_all(&p.out_dir).map_err(|e|e.to_string())?;
    p.images.chunks(12).enumerate().collect::<Vec<_>>().par_iter()
        .try_for_each(|(page, paths)| {
            let mut sheet = RgbaImage::from_pixel(1560, 1760, Rgba([255,255,255,255]));
            for (index,path) in paths.iter().enumerate() {
                let source = image::open(path).map_err(|e|format!("{}: {e}",path.display()))?;
                let tile = source.thumbnail(510, 405).to_rgba8();
                let x = (index%3) as i64 * 520;
                let y = (index/3) as i64 * 440;
                label(&mut sheet, &path.file_name().ok_or("missing image name")?.to_string_lossy(), x as i32+5,y as i32+8);
                image::imageops::overlay(&mut sheet,&tile,x+5,y+28);
            }
            write(&sheet,&p.out_dir.join(format!("review-{:03}.png",page+1)))
        })
}
