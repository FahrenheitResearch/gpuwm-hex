//! Exact float32 host references for the regional contract. No fused operations.
use std::collections::HashMap;
use std::io::{self, Read, Write};

struct Fields {
    f: HashMap<String, Vec<f32>>,
    i: HashMap<String, Vec<i64>>,
    nc: usize, ne: usize, nl: usize, me: usize, width: usize, nt: usize,
}
impl Fields {
    fn f(&self, key: &str) -> &[f32] { &self.f[key] }
    fn i(&self, key: &str) -> &[i64] { &self.i[key] }
}
fn u64_read(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b))
}
fn index(v: i64, n: usize) -> usize {
    // NumPy negative indexing is part of the authority's dead-lane convention.
    let v = if v < 0 { n as i64 + v } else { v };
    assert!(v >= 0 && v < n as i64, "active connectivity index out of range");
    v as usize
}
fn horizontal(x: &Fields) -> Vec<f32> {
    let a_f_adv = x.f("adv");
    let a_f_adv3 = x.f("adv3");
    let a_f_dv = x.f("dv");
    let a_f_stage = x.f("stage");
    let a_f_velocity = x.f("velocity");
    let a_i_adv_cells = x.i("adv_cells");
    let a_i_cells_on_edge = x.i("cells_on_edge");
    let a_i_mask_edge = x.i("mask_edge");
    let a_i_n_adv = x.i("n_adv");
    let mut out = vec![0.0; x.nt * x.nl * x.ne];
    for e in 0..x.ne {
        let mask = a_i_mask_edge[e];
        if mask > 5 { continue; }
        for k in 0..x.nl {
            let velocity = a_f_velocity[k*x.ne+e];
            for s in 0..x.nt {
                let pos = (s*x.nl+k)*x.ne+e;
                if mask >= 4 {
                    let c0 = index(a_i_cells_on_edge[2*e], x.nc);
                    let c1 = index(a_i_cells_on_edge[2*e+1], x.nc);
                    let dir = 0.5_f32.copysign(velocity);
                    let up = a_f_dv[e] * (dir+0.5).abs();
                    let down = a_f_dv[e] * (dir-0.5).abs();
                    out[pos] = up*a_f_stage[(s*x.nl+k)*x.nc+c0]
                        + down*a_f_stage[(s*x.nl+k)*x.nc+c1];
                } else {
                    let sign = 1.0_f32.copysign(velocity);
                    let n = a_i_n_adv[e];
                    assert!(n >= 0 && n as usize <= x.width, "invalid stencil width");
                    for slot in 0..n as usize {
                        let j = e*x.width+slot;
                        let c = a_i_adv_cells[j];
                        assert!(c >= 0 && c < x.nc as i64, "active adv_cells_for_edge entry is out of range");
                        let weight = a_f_adv[j] + 0.25*sign*a_f_adv3[j];
                        out[pos] += weight*a_f_stage[(s*x.nl+k)*x.nc+c as usize];
                    }
                }
            }
        }
    }
    out
}
fn vertical(x: &Fields) -> Vec<f32> {
    let a_f_fzm = x.f("fzm");
    let a_f_fzp = x.f("fzp");
    let a_f_stage = x.f("stage");
    let a_f_ww = x.f("ww");
    let mut out = vec![0.0; (x.nl+1)*x.nc];
    for c in 0..x.nc {
        for k in 1..x.nl {
            let vel = a_f_ww[k*x.nc+c];
            let q = |j: usize| a_f_stage[j*x.nc+c];
            out[k*x.nc+c] = if k == 1 || k == x.nl-1 {
                vel * (a_f_fzm[k]*q(k) + a_f_fzp[k]*q(k-1))
            } else {
                let base = vel*(7.0*(q(k)+q(k-1))-(q(k+1)+q(k-2)))/12.0;
                let correction = 0.25*vel.abs()*((q(k+1)-q(k-2))-3.0*(q(k)-q(k-1)))/12.0;
                base+correction
            };
        }
    }
    out
}
fn finish(x: &Fields) -> (Vec<f32>, Vec<f32>) {
    let a_f_edge_values = x.f("edge_values");
    let a_f_flux = x.f("flux");
    let a_f_inv_area = x.f("inv_area");
    let a_f_old = x.f("old");
    let a_f_rdzw = x.f("rdzw");
    let a_f_rho_new = x.f("rho_new");
    let a_f_rho_old = x.f("rho_old");
    let a_f_signs = x.f("signs");
    let a_f_stage = x.f("stage");
    let a_f_velocity = x.f("velocity");
    let a_i_counts = x.i("counts");
    let a_i_edges = x.i("edges");
    let a_i_mask_cell = x.i("mask_cell");
    let mut out = a_f_stage.to_vec();
    let mut density = vec![0.0; x.nl*x.nc];
    for (j, value) in density.iter_mut().enumerate() {
        *value = (1.0_f32-1.0)*a_f_rho_old[j]+1.0*a_f_rho_new[j];
        assert!(*value > 0.0, "time-interpolated rho_zz must stay positive");
    }
    for s in 0..x.nt {
        for c in 0..x.nc {
            if a_i_mask_cell[c] > 5 { continue; }
            for k in 0..x.nl {
                let mut tendency = 0.0;
                for slot in 0..a_i_counts[c] as usize {
                    let j = c*x.me+slot;
                    let e = index(a_i_edges[j], x.ne);
                    tendency += a_f_signs[j]*a_f_velocity[k*x.ne+e]
                        *a_f_edge_values[(s*x.nl+k)*x.ne+e];
                }
                tendency = tendency*a_f_inv_area[c]+0.0;
                let j = (s*x.nl+k)*x.nc+c;
                let numerator = a_f_old[j]*a_f_rho_old[k*x.nc+c]+120.0
                    *(tendency-a_f_rdzw[k]*(a_f_flux[((s*(x.nl+1))+k+1)*x.nc+c]
                        -a_f_flux[((s*(x.nl+1))+k)*x.nc+c]));
                let inverse = 1.0/density[k*x.nc+c];
                out[j] = numerator*inverse;
            }
        }
    }
    (out, density)
}
fn acoustic(x: &Fields) -> Vec<Vec<f32>> {
    let a_f_a_tri = x.f("a_tri");
    let a_f_alpha_tri = x.f("alpha_tri");
    let a_f_cofrz = x.f("cofrz");
    let a_f_coftz = x.f("coftz");
    let a_f_cofwr = x.f("cofwr");
    let a_f_cofwt = x.f("cofwt");
    let a_f_cofwz = x.f("cofwz");
    let a_f_cqu = x.f("cqu");
    let a_f_dss = x.f("dss");
    let a_f_dv = x.f("dv");
    let a_f_etm = x.f("etm");
    let a_f_etp = x.f("etp");
    let a_f_ewm = x.f("ewm");
    let a_f_ewp = x.f("ewp");
    let a_f_exner = x.f("exner");
    let a_f_fzm = x.f("fzm");
    let a_f_fzp = x.f("fzp");
    let a_f_gamma_tri = x.f("gamma_tri");
    let a_f_inv_area = x.f("inv_area");
    let a_f_inv_dc = x.f("inv_dc");
    let a_f_rdzw = x.f("rdzw");
    let a_f_rho_pp = x.f("rho_pp");
    let a_f_rho_zz = x.f("rho_zz");
    let a_f_rtheta_pp = x.f("rtheta_pp");
    let a_f_ru_avg = x.f("ru_avg");
    let a_f_ru_p = x.f("ru_p");
    let a_f_rw = x.f("rw");
    let a_f_rw_p = x.f("rw_p");
    let a_f_rw_save = x.f("rw_save");
    let a_f_signs = x.f("signs");
    let a_f_spec_cell = x.f("spec_cell");
    let a_f_spec_edge = x.f("spec_edge");
    let a_f_tend_rho = x.f("tend_rho");
    let a_f_tend_rt = x.f("tend_rt");
    let a_f_tend_ru = x.f("tend_ru");
    let a_f_tend_rw = x.f("tend_rw");
    let a_f_theta = x.f("theta");
    let a_f_w = x.f("w");
    let a_f_ww_avg = x.f("ww_avg");
    let a_f_zxu = x.f("zxu");
    let a_f_zz = x.f("zz");
    let a_i_cells_on_edge = x.i("cells_on_edge");
    let a_i_counts = x.i("counts");
    let a_i_edges = x.i("edges");
    let (nc, ne, nl) = (x.nc, x.ne, x.nl);
    let dt = 120.0_f32/3.0/6.0;
    let c2 = 1004.5_f32*(287.0_f32/(1004.5_f32-287.0));
    let mut ru = a_f_ru_p.to_vec();
    let mut avg = a_f_ru_avg.to_vec();
    let mut rw = a_f_rw_p.to_vec();
    let mut ww = a_f_ww_avg.to_vec();
    let mut rho = a_f_rho_pp.to_vec();
    let mut rt = a_f_rtheta_pp.to_vec();
    for e in 0..ne {
        let c0 = a_i_cells_on_edge[2*e];
        let c1 = a_i_cells_on_edge[2*e+1];
        for k in 0..nl {
            let j = k*ne+e;
            if c0 < 0 || c1 < 0 {
                assert_eq!(a_f_spec_edge[e], 1.0, "acoustic pressure gradient needs both edge cells");
                ru[j] += dt*a_f_tend_ru[j];
            } else {
                let a = k*nc+index(c0,nc);
                let b = k*nc+index(c1,nc);
                let mut pg = ((rt[b]-rt[a])*a_f_inv_dc[e])/(0.5*(a_f_zz[b]+a_f_zz[a]));
                pg = a_f_cqu[j]*0.5*c2*(a_f_exner[a]+a_f_exner[b])*pg;
                pg += 0.5*a_f_zxu[j]*9.80616*(rho[a]+rho[b]);
                ru[j] += dt*(a_f_tend_ru[j]-(1.0-a_f_spec_edge[e])*pg);
            }
            avg[j] += ru[j];
        }
    }
    for c in 0..nc {
        if a_f_spec_cell[c] != 0.0 {
            for k in 0..nl {
                let j = k*nc+c;
                rho[j] += dt*a_f_tend_rho[j];
                rt[j] += dt*a_f_tend_rt[j];
                rw[j] += dt*a_f_tend_rw[j];
                ww[j] += a_f_ewp[k]*rw[j];
            }
            continue;
        }
        let mut rs = vec![0.0_f32; nl];
        let mut ts = vec![0.0_f32; nl];
        for slot in 0..a_i_counts[c] as usize {
            let slot_id = c*x.me+slot;
            let e = index(a_i_edges[slot_id],ne);
            let c0 = index(a_i_cells_on_edge[2*e],nc);
            let c1 = index(a_i_cells_on_edge[2*e+1],nc);
            for k in 0..nl {
                let flux = a_f_signs[slot_id]*dt*a_f_dv[e]*ru[k*ne+e]*a_f_inv_area[c];
                rs[k] -= flux;
                ts[k] -= flux*0.5*(a_f_theta[k*nc+c1]+a_f_theta[k*nc+c0]);
            }
        }
        for k in 0..nl {
            let j = k*nc+c;
            let p = j+nc;
            rs[k] = rho[j]+dt*a_f_tend_rho[j]+rs[k]-dt*a_f_cofrz[k]
                *(a_f_ewm[k+1]*rw[p]-a_f_ewm[k]*rw[j]);
            ts[k] = rt[j]+dt*a_f_tend_rt[j]+ts[k]-dt*a_f_rdzw[k]
                *(a_f_ewm[k+1]*a_f_coftz[p]*rw[p]-a_f_ewm[k]*a_f_coftz[j]*rw[j]);
        }
        for k in 1..nl {
            let j = k*nc+c;
            let m = j-nc;
            ww[j] += a_f_ewm[k]*rw[j];
            rw[j] += dt*(a_f_tend_rw[j]
                -a_f_cofwz[j]*((a_f_etp[k]*a_f_zz[j]*ts[k]
                    -a_f_etp[k-1]*a_f_zz[m]*ts[k-1])
                    +(a_f_etm[k]*a_f_zz[j]*rt[j]
                    -a_f_etm[k-1]*a_f_zz[m]*rt[m]))
                -a_f_cofwr[j]*((a_f_etp[k]*rs[k]+a_f_etp[k-1]*rs[k-1])
                    +(a_f_etm[k]*rho[j]+a_f_etm[k-1]*rho[m]))
                +a_f_cofwt[j]*(a_f_etp[k]*ts[k]+a_f_etm[k]*rt[j])
                +a_f_cofwt[m]*(a_f_etp[k-1]*ts[k-1]+a_f_etm[k-1]*rt[m]));
        }
        for k in 1..nl {
            let j = k*nc+c;
            rw[j] = (rw[j]-(dt*dt)*a_f_a_tri[j]*rw[j-nc])*a_f_alpha_tri[j];
        }
        for k in (0..nl).rev() {
            let j = k*nc+c;
            rw[j] -= a_f_gamma_tri[j]*rw[j+nc];
        }
        for k in 1..nl {
            let j = k*nc+c;
            let m = j-nc;
            let saved = a_f_rw_save[j]-a_f_rw[j];
            let damping = a_f_dss[j];
            // native left-to-right order: ((dts*dss)*zz_interface)*rho_interface*w,
            // as hexcore.acoustic_v841 follows since dycore-match (the product
            // of the two interface means first differed by 2 binary32 ULP)
            let zz_interface = a_f_fzm[k]*a_f_zz[j]+a_f_fzp[k]*a_f_zz[m];
            let rho_interface = a_f_fzm[k]*a_f_rho_zz[j]+a_f_fzp[k]*a_f_rho_zz[m];
            rw[j] = (rw[j]+saved-dt*damping*zz_interface*rho_interface*a_f_w[j])/(1.0+dt*damping)-saved;
            ww[j] += a_f_ewp[k]*rw[j];
        }
        for k in 0..nl {
            let j = k*nc+c;
            rho[j] = rs[k]-dt*a_f_cofrz[k]*(a_f_ewp[k+1]*rw[j+nc]-a_f_ewp[k]*rw[j]);
            rt[j] = ts[k]-dt*a_f_rdzw[k]*(a_f_ewp[k+1]*a_f_coftz[j+nc]*rw[j+nc]
                -a_f_ewp[k]*a_f_coftz[j]*rw[j]);
        }
    }
    vec![ru, avg, rw, ww, rho, rt]
}
fn main() -> io::Result<()> {
    let mut input = io::BufReader::new(io::stdin().lock());
    let mut magic = [0;8]; input.read_exact(&mut magic)?;
    if magic != *b"HEXREF01" { return Err(io::Error::other("wrong contract host-prep protocol")); }
    let mode = u64_read(&mut input)?;
    let mut dim = [0_usize;6];
    for d in &mut dim { *d = u64_read(&mut input)? as usize; }
    let mut x = Fields { f: HashMap::new(), i: HashMap::new(), nc:dim[0], ne:dim[1],
        nl:dim[2], me:dim[3], width:dim[4], nt:dim[5] };
    let arrays = u64_read(&mut input)?;
    for _ in 0..arrays {
        let len = u64_read(&mut input)? as usize;
        let mut key = vec![0;len]; input.read_exact(&mut key)?;
        let key = String::from_utf8(key).map_err(io::Error::other)?;
        let mut kind = [0]; input.read_exact(&mut kind)?;
        let len = u64_read(&mut input)? as usize;
        match kind[0] {
            4 => {
                let mut bytes = vec![0;len*4]; input.read_exact(&mut bytes)?;
                x.f.insert(key, bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect());
            }
            8 => {
                let mut bytes = vec![0;len*8]; input.read_exact(&mut bytes)?;
                x.i.insert(key, bytes.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect());
            }
            _ => return Err(io::Error::other("wrong contract array type")),
        }
    }
    let outputs = match mode {
        0 => vec![horizontal(&x)], 1 => vec![vertical(&x)],
        2 => { let (out, rho) = finish(&x); vec![out,rho] },
        3 => acoustic(&x),
        _ => return Err(io::Error::other("unknown contract reference mode")),
    };
    let mut output = io::BufWriter::new(io::stdout().lock());
    for array in outputs { for value in array { output.write_all(&value.to_le_bytes())?; } }
    output.flush()
}
