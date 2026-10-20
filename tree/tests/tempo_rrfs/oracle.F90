module tempo_oracle
use iso_c_binding
use module_mp_tempo_params
use module_mp_tempo_cfgs
use module_mp_tempo_driver
use module_mp_tempo_main
implicit none
type(ty_tempo_cfgs),save :: cfg
integer,save :: carrier_presence=31
contains
subroutine oracle_init(flags) bind(c)
integer(c_int),intent(in)::flags(4)
call tempo_init(aerosolaware_flag=flags(1)/=0,hailaware_flag=flags(2)/=0,ml_for_bl_nc_flag=flags(3)/=0,refl10cm_from_melting_flag=flags(4)/=0,tempo_cfgs=cfg)
end subroutine
! RRFS TEMPO/src/module_mp_tempo_driver.F90:439-454 @ 1d6e370e.
subroutine oracle_emissions(nz,nc,dt,nwfa,nwfa2d) bind(c)
integer(c_int),value::nz,nc
real(c_float),value::dt
real(c_float),intent(inout)::nwfa(nc,nz,1)
real(c_float),intent(in)::nwfa2d(nc,1)
call tempo_aerosol_surface_emissions(dt,nwfa,nwfa2d,1,nc,1,1,1,nz,1)
end subroutine
subroutine oracle_blob(out) bind(c)
real(c_double),intent(out)::out(*)
integer::i,n
i=1
out(i)=real(av_r,c_double)
i=i+1
out(i)=real(bv_r,c_double)
i=i+1
out(i)=real(av_s,c_double)
i=i+1
out(i)=real(bv_s,c_double)
i=i+1
out(i)=real(fv_s,c_double)
i=i+1
out(i)=real(bv_c,c_double)
i=i+1
out(i)=real(bv_i,c_double)
i=i+1
out(i)=real(av_g_old,c_double)
i=i+1
out(i)=real(bv_g_old,c_double)
i=i+1
out(i)=real(av_g_new,c_double)
i=i+1
out(i)=real(bv_g_new,c_double)
i=i+1
out(i)=real(fv_r,c_double)
i=i+1
out(i)=real(av_c,c_double)
i=i+1
out(i)=real(a_coeff,c_double)
i=i+1
out(i)=real(b_coeff,c_double)
i=i+1
out(i)=real(av_i,c_double)
i=i+1
out(i)=real(am_s,c_double)
i=i+1
out(i)=real(bm_s,c_double)
i=i+1
out(i)=real(bm_g,c_double)
i=i+1
out(i)=real(bm_i,c_double)
i=i+1
out(i)=real(bm_r,c_double)
i=i+1
out(i)=real(rho_i,c_double)
i=i+1
out(i)=real(xm0i,c_double)
i=i+1
out(i)=real(d0c,c_double)
i=i+1
out(i)=real(d0r,c_double)
i=i+1
out(i)=real(d0s,c_double)
i=i+1
out(i)=real(d0g,c_double)
i=i+1
out(i)=real(d0r_max,c_double)
i=i+1
out(i)=real(c_cube,c_double)
i=i+1
out(i)=real(c_sqrd,c_double)
i=i+1
out(i)=real(mu_r,c_double)
i=i+1
out(i)=real(mu_s,c_double)
i=i+1
out(i)=real(mu_g,c_double)
i=i+1
out(i)=real(mu_i,c_double)
i=i+1
out(i)=real(nu_c_scale,c_double)
i=i+1
out(i)=real(nu_c_max,c_double)
i=i+1
out(i)=real(nu_c_min,c_double)
i=i+1
out(i)=real(naccn0,c_double)
i=i+1
out(i)=real(naccn1,c_double)
i=i+1
out(i)=real(nain0,c_double)
i=i+1
out(i)=real(nain1,c_double)
i=i+1
out(i)=real(nwfa_default,c_double)
i=i+1
out(i)=real(nifa_default,c_double)
i=i+1
out(i)=real(aero_max,c_double)
i=i+1
out(i)=real(hgfrz,c_double)
i=i+1
out(i)=real(nt_c_o,c_double)
i=i+1
out(i)=real(nt_c_l,c_double)
i=i+1
out(i)=real(nt_c_max,c_double)
i=i+1
out(i)=real(nt_c_min,c_double)
i=i+1
out(i)=real(tno,c_double)
i=i+1
out(i)=real(ato,c_double)
i=i+1
out(i)=real(rho_s,c_double)
i=i+1
out(i)=real(demott_nuc_ssati,c_double)
i=i+1
out(i)=real(max_ni,c_double)
i=i+1
out(i)=real(icenuc_max,c_double)
i=i+1
out(i)=real(rime_threshold,c_double)
i=i+1
out(i)=real(rime_conversion,c_double)
i=i+1
out(i)=real(ef_si,c_double)
i=i+1
out(i)=real(ef_rs,c_double)
i=i+1
out(i)=real(ef_rg,c_double)
i=i+1
out(i)=real(ef_ri,c_double)
i=i+1
out(i)=real(autocon_nr_factor,c_double)
i=i+1
out(i)=real(timestep_conversion_rime_to_rain,c_double)
i=i+1
out(i)=real(nrhg,c_double)
i=i+1
out(i)=real(nrhg1,c_double)
i=i+1
out(i)=real(min_qv,c_double)
i=i+1
out(i)=real(r1,c_double)
i=i+1
out(i)=real(r2,c_double)
i=i+1
out(i)=real(low_limit_mass_for_precip,c_double)
i=i+1
out(i)=real(eps,c_double)
i=i+1
out(i)=real(meters3_to_liters,c_double)
i=i+1
out(i)=real(gonv_min,c_double)
i=i+1
out(i)=real(gonv_max,c_double)
i=i+1
out(i)=real(t0,c_double)
i=i+1
out(i)=real(rho_w,c_double)
i=i+1
out(i)=real(sc,c_double)
i=i+1
out(i)=real(pi,c_double)
i=i+1
out(i)=real(lsub,c_double)
i=i+1
out(i)=real(lvap0,c_double)
i=i+1
out(i)=real(rv,c_double)
i=i+1
out(i)=real(rdry,c_double)
i=i+1
out(i)=real(roverrv,c_double)
i=i+1
out(i)=real(r,c_double)
i=i+1
out(i)=real(rho_not,c_double)
i=i+1
out(i)=real(rho_not0,c_double)
i=i+1
out(i)=real(cp,c_double)
i=i+1
out(i)=real(r_uni,c_double)
i=i+1
out(i)=real(kap0,c_double)
i=i+1
out(i)=real(kap1,c_double)
i=i+1
out(i)=real(lam0,c_double)
i=i+1
out(i)=real(lam1,c_double)
i=i+1
out(i)=real(nbins,c_double)
i=i+1
out(i)=real(nbc,c_double)
i=i+1
out(i)=real(nbr,c_double)
i=i+1
out(i)=real(nbs,c_double)
i=i+1
out(i)=real(nbi,c_double)
i=i+1
out(i)=real(nbg,c_double)
i=i+1
out(i)=real(ntb_i,c_double)
i=i+1
out(i)=real(ntb_i1,c_double)
i=i+1
out(i)=real(ntb_c,c_double)
i=i+1
out(i)=real(ntb_t,c_double)
i=i+1
out(i)=real(ntb_g1,c_double)
i=i+1
out(i)=real(ntb_s,c_double)
i=i+1
out(i)=real(ntb_g,c_double)
i=i+1
out(i)=real(ntb_r,c_double)
i=i+1
out(i)=real(ntb_r1,c_double)
i=i+1
out(i)=real(ntb_t1,c_double)
i=i+1
out(i)=real(ntb_in,c_double)
i=i+1
out(i)=real(ntb_arc,c_double)
i=i+1
out(i)=real(ntb_arw,c_double)
i=i+1
out(i)=real(ntb_art,c_double)
i=i+1
out(i)=real(ntb_arr,c_double)
i=i+1
out(i)=real(ntb_ark,c_double)
i=i+1
out(i)=real(idx_bg1,c_double)
i=i+1
out(i)=real(nhbins,c_double)
i=i+1
out(i)=real(radar_bins,c_double)
i=i+1
out(i:i+8)=reshape(real(rho_g,c_double),[9])
i=i+9
out(i:i+36)=reshape(real(r_c,c_double),[37])
i=i+37
out(i:i+63)=reshape(real(r_i,c_double),[64])
i=i+64
out(i:i+36)=reshape(real(r_r,c_double),[37])
i=i+37
out(i:i+36)=reshape(real(r_s,c_double),[37])
i=i+37
out(i:i+36)=reshape(real(r_g,c_double),[37])
i=i+37
out(i:i+36)=reshape(real(n0r_exp,c_double),[37])
i=i+37
out(i:i+36)=reshape(real(n0g_exp,c_double),[37])
i=i+37
out(i:i+54)=reshape(real(nt_i,c_double),[55])
i=i+55
out(i:i+54)=reshape(real(nt_in,c_double),[55])
i=i+55
out(i:i+6)=reshape(real(ta_na,c_double),[7])
i=i+7
out(i:i+8)=reshape(real(ta_ww,c_double),[9])
i=i+9
out(i:i+6)=reshape(real(ta_tk,c_double),[7])
i=i+7
out(i:i+4)=reshape(real(ta_ra,c_double),[5])
i=i+5
out(i:i+3)=reshape(real(ta_ka,c_double),[4])
i=i+4
out(i:i+9)=reshape(real(sa,c_double),[10])
i=i+10
out(i:i+9)=reshape(real(sb,c_double),[10])
i=i+10
out(i:i+8)=reshape(real(tc,c_double),[9])
i=i+9
out(i)=real(dim_nrhg,c_double)
i=i+1
out(i:i+8)=reshape(real(av_g,c_double),[9])
i=i+9
out(i:i+8)=reshape(real(bv_g,c_double),[9])
i=i+9
out(i)=real(am_i,c_double)
i=i+1
out(i)=real(am_r,c_double)
i=i+1
out(i:i+8)=reshape(real(am_g,c_double),[9])
i=i+9
out(i)=real(lfus,c_double)
i=i+1
out(i)=real(olfus,c_double)
i=i+1
out(i)=real(orv,c_double)
i=i+1
out(i)=real(ar_volume,c_double)
i=i+1
out(i)=real(sc3,c_double)
i=i+1
out(i)=real(d0i,c_double)
i=i+1
out(i)=real(xm0s,c_double)
i=i+1
out(i)=real(xm0g,c_double)
i=i+1
out(i)=real(obmi,c_double)
i=i+1
out(i)=real(obmr,c_double)
i=i+1
out(i)=real(oams,c_double)
i=i+1
out(i)=real(obms,c_double)
i=i+1
out(i)=real(ocms,c_double)
i=i+1
out(i:i+8)=reshape(real(oamg,c_double),[9])
i=i+9
out(i:i+8)=reshape(real(ocmg,c_double),[9])
i=i+9
out(i)=real(obmg,c_double)
i=i+1
out(i:i+74)=reshape(real(cce,c_double),[75])
i=i+75
out(i:i+74)=reshape(real(ccg,c_double),[75])
i=i+75
out(i:i+14)=reshape(real(ocg1,c_double),[15])
i=i+15
out(i:i+14)=reshape(real(ocg2,c_double),[15])
i=i+15
out(i:i+6)=reshape(real(cie,c_double),[7])
i=i+7
out(i:i+6)=reshape(real(cig,c_double),[7])
i=i+7
out(i)=real(oig1,c_double)
i=i+1
out(i)=real(oig2,c_double)
i=i+1
out(i:i+12)=reshape(real(cre,c_double),[13])
i=i+13
out(i:i+12)=reshape(real(crg,c_double),[13])
i=i+13
out(i)=real(ore1,c_double)
i=i+1
out(i)=real(org1,c_double)
i=i+1
out(i)=real(org2,c_double)
i=i+1
out(i)=real(org3,c_double)
i=i+1
out(i:i+16)=reshape(real(cse,c_double),[17])
i=i+17
out(i:i+16)=reshape(real(csg,c_double),[17])
i=i+17
out(i:i+107)=reshape(real(cge,c_double),[108])
i=i+108
out(i:i+107)=reshape(real(cgg,c_double),[108])
i=i+108
out(i)=real(oge1,c_double)
i=i+1
out(i)=real(ogg1,c_double)
i=i+1
out(i)=real(ogg2,c_double)
i=i+1
out(i)=real(ogg3,c_double)
i=i+1
out(i)=real(t1_qr_qc,c_double)
i=i+1
out(i)=real(t1_qr_qi,c_double)
i=i+1
out(i)=real(t2_qr_qi,c_double)
i=i+1
out(i)=real(t1_qs_qc,c_double)
i=i+1
out(i)=real(t1_qs_qi,c_double)
i=i+1
out(i)=real(t1_qr_ev,c_double)
i=i+1
out(i)=real(t2_qr_ev,c_double)
i=i+1
out(i)=real(t1_qs_sd,c_double)
i=i+1
out(i)=real(t2_qs_sd,c_double)
i=i+1
out(i)=real(t1_qs_me,c_double)
i=i+1
out(i)=real(t2_qs_me,c_double)
i=i+1
out(i)=real(t1_qg_sd,c_double)
i=i+1
out(i)=real(t1_qg_me,c_double)
i=i+1
out(i)=real(nic2,c_double)
i=i+1
out(i)=real(nii2,c_double)
i=i+1
out(i)=real(nii3,c_double)
i=i+1
out(i)=real(nir2,c_double)
i=i+1
out(i)=real(nir3,c_double)
i=i+1
out(i)=real(nis2,c_double)
i=i+1
out(i)=real(nig2,c_double)
i=i+1
out(i)=real(nig3,c_double)
i=i+1
out(i)=real(niin2,c_double)
i=i+1
out(i)=real(nic1,c_double)
i=i+1
out(i:i+99)=reshape(real(dc,c_double),[100])
i=i+100
out(i:i+99)=reshape(real(dtc,c_double),[100])
i=i+100
out(i:i+99)=reshape(real(di,c_double),[100])
i=i+100
out(i:i+99)=reshape(real(dti,c_double),[100])
i=i+100
out(i:i+99)=reshape(real(dr,c_double),[100])
i=i+100
out(i:i+99)=reshape(real(dtr,c_double),[100])
i=i+100
out(i:i+99)=reshape(real(ds,c_double),[100])
i=i+100
out(i:i+99)=reshape(real(dts,c_double),[100])
i=i+100
out(i:i+99)=reshape(real(dg,c_double),[100])
i=i+100
out(i:i+99)=reshape(real(dtg,c_double),[100])
i=i+100
out(i:i+99)=reshape(real(t_nc,c_double),[100])
i=i+100
if(allocated(hbins)) then
out(i:i+49)=reshape(real(hbins,c_double),[50])
else
out(i:i+49)=0.0_c_double
endif
i=i+50
if(allocated(dhbins)) then
out(i:i+49)=reshape(real(dhbins,c_double),[50])
else
out(i:i+49)=0.0_c_double
endif
i=i+50
if(allocated(sbins_radar)) then
out(i:i+49)=reshape(real(sbins_radar,c_double),[50])
else
out(i:i+49)=0.0_c_double
endif
i=i+50
if(allocated(dsbins_radar)) then
out(i:i+49)=reshape(real(dsbins_radar,c_double),[50])
else
out(i:i+49)=0.0_c_double
endif
i=i+50
if(allocated(gbins_radar)) then
out(i:i+49)=reshape(real(gbins_radar,c_double),[50])
else
out(i:i+49)=0.0_c_double
endif
i=i+50
if(allocated(dgbins_radar)) then
out(i:i+49)=reshape(real(dgbins_radar,c_double),[50])
else
out(i:i+49)=0.0_c_double
endif
i=i+50
if(allocated(t_efrw)) then
out(i:i+9999)=reshape(real(t_efrw,c_double),[10000])
else
out(i:i+9999)=0.0_c_double
endif
i=i+10000
if(allocated(t_efsw)) then
out(i:i+9999)=reshape(real(t_efsw,c_double),[10000])
else
out(i:i+9999)=0.0_c_double
endif
i=i+10000
if(allocated(tpc_wev)) then
out(i:i+369999)=reshape(real(tpc_wev,c_double),[370000])
else
out(i:i+369999)=0.0_c_double
endif
i=i+370000
if(allocated(tnc_wev)) then
out(i:i+369999)=reshape(real(tnc_wev,c_double),[370000])
else
out(i:i+369999)=0.0_c_double
endif
i=i+370000
if(allocated(tnccn_act)) then
out(i:i+8819)=reshape(real(tnccn_act,c_double),[8820])
else
out(i:i+8819)=0.0_c_double
endif
i=i+8820
if(allocated(tps_iaus)) then
out(i:i+3519)=reshape(real(tps_iaus,c_double),[3520])
else
out(i:i+3519)=0.0_c_double
endif
i=i+3520
if(allocated(tni_iaus)) then
out(i:i+3519)=reshape(real(tni_iaus,c_double),[3520])
else
out(i:i+3519)=0.0_c_double
endif
i=i+3520
if(allocated(tpi_ide)) then
out(i:i+3519)=reshape(real(tpi_ide,c_double),[3520])
else
out(i:i+3519)=0.0_c_double
endif
i=i+3520
if(allocated(tpi_qcfz)) then
out(i:i+9157499)=reshape(real(tpi_qcfz,c_double),[9157500])
else
out(i:i+9157499)=0.0_c_double
endif
i=i+9157500
if(allocated(tni_qcfz)) then
out(i:i+9157499)=reshape(real(tni_qcfz,c_double),[9157500])
else
out(i:i+9157499)=0.0_c_double
endif
i=i+9157500
if(allocated(tpi_qrfz)) then
out(i:i+3388274)=reshape(real(tpi_qrfz,c_double),[3388275])
else
out(i:i+3388274)=0.0_c_double
endif
i=i+3388275
if(allocated(tpg_qrfz)) then
out(i:i+3388274)=reshape(real(tpg_qrfz,c_double),[3388275])
else
out(i:i+3388274)=0.0_c_double
endif
i=i+3388275
if(allocated(tni_qrfz)) then
out(i:i+3388274)=reshape(real(tni_qrfz,c_double),[3388275])
else
out(i:i+3388274)=0.0_c_double
endif
i=i+3388275
if(allocated(tnr_qrfz)) then
out(i:i+3388274)=reshape(real(tnr_qrfz,c_double),[3388275])
else
out(i:i+3388274)=0.0_c_double
endif
i=i+3388275
if(allocated(tcs_racs1)) then
out(i:i+455876)=reshape(real(tcs_racs1,c_double),[455877])
else
out(i:i+455876)=0.0_c_double
endif
i=i+455877
if(allocated(tmr_racs1)) then
out(i:i+455876)=reshape(real(tmr_racs1,c_double),[455877])
else
out(i:i+455876)=0.0_c_double
endif
i=i+455877
if(allocated(tcs_racs2)) then
out(i:i+455876)=reshape(real(tcs_racs2,c_double),[455877])
else
out(i:i+455876)=0.0_c_double
endif
i=i+455877
if(allocated(tmr_racs2)) then
out(i:i+455876)=reshape(real(tmr_racs2,c_double),[455877])
else
out(i:i+455876)=0.0_c_double
endif
i=i+455877
if(allocated(tcr_sacr1)) then
out(i:i+455876)=reshape(real(tcr_sacr1,c_double),[455877])
else
out(i:i+455876)=0.0_c_double
endif
i=i+455877
if(allocated(tms_sacr1)) then
out(i:i+455876)=reshape(real(tms_sacr1,c_double),[455877])
else
out(i:i+455876)=0.0_c_double
endif
i=i+455877
if(allocated(tcr_sacr2)) then
out(i:i+455876)=reshape(real(tcr_sacr2,c_double),[455877])
else
out(i:i+455876)=0.0_c_double
endif
i=i+455877
if(allocated(tms_sacr2)) then
out(i:i+455876)=reshape(real(tms_sacr2,c_double),[455877])
else
out(i:i+455876)=0.0_c_double
endif
i=i+455877
if(allocated(tnr_racs1)) then
out(i:i+455876)=reshape(real(tnr_racs1,c_double),[455877])
else
out(i:i+455876)=0.0_c_double
endif
i=i+455877
if(allocated(tnr_racs2)) then
out(i:i+455876)=reshape(real(tnr_racs2,c_double),[455877])
else
out(i:i+455876)=0.0_c_double
endif
i=i+455877
if(allocated(tnr_sacr1)) then
out(i:i+455876)=reshape(real(tnr_sacr1,c_double),[455877])
else
out(i:i+455876)=0.0_c_double
endif
i=i+455877
if(allocated(tnr_sacr2)) then
out(i:i+455876)=reshape(real(tnr_sacr2,c_double),[455877])
else
out(i:i+455876)=0.0_c_double
endif
i=i+455877
if(allocated(tcg_racg)) then
out(i:i+16867448)=reshape(real(tcg_racg,c_double),[16867449])
else
out(i:i+16867448)=0.0_c_double
endif
i=i+16867449
if(allocated(tmr_racg)) then
out(i:i+16867448)=reshape(real(tmr_racg,c_double),[16867449])
else
out(i:i+16867448)=0.0_c_double
endif
i=i+16867449
if(allocated(tcr_gacr)) then
out(i:i+16867448)=reshape(real(tcr_gacr,c_double),[16867449])
else
out(i:i+16867448)=0.0_c_double
endif
i=i+16867449
if(allocated(tnr_racg)) then
out(i:i+16867448)=reshape(real(tnr_racg,c_double),[16867449])
else
out(i:i+16867448)=0.0_c_double
endif
i=i+16867449
if(allocated(tnr_gacr)) then
out(i:i+16867448)=reshape(real(tnr_gacr,c_double),[16867449])
else
out(i:i+16867448)=0.0_c_double
endif
i=i+16867449
end subroutine
subroutine oracle_presence(mask) bind(c)
integer(c_int),value::mask
carrier_presence=mask
end subroutine
! Binding only: the pristine MPAS TEMPO driver receives exactly these arrays.
subroutine oracle_step(nz,dt,a,bl,diags) bind(c)
integer(c_int),value::nz
real(c_float),value::dt
real(c_float),intent(inout)::a(nz,17)
real(c_float),intent(in)::bl(nz,2)
real(c_float),intent(out)::diags(nz,15)
real(wp)::x(1,nz,1,17),b(1,nz,1,2)
real(wp),allocatable::nc(:,:,:),nwfa(:,:,:),nifa(:,:,:),ng(:,:,:),qb(:,:,:),qc_bl(:,:,:),qcfrac_bl(:,:,:)
type(ty_tempo_driver_diags)::d
x=reshape(a,shape(x));b=reshape(bl,shape(b))
if(btest(carrier_presence,0)) allocate(nc,source=x(:,:,:,13))
if(btest(carrier_presence,1)) allocate(nwfa,source=x(:,:,:,14))
if(btest(carrier_presence,2)) allocate(nifa,source=x(:,:,:,15))
if(btest(carrier_presence,3)) allocate(ng,source=x(:,:,:,16))
if(btest(carrier_presence,4)) allocate(qb,source=x(:,:,:,17))
! RRFS mpas_atmphys_driver_microphysics.F:230-232 allocates these only
! when config_tempo_ml_for_bl_nc is enabled. Unallocated actuals are absent.
if(cfg%ml_for_bl_nc_flag) then
 allocate(qc_bl,source=b(:,:,:,1));allocate(qcfrac_bl,source=b(:,:,:,2))
endif
call tempo_run(tempo_cfgs=cfg,dt=dt,itimestep=1, &
t=x(:,:,:,1),p=x(:,:,:,2),w=x(:,:,:,3),dz=x(:,:,:,4), &
qv=x(:,:,:,5),qc=x(:,:,:,6),qr=x(:,:,:,7),qi=x(:,:,:,8),qs=x(:,:,:,9),qg=x(:,:,:,10),ni=x(:,:,:,11),nr=x(:,:,:,12), &
nc=nc,nwfa=nwfa,nifa=nifa,ng=ng,qb=qb,qc_bl=qc_bl,qcfrac_bl=qcfrac_bl, &
ids=1,ide=1,jds=1,jde=1,kds=1,kde=nz,ims=1,ime=1,jms=1,jme=1,kms=1,kme=nz,its=1,ite=1,jts=1,jte=1,kts=1,kte=nz,tempo_diags=d)
a=reshape(x,shape(a))
if(allocated(nc)) a(:,13)=reshape(nc,[nz])
if(allocated(nwfa)) a(:,14)=reshape(nwfa,[nz])
if(allocated(nifa)) a(:,15)=reshape(nifa,[nz])
if(allocated(ng)) a(:,16)=reshape(ng,[nz])
if(allocated(qb)) a(:,17)=reshape(qb,[nz])
diags=0._wp
diags(1,1)=d%rain_precip(1,1)
diags(1,2)=d%ice_liquid_equiv_precip(1,1)
diags(1,3)=d%snow_liquid_equiv_precip(1,1)
diags(1,4)=d%graupel_liquid_equiv_precip(1,1)
diags(1,5)=d%frz_rain_precip(1,1)
diags(1,6)=d%frozen_fraction(1,1)
if(allocated(d%max_hail_diameter_sfc)) diags(1,7)=d%max_hail_diameter_sfc(1,1)
if(allocated(d%max_hail_diameter_column)) diags(1,8)=d%max_hail_diameter_column(1,1)
if(allocated(d%refl10cm)) diags(:,9)=d%refl10cm(1,:,1)
if(allocated(d%re_cloud)) diags(:,10)=d%re_cloud(1,:,1)
if(allocated(d%re_ice)) diags(:,11)=d%re_ice(1,:,1)
if(allocated(d%re_snow)) diags(:,12)=d%re_snow(1,:,1)
if(allocated(d%rain_med_vol_diam)) diags(:,13)=d%rain_med_vol_diam(1,:,1)
if(allocated(d%graupel_med_vol_diam)) diags(:,14)=d%graupel_med_vol_diam(1,:,1)
if(allocated(d%cloud_number_mixing_ratio)) diags(:,15)=d%cloud_number_mixing_ratio(1,:,1)
end subroutine
subroutine oracle_ml(nz,a,stages) bind(c)
use module_mp_tempo_ml
integer(c_int),value::nz
real(c_float)::a(nz,7),stages(nz,33)
real(wp),allocatable::raw(:,:),trans(:,:),layer(:,:),output(:,:)
type(ty_tempo_ml_data)::data
type(ty_tempo_ml_data)::saved(1)
integer::i
call save_or_read_ml_data(ml_data_out=saved)
data=saved(1)
allocate(raw(7,nz));raw=transpose(a)
allocate(trans(7,nz))
do i=1,7
trans(i,:)=(raw(i,:)-data%transform_mean(i))/sqrt(data%transform_var(i))
enddo
layer=matmul(data%weights00,trans)
layer=layer+spread(data%bias00,2,nz)
layer=max(layer,0._wp)
output=matmul(data%weights01,layer)+data%bias01(1)
stages(:,1:7)=transpose(trans);stages(:,8:31)=transpose(layer);stages(:,32)=max(output(1,:),0._wp)
call tempo_ml_predict_cloud_number(a(:,1),a(:,2),a(:,3),a(:,4),a(:,5),a(:,6),a(:,7),stages(:,33))
end subroutine
end module
