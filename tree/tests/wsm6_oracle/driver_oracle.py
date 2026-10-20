"""Extract MPAS's own WSM6 driver loops verbatim into a standalone C bridge."""
from pathlib import Path
from hexcore.mpasphys.wsm6_v842 import FIELD_NAMES


def generate(reference: Path) -> str:
    src=reference/'src/core_atmosphere/physics'
    interface=(src/'mpas_atmphys_interface.F').read_text().splitlines()
    driver=(src/'mpas_atmphys_driver_microphysics.F').read_text().splitlines()
    def original(lines,filename,start,end):
        return f'! MPAS-A v8.4.2 {filename}:{start}-{end}\n'+'\n'.join(lines[start-1:end])+'\n'
    rename={'pressure_base':'pressure_b','rtheta_base':'rtheta_b','exner_base':'exner_b'}
    surfaces=set(FIELD_NAMES[30:40])|set(FIELD_NAMES[41:])
    s='''subroutine wsm6_driver_oracle(nz,nc,ns,top,dt,dt_dyn,hail,c,fields,i_rainnc,radii_on,radar_on,bucket,acrain,rvord,rcv,P0) bind(C)
 use iso_c_binding
 use mpas_kind_types,only:RKIND
 use mp_wsm6
 use module_mp_wsm6
 implicit none
 integer(c_int),value::nz,nc,ns,top,hail,radii_on,radar_on,acrain
 real(c_float),value::dt,dt_dyn,bucket,rvord,rcv,P0
 real(c_float)::c(26)
 type(c_ptr)::fields(*)
 integer(c_int)::i_rainnc(nc)
 integer::i,j,k,kp,err,its,ite,kts,kte,jts,jte,mp_top_level
 real(RKIND)::R_d,R_v,gravity,config_bucket_rainnc,re_qc_bg,re_qi_bg,re_qs_bg,rho_a,rho1,rho2,tem1,tem2,w1,w2
 logical::l_acrain
 character(len=256)::msg
 real(RKIND),pointer::alias(:,:)
 real(RKIND)::w(nz,nc),qv1d(nz),qr1d(nz),qs1d(nz),qg1d(nz),t1d(nz),p1d(nz),dbz1d(nz),zp(nz)
'''
    for name in FIELD_NAMES:
        local=rename.get(name,name)
        shape='nz+1,nc' if name=='zgrid' else 'nz,nc'
        s+=f' real(RKIND),pointer::{local}(:)\n' if name in surfaces else f' real(RKIND)::{local}({shape})\n'
    work='th qv qc qi qr qs qg rho pi pres z dz w rainprod evapprod recloud reice resnow'.split()
    for name in work:s+=f' real(RKIND)::{name}_p(nc,nz,1)\n'
    for name in 'rainnc rainncv snownc snowncv graupelnc graupelncv sr'.split():s+=f' real(RKIND)::{name}_p(nc,1)\n'
    s+='its=1;ite=ns;kts=1;kte=nz;jts=1;jte=1;mp_top_level=top\n'
    s+='R_d=c(4);R_v=c(5);gravity=c(1);config_bucket_rainnc=bucket;l_acrain=acrain/=0\n'
    s+='re_qc_bg=c(21);re_qi_bg=c(22);re_qs_bg=c(23)\n'
    for index,name in enumerate(FIELD_NAMES,1):
        local=rename.get(name,name)
        if name in surfaces:s+=f'call c_f_pointer(fields({index}),{local},[nc])\n'
        else:
            s+=f'call c_f_pointer(fields({index}),alias,[nc,{"nz+1" if name=="zgrid" else "nz"}])\n{local}=transpose(alias)\n'
    s+='w=0.;w_p=0.;rainnc_p=0.;rainncv_p=0.;snownc_p=0.;snowncv_p=0.;graupelnc_p=0.;graupelncv_p=0.;sr_p=0.\n'
    s+=original(interface,'mpas_atmphys_interface.F',657,675)
    s+=original(interface,'mpas_atmphys_interface.F',693,707)
    s+=original(interface,'mpas_atmphys_interface.F',776,786)
    s+='call mp_wsm6_init(c(16),c(17),100.,c(18),c(3),hail,msg,err)\n'
    s+='''call wsm6(th_p,qv_p,qc_p,qr_p,qi_p,qs_p,qg_p,rho_p,pi_p,pres_p,dz_p, &
 dt,c(1),c(2),c(3),c(4),c(5),c(6),c(7),c(8),c(12),c(13),c(14),c(15),c(16),c(17),c(18),c(19),c(20), &
 rain=rainnc_p,rainncv=rainncv_p,snow=snownc_p,snowncv=snowncv_p,graupel=graupelnc_p,graupelncv=graupelncv_p,sr=sr_p, &
 has_reqc=radii_on,has_reqi=radii_on,has_reqs=radii_on, &
 re_qc_bg=c(21),re_qi_bg=c(22),re_qs_bg=c(23),re_qc_max=c(24),re_qi_max=c(25),re_qs_max=c(26), &
 re_cloud=recloud_p,re_ice=reice_p,re_snow=resnow_p, &
 ids=1,ide=nc,jds=1,jde=1,kds=1,kde=nz,ims=1,ime=nc,jms=1,jme=1,kms=1,kme=nz, &
 its=1,ite=ns,jts=1,jte=1,kts=1,kte=nz,errmsg=msg,errflg=err)
if(radar_on/=0) then
'''
    s+=original(driver,'mpas_atmphys_driver_microphysics.F',700,727)+'endif\n'
    for start,end in ((605,632),(637,648)):s+=original(driver,'mpas_atmphys_driver_microphysics.F',start,end)
    for start,end in ((903,946),(949,964),(983,1013),(1076,1101)):s+=original(interface,'mpas_atmphys_interface.F',start,end)
    for index,name in enumerate(FIELD_NAMES,1):
        if name not in surfaces:
            s+=f'call c_f_pointer(fields({index}),alias,[nc,{"nz+1" if name=="zgrid" else "nz"}])\nalias=transpose({rename.get(name,name)})\n'
    s+='end subroutine\n'
    return s
