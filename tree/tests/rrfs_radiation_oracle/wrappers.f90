! Test ABI around the complete pristine fork cloudiness and gt_aod modules.
module rrfs_oracle
 use iso_c_binding
 use mpas_atmphys_vars
 use oracle_state
 use mpas_pool_routines
 use mpas_atmphys_driver_cloudiness
 use module_mp_thompson_aerosols
 use module_ra_rrtmg_sw_aerosols
 implicit none
contains
 subroutine cloud(ncol,nlev,mode,step,gfl,fqc,fqi,len,dens,ln,input,extra,out,dx) bind(C)
 integer(c_int),value::ncol,nlev,mode,step,gfl,fqc,fqi
 real(c_float),value::len
 real(c_float)::dens(ncol),ln(ncol),input(nlev,ncol,8),extra(nlev,ncol,6),out(nlev,ncol,9),dx(ncol)
 type(mpas_pool_type)::pool
 ! Deallocate against the previous config before changing selection.
 call deallocate_cloudiness(pool)
 if(allocated(qv_p)) deallocate(qv_p,qc_p,qi_p,qs_p,t_p,pres_p,pres_hyd_p,rho_p,density,land,area,cld, &
     blfrac,blc,bli,cufrac,cuc,cui)
 nc=ncol;nz=nlev;ime=nc;ite=nc;ide=nc;kme=nz;kte=nz;kde=nz
 f_qc=fqc/=0;f_qi=fqi/=0;len_disp=len
 select case(mode)
 case(0);scheme='cld_incidence'
 case(1);scheme='cld_fraction'
 case(2);scheme='cld_fraction_thompson'
 case(3);scheme='cld_fraction_mynn'
 case(4);scheme='cld_tempo'
 end select
 convection='off'
 if(gfl/=0)convection='cu_grell_freitas_li'
 allocate(qv_p(nc,nz,1),qc_p(nc,nz,1),qi_p(nc,nz,1),qs_p(nc,nz,1))
 allocate(t_p(nc,nz,1),pres_p(nc,nz,1),pres_hyd_p(nc,nz,1),rho_p(nc,nz,1))
 allocate(density(nc),land(nc),area(nc),cld(nz,nc))
 allocate(blfrac(nz,nc),blc(nz,nc),bli(nz,nc),cufrac(nz,nc),cuc(nz,nc),cui(nz,nc))
 density=dens;land=ln;area=1.
 qv_p(:,:,1)=transpose(input(:,:,1));qc_p(:,:,1)=transpose(input(:,:,2))
 qi_p(:,:,1)=transpose(input(:,:,3));qs_p(:,:,1)=transpose(input(:,:,4))
 t_p(:,:,1)=transpose(input(:,:,5));pres_p(:,:,1)=transpose(input(:,:,6))
 pres_hyd_p(:,:,1)=transpose(input(:,:,7));rho_p(:,:,1)=transpose(input(:,:,8))
 blfrac=extra(:,:,1);blc=extra(:,:,2);bli=extra(:,:,3)
 cufrac=extra(:,:,4);cuc=extra(:,:,5);cui=extra(:,:,6)
 call allocate_cloudiness(pool)
 call allocate_cloudiness(pool)
 call driver_cloudiness(step,pool,pool,pool,pool,1,nc)
 out(:,:,1)=cld
 out(:,:,2)=transpose(qvrad_p(:,:,1));out(:,:,3)=transpose(qcrad_p(:,:,1))
 out(:,:,4)=transpose(qirad_p(:,:,1));out(:,:,5)=transpose(qsrad_p(:,:,1))
 if(mode==3)then
 out(:,:,6)=transpose(qcbl_p(:,:,1));out(:,:,7)=transpose(qibl_p(:,:,1))
 endif
 if(gfl/=0)then
 out(:,:,8)=transpose(qccu_p(:,:,1));out(:,:,9)=transpose(qicu_p(:,:,1))
 endif
 dx=dx_p(:,1)
 call deallocate_cloudiness(pool)
 call deallocate_cloudiness(pool)
 end subroutine

 subroutine aerosol(ncol,nlev,input,out) bind(C)
 integer(c_int),value::ncol,nlev
 real(c_float)::input(nlev,ncol,6),out(nlev,ncol)
 real::p(ncol,nlev,1),t(ncol,nlev,1),v(ncol,nlev,1),dz(ncol,nlev,1), &
     nwfa(ncol,nlev,1),nifa(ncol,nlev,1),aod(ncol,nlev,1)
 p(:,:,1)=transpose(input(:,:,1));t(:,:,1)=transpose(input(:,:,2))
 v(:,:,1)=transpose(input(:,:,3));dz(:,:,1)=transpose(input(:,:,4))
 nwfa(:,:,1)=transpose(input(:,:,5));nifa(:,:,1)=transpose(input(:,:,6));aod=-999.
 call gt_aod(p,dz,t,v,nwfa,nifa,aod,1,ncol,1,1,1,nlev,1,ncol,1,1,1,nlev)
 out=transpose(aod(:,:,1))
 end subroutine
 subroutine spectral(ncol,nlev,input,types,out) bind(C)
 integer(c_int),value::ncol,nlev
 integer(c_int)::types(ncol)
 real(c_float)::input(nlev,ncol,6),out(nlev,ncol,14,3)
 real::p(ncol,nlev,1),t(ncol,nlev,1),v(ncol,nlev,1),dz(ncol,nlev,1), &
     aod(ncol,nlev,1),ht(ncol,1),total(ncol,1), &
     tau(ncol,nlev,1,14),ssa(ncol,nlev,1,14),asy(ncol,nlev,1,14)
 integer::k,b,kind(ncol,1)
 p(:,:,1)=transpose(input(:,:,1));t(:,:,1)=transpose(input(:,:,2))
 v(:,:,1)=transpose(input(:,:,3));dz(:,:,1)=transpose(input(:,:,4))
 call aerosol(ncol,nlev,input,out(:,:,1,1))
 aod(:,:,1)=transpose(out(:,:,1,1));total=0.;ht=0.;kind(:,1)=types
 do k=1,nlev
 total(:,1)=total(:,1)+aod(:,k,1)
 enddo
 tau=0.;ssa=1.;asy=0.
 call calc_aerosol_rrtmg_sw(ht=ht,dz8w=dz,p=p,t3d=t,qv3d=v,aer_type=kind, &
     aer_aod550_opt=2,aer_angexp_opt=3,aer_ssa_opt=3,aer_asy_opt=3, &
     aer_aod550_val=.12,aer_angexp_val=1.3,aer_ssa_val=.85,aer_asy_val=.9, &
     aod5502d=total,aod5503d=aod,tauaer=tau,ssaaer=ssa,asyaer=asy, &
     ims=1,ime=ncol,jms=1,jme=1,kms=1,kme=nlev, &
     its=1,ite=ncol,jts=1,jte=1,kts=1,kte=nlev)
 do b=1,14
 out(:,:,b,1)=transpose(tau(:,:,1,b))
 out(:,:,b,2)=transpose(ssa(:,:,1,b));out(:,:,b,3)=transpose(asy(:,:,1,b))
 enddo
 end subroutine
end module
