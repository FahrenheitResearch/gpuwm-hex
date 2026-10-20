module oracle_wrappers
 use iso_c_binding
 use mpas_atmphys_vars
 use oracle_state
 use mpas_pool_routines
 use mpas_atmphys_driver_cloudiness
 use mpas_atmphys_o3climatology
 use module_ra_rrtmg_vinterp
 implicit none
contains
 subroutine cloud(ncol,nlev,mode,fqc,fqi,len,dens,ln,v,c,ice,snow,t,p,ph,r,out,dx) bind(C)
 integer(c_int),value::ncol,nlev,mode,fqc,fqi
 real(c_float),value::len
 real(c_float)::dens(ncol),ln(ncol),v(nlev,ncol),c(nlev,ncol),ice(nlev,ncol),snow(nlev,ncol)
 real(c_float)::t(nlev,ncol),p(nlev,ncol),ph(nlev,ncol),r(nlev,ncol),out(nlev,ncol,5),dx(ncol)
 type(mpas_pool_type)::pool
 nc=ncol;nz=nlev
 ime=nc;ite=nc;ide=nc;kme=nz;kte=nz;kde=nz
 f_qc=fqc/=0;f_qi=fqi/=0;len_disp=len
 if(allocated(qv_p)) deallocate(qv_p,qc_p,qi_p,qs_p,t_p,pres_p,pres_hyd_p,rho_p,density,land,area,cld)
 call deallocate_cloudiness
 allocate(qv_p(nc,nz,1),qc_p(nc,nz,1),qi_p(nc,nz,1),qs_p(nc,nz,1))
 allocate(t_p(nc,nz,1),pres_p(nc,nz,1),pres_hyd_p(nc,nz,1),rho_p(nc,nz,1))
 allocate(density(nc),land(nc),area(nc),cld(nz,nc))
 density=dens;land=ln;area=1.
 qv_p(:,:,1)=transpose(v);qc_p(:,:,1)=transpose(c);qi_p(:,:,1)=transpose(ice);qs_p(:,:,1)=transpose(snow)
 t_p(:,:,1)=transpose(t);pres_p(:,:,1)=transpose(p);pres_hyd_p(:,:,1)=transpose(ph);rho_p(:,:,1)=transpose(r)
 select case(mode)
 case(0);scheme='cld_incidence'
 case(1);scheme='cld_fraction'
 case(2);scheme='cld_fraction_thompson'
 end select
 call allocate_cloudiness
 call allocate_cloudiness ! cover the already-allocated native lifecycle
 call driver_cloudiness(pool,pool,pool,pool,1,nc)
 out(:,:,1)=cld
 out(:,:,2)=transpose(qvrad_p(:,:,1));out(:,:,3)=transpose(qcrad_p(:,:,1))
 out(:,:,4)=transpose(qirad_p(:,:,1));out(:,:,5)=transpose(qsrad_p(:,:,1))
 dx=dx_p(:,1)
 end subroutine

 subroutine ozone_init(ncol,nlev,y,press,out) bind(C)
 integer(c_int),value::ncol,nlev
 real(c_float)::y(ncol),press(nlev),out(12,nlev,ncol)
 type(mpas_pool_type)::pool
 nc=ncol;nl=nlev
 if(allocated(lat)) deallocate(lat,lon,pin,monthly,ozn)
 allocate(lat(nc),lon(nc),pin(nl),monthly(12,nl,nc),ozn(nl,nc))
 lat=y;lon=0.
 call init_o3climatology(pool,pool)
 press=pin;out=monthly
 end subroutine
 subroutine ozone_time(ncol,nlev,jul,input,out) bind(C)
 integer(c_int),value::ncol,nlev
 real(c_float),value::jul
 real(c_float)::input(12,nlev,ncol),out(nlev,ncol)
 type(mpas_pool_type)::pool
 nc=ncol;nl=nlev;monthly=input
 call o3climatology_from_MPAS(jul,pool,pool,pool)
 out=ozn
 end subroutine
 subroutine ozone_update(ncol,nlev,year,month,day,input,out) bind(C)
 integer(c_int),value::ncol,nlev,year,month,day
 real(c_float)::input(12,nlev,ncol),out(nlev,ncol)
 type(mpas_pool_type)::pool
 character(512)::date
 write(date,'(I4.4,A,I2.2,A,I2.2,A)') year,'-',month,'-',day,'_12:00:00.0000'
 nc=ncol;nl=nlev;monthly=input
 call update_o3climatology(date,pool,pool,pool)
 out=ozn
 end subroutine
 subroutine ozone_vertical(ncol,nlev,nlevels,p,levels,oz,out) bind(C)
 integer(c_int),value::ncol,nlev,nlevels
 real(c_float)::p(nlev,ncol),levels(nlevels),oz(nlevels,ncol),out(nlev,ncol)
 real::p2(ncol,nlev),oz2(ncol,nlevels),o2(ncol,nlev)
 p2=transpose(p);oz2=transpose(oz)
 call vinterp_ozn(1,ncol,ncol,nlev,p2,levels,nlevels,oz2,o2)
 out=transpose(o2)
 end subroutine
 ! MPAS mpas_atmphys_driver_radiation_lw.F:489-500
 ! Keep native Pa-to-hPa arithmetic independent of CUDA preparation.
 subroutine ozone_vertical_pa(ncol,nlev,nlevels,p,levels,oz,out) bind(C)
 integer(c_int),value::ncol,nlev,nlevels
 real(c_float)::p(nlev,ncol),levels(nlevels),oz(nlevels,ncol),out(nlev,ncol)
 real(c_float)::native_hpa(nlev,ncol)
 integer::i,k
 do i=1,ncol
    do k=1,nlev
       native_hpa(k,i)=p(k,i)/100._c_float
    enddo
 enddo
 call ozone_vertical(ncol,nlev,nlevels,native_hpa,levels,oz,out)
 end subroutine
end module
