! MPAS-A v8.4.2 module_cu_kfeta.F:2855-2982.
subroutine kf_native_tables(params) bind(c)
use iso_c_binding
use module_cu_kfeta
implicit none
real(c_float),intent(in) :: params(16)
call kf_lutab(params(13),params(14),params(15),params(16))
end subroutine
! MPAS-A v8.4.2 module_cu_kfeta.F:19-361.
subroutine kf_native_column(nz,fields,state,surface,active,params,flags) bind(c)
use iso_c_binding
use module_cu_kfeta
implicit none
integer(c_int),value :: nz
real(c_float),intent(in) :: fields(1,nz+1,1,10),params(16)
real(c_float),intent(inout) :: state(1,nz+1,1,7),surface(1,1,7)
integer(c_int),intent(in) :: flags(7)
integer(c_int),intent(inout) :: active
logical :: cu_act_flag(1,1),adapt_step_flag,warm_rain,f_qr,f_qi,f_qs
cu_act_flag=active/=0
adapt_step_flag=flags(3)/=0
! Preserve the native driver argument word, including its unassigned logical.
warm_rain=transfer(flags(4),warm_rain)
f_qr=flags(5)/=0
f_qi=flags(6)/=0
f_qs=flags(7)/=0
call kf_eta_cps( &
  ids=1, &
  ide=1, &
  jds=1, &
  jde=1, &
  kds=1, &
  kde=nz+1, &
  ims=1, &
  ime=1, &
  jms=1, &
  jme=1, &
  kms=1, &
  kme=nz+1, &
  its=1, &
  ite=1, &
  jts=1, &
  jte=1, &
  kts=1, &
  kte=nz, &
  dt=params(1), &
  ktau=flags(2), &
  dxcell=surface(:,:,1), &
  areacell=surface(:,:,2), &
  cudt=params(2), &
  curr_secs=params(3), &
  adapt_step_flag=adapt_step_flag, &
  rho=fields(:,:,:,9), &
  raincv=surface(:,:,3), &
  pratec=surface(:,:,4), &
  nca=surface(:,:,5), &
  u=fields(:,:,:,1), &
  v=fields(:,:,:,2), &
  th=fields(:,:,:,3), &
  t=fields(:,:,:,4), &
  w=fields(:,:,:,5), &
  dz8w=fields(:,:,:,6), &
  pcps=fields(:,:,:,7), &
  pi=fields(:,:,:,8), &
  w0avg=state(:,:,:,1), &
  xlv0=params(4), &
  xlv1=params(5), &
  xls0=params(6), &
  xls1=params(7), &
  cp=params(8), &
  r=params(9), &
  g=params(10), &
  ep1=params(11), &
  ep2=params(12), &
  svp1=params(13), &
  svp2=params(14), &
  svp3=params(15), &
  svpt0=params(16), &
  stepcu=flags(1), &
  cu_act_flag=cu_act_flag, &
  warm_rain=warm_rain, &
  cutop=surface(:,:,6), &
  cubot=surface(:,:,7), &
  qv=fields(:,:,:,10), &
  f_qr=f_qr, &
  f_qi=f_qi, &
  f_qs=f_qs, &
  rthcuten=state(:,:,:,2), &
  rqvcuten=state(:,:,:,3), &
  rqccuten=state(:,:,:,4), &
  rqrcuten=state(:,:,:,5), &
  rqicuten=state(:,:,:,6), &
  rqscuten=state(:,:,:,7))
active=0
if(cu_act_flag(1,1)) active=1
end subroutine
! MPAS-A v8.4.2 module_cu_kfeta.F:2765-2851.
subroutine kf_native_init(nz,state,surface,restart,qi,qs,params) bind(c)
use iso_c_binding
use module_cu_kfeta
implicit none
integer(c_int),value :: nz,restart,qi,qs
real(c_float),intent(inout) :: state(1,nz+1,1,7),surface(1,1,7)
real(c_float),intent(in) :: params(16)
call kf_eta_init(state(:,:,:,2),state(:,:,:,3),state(:,:,:,4),state(:,:,:,5), &
 state(:,:,:,6),state(:,:,:,7),surface(:,:,5),state(:,:,:,1),qi,qs, &
 params(13),params(14),params(15),params(16),1,restart/=0,.true., &
 1,2,1,2,1,nz+1,1,1,1,1,1,nz+1,1,1,1,1,1,nz)
end subroutine
! MPAS-A v8.4.2 mpas_atmphys_driver_convection.F:1030-1054.
subroutine kf_native_step1(nz,dt,state,surface) bind(c)
use iso_c_binding
implicit none
integer(c_int),value :: nz
real(c_float),value :: dt
real(c_float),intent(inout) :: state(nz+1,7),surface(7)
integer :: k,f
if(surface(5)>0.)then
  surface(5)=surface(5)-dt
  if(surface(5)<0.5*dt)then
    do k=1,nz
      do f=2,7
        state(k,f)=0.
      enddo
    enddo
    surface(3)=0.
    surface(4)=0.
    surface(7)=nz+1
    surface(6)=1
  endif
endif
end subroutine
! MPAS-A v8.4.2 mpas_atmphys_driver_convection.F:1088-1099.
subroutine kf_native_step2(nc,dt,bucket,alarm,cuprec,rainc,counter) bind(c)
use iso_c_binding
implicit none
integer(c_int),value :: nc,alarm
real(c_float),value :: dt,bucket
real(c_float),intent(in) :: cuprec(nc)
real(c_float),intent(inout) :: rainc(nc)
integer(c_int),intent(inout) :: counter(nc)
integer :: i
do i=1,nc
  rainc(i)=rainc(i)+cuprec(i)*dt
  if(alarm/=0 .and. bucket>0. .and. rainc(i)>bucket)then
    counter(i)=counter(i)+1
    rainc(i)=rainc(i)-bucket
  endif
enddo
end subroutine
! MPAS-A v8.4.2 mpas_atmphys_driver_convection.F:766.
subroutine kf_native_dx(nc,len_disp,density,dx) bind(c)
use iso_c_binding
implicit none
integer(c_int),value :: nc
real(c_float),value :: len_disp
real(c_float),intent(in) :: density(nc)
real(c_float),intent(out) :: dx(nc)
integer :: i
do i=1,nc
  dx(i)=len_disp/density(i)**0.25
enddo
end subroutine
