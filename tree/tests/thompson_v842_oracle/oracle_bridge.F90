! MPAS module_mp_thompson.F:1001-1387 real driver; no numerical edits.
module oracle_bridge
use iso_c_binding
use module_mp_thompson
implicit none
contains
subroutine init_oracle() bind(C)
  call thompson_init(.true.)
end subroutine
subroutine run_oracle(nz,ncol,dt,f,s) bind(C)
  integer(c_int), value :: nz,ncol
  real(c_float), value :: dt
  real(c_float) :: f(ncol,nz,18),s(ncol,9)
  call run_oracle_flags(nz,ncol,dt,f,s,[1,1,1])
end subroutine
subroutine run_oracle_flags(nz,ncol,dt,f,s,flags) bind(C)
  integer(c_int), value :: nz,ncol
  real(c_float), value :: dt
  real(c_float) :: f(ncol,nz,18),s(ncol,9)
  integer(c_int) :: flags(3)
  call mp_gt_driver(qv=f(:,:,1:1),qc=f(:,:,2:2),qi=f(:,:,3:3),qr=f(:,:,4:4), &
      qs=f(:,:,5:5),qg=f(:,:,6:6),ni=f(:,:,7:7),nr=f(:,:,8:8),th=f(:,:,9:9), &
      pii=f(:,:,10:10),p=f(:,:,11:11),w=f(:,:,12:12),dz=f(:,:,13:13), &
      rainprod=f(:,:,14:14),evapprod=f(:,:,15:15), &
      re_cloud=f(:,:,16:16),re_ice=f(:,:,17:17),re_snow=f(:,:,18:18), &
      rainnc=s(:,1:1),rainncv=s(:,2:2),snownc=s(:,3:3),snowncv=s(:,4:4), &
      graupelnc=s(:,5:5),graupelncv=s(:,6:6),sr=s(:,7:7),ntc=s(:,8:8),muc=s(:,9:9), &
      dt_in=dt,itimestep=1,has_reqc=flags(1),has_reqi=flags(2),has_reqs=flags(3), &
      ids=1,ide=ncol+1,jds=1,jde=2,kds=1,kde=nz, &
      ims=1,ime=ncol,jms=1,jme=1,kms=1,kme=nz, &
      its=1,ite=ncol,jts=1,jte=1,kts=1,kte=nz)
end subroutine
subroutine oracle_radar(nz,ncol,f,out) bind(C)
  integer(c_int), value :: nz,ncol
  real(c_float) :: f(ncol,nz,18),out(ncol,nz)
  real :: t(nz),dbz(nz)
  integer :: c
  do c=1,ncol
    t=f(c,:,9)*f(c,:,10)
    call calc_refl10cm(f(c,:,1),f(c,:,2),f(c,:,4),f(c,:,8), &
          f(c,:,5),f(c,:,6),t,f(c,:,11),dbz,1,nz,c,1)
    out(c,:)=max(-35.0,dbz)
  enddo
end subroutine
! MPAS mpas_atmphys_driver_microphysics.F:754-780 diagnostic driver slice.
subroutine oracle_radar_diagnostics(nz,ncol,f,z,out) bind(C)
  integer(c_int), value :: nz,ncol
  real(c_float) :: f(ncol,nz,18),z(ncol,nz),out(ncol,nz+3)
  real :: t1d(nz),dbz1d(nz),zp(nz),w1,w2
  integer :: i,k,kp
  do i=1,ncol
    do k=1,nz
      t1d(k)=f(i,k,9)*f(i,k,10)
      dbz1d(k)=-35.
      zp(k)=z(i,k)-z(i,1)+0.5*f(i,k,13)
    enddo
    call calc_refl10cm(f(i,:,1),f(i,:,2),f(i,:,4),f(i,:,8), &
      f(i,:,5),f(i,:,6),t1d,f(i,:,11),dbz1d,1,nz,i,1)
    kp=1
    do k=1,nz
      dbz1d(k)=max(-35.,dbz1d(k))
      out(i,k)=dbz1d(k)
      if(zp(k).lt.1000.)kp=k
    enddo
    out(i,nz+1)=maxval(dbz1d(:))
    w1=(zp(kp+1)-1000.)/(zp(kp+1)-zp(kp))
    w2=1.0-w1
    out(i,nz+2)=w1*dbz1d(kp)+w2*dbz1d(kp+1)
    out(i,nz+3)=max(out(i,nz+3),out(i,nz+2))
  enddo
end subroutine
! MPAS module_mp_thompson.F:4176-4212,4391-4466,4662-4795.
! These calls include active helpers whose classic driver input is constant.
subroutine oracle_helpers(mode,n,a,out) bind(C)
  integer(c_int),value :: mode,n
  real(c_float) :: a(n,6),out(n)
  integer :: k
  do k=1,n
    select case(mode)
    case(1)
      out(k)=delta_p(a(k,1),a(k,2),a(k,3),a(k,4),a(k,5))
    case(2)
      out(k)=iceDeMott(a(k,1),a(k,2),a(k,3),a(k,4),a(k,5),a(k,6))
    case(3)
      out(k)=iceKoop(a(k,1),a(k,2),a(k,3),a(k,4),a(k,5))
    case(4)
      out(k)=Eff_aero(a(k,1),a(k,2),a(k,3),a(k,4),a(k,5),achar(int(a(k,6))))
    case(5)
      out(k)=activ_ncloud(a(k,1),a(k,2),a(k,3))
    end select
  enddo
end subroutine
! MPAS module_mp_thompson.F:4809-4913; direct helper-boundary radius cases.
subroutine oracle_effectrad(nz,ncol,a,prescribed,out) bind(C)
  integer(c_int),value :: nz,ncol
  real(c_float) :: a(ncol,nz,8),prescribed(ncol),out(ncol,nz,3)
  real(c_float) :: clear(1,2,18),surface(1,9)
  integer :: c
  do c=1,ncol
    ! Nt_c is private: initialize it through the public, unchanged driver.
    clear=0.;surface=0.
    clear(:,:,1)=0.001;clear(:,:,9)=250.;clear(:,:,10)=1.
    clear(:,:,11)=80000.;clear(:,:,13)=100.
    surface(:,8)=prescribed(c);surface(:,9)=2.
    call run_oracle(2,1,1.0_c_float,clear,surface)
    call calc_effectrad(a(c,:,1),a(c,:,2),a(c,:,3),a(c,:,4),a(c,:,5), &
        a(c,:,6),a(c,:,7),a(c,:,8),out(c,:,1),out(c,:,2),out(c,:,3),1,nz)
  enddo
end subroutine
subroutine export_aux() bind(C)
  open(72,file='auxiliary.bin',access='stream',form='unformatted',status='replace',convert='little_endian')
  write(72) t_Efrw,t_Efsw,tnr_rev,tpc_wev,tnc_wev,tnccn_act
  close(72)
end subroutine
end module
