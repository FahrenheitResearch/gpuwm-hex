! RRFSx/MPAS-Model 1d6e370e7b43a332074809f170532a216c2212fb
! src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:498-503,728-776.
! Complete pristine module, exposed through the actual MPAS moment selections.
module rrfs_nssl2m_oracle
  use iso_c_binding, only: c_int, c_float, c_double
  use module_mp_nssl_2mom, only: nssl_2mom_init, nssl_2mom_driver
  implicit none
  integer :: oracle_moments = 2
contains
  subroutine initialize(moments) bind(C, name='rrfs_nssl2m_initialize')
    integer(c_int), value :: moments
    oracle_moments = moments
    if (moments == 3) then
      call nssl_2mom_init(ipctmp=8, mixphase=0, ihvol=1, myrank=0, mpiroot=0)
    else
      call nssl_2mom_init(ipctmp=5, mixphase=0, ihvol=1, myrank=0, mpiroot=0)
    endif
  end subroutine initialize

  subroutine step(nc,nz,nt,nmicro,diagnose,dbz_due,reqc,reqi,reqs,dt, &
                  state,environment,surface,diagnostics,refl1km,rainprod,evapprod) &
                  bind(C, name='rrfs_nssl2m_step')
    integer(c_int), value :: nc,nz,nt,nmicro,diagnose,dbz_due,reqc,reqi,reqs
    real(c_float), value :: dt
    ! C contiguous buffers: (field, level, column). MPAS column index is fastest.
    real(c_float), intent(inout), target :: state(nc,nz,1,*)
    real(c_float), intent(inout) :: surface(nc,1,7)
    real(c_float), pointer :: zrw(:,:,:), zhw(:,:,:), zhl(:,:,:)
    real(c_float), intent(in) :: environment(nc,nz,1,5)
    real(c_float), intent(inout) :: diagnostics(nc,nz,1,4), refl1km(nc,1)
    real(c_float), intent(inout) :: rainprod(nc,nz,1), evapprod(nc,nz,1)
    nullify(zrw,zhw,zhl)
    if (oracle_moments == 3) then
      zrw => state(:,:,:,18)
      zhw => state(:,:,:,19)
      zhl => state(:,:,:,20)
    endif
    call nssl_2mom_driver( &
      th=state(:,:,:,1), qv=state(:,:,:,2), qc=state(:,:,:,3), &
      qr=state(:,:,:,4), qi=state(:,:,:,5), qs=state(:,:,:,6), &
      qh=state(:,:,:,7), qhl=state(:,:,:,8), cn=state(:,:,:,15), &
      ccw=state(:,:,:,9), crw=state(:,:,:,10), cci=state(:,:,:,11), &
      csw=state(:,:,:,12), chw=state(:,:,:,13), chl=state(:,:,:,14), &
      vhw=state(:,:,:,16), vhl=state(:,:,:,17), dn=environment(:,:,:,1), &
      zrw=zrw, zhw=zhw, zhl=zhl, &
      pii=environment(:,:,:,3), p=environment(:,:,:,2), dz=environment(:,:,:,4), &
      w=environment(:,:,:,5), dtp=dt, itimestep=nt, &
      rainnc=surface(:,:,1), rainncv=surface(:,:,2), &
      snownc=surface(:,:,3), snowncv=surface(:,:,4), &
      grplnc=surface(:,:,5), grplncv=surface(:,:,6), sr=surface(:,:,7), &
      rainprod=rainprod, evapprod=evapprod, re_cloud=diagnostics(:,:,:,2), &
      re_ice=diagnostics(:,:,:,3), re_snow=diagnostics(:,:,:,4), &
      has_reqc=reqc, has_reqi=reqi, has_reqs=reqs, &
      diagflag=diagnose/=0, dbz=diagnostics(:,:,:,1), diag_dbz=dbz_due/=0, &
      ntmul=nmicro, ntcnt=1, lastloop=.true., refl_diagnostic=refl1km, &
      ids=1, ide=nc, jds=1, jde=1, kds=1, kde=nz, &
      ims=1, ime=nc, jms=1, jme=1, kms=1, kme=nz, &
      its=1, ite=nc, jts=1, jte=1, kts=1, kte=nz)
  end subroutine step

  ! Compiler diagnostic for the integer powers in the pristine module.
  ! Every exponent is literal, matching the scheme's GNU lowering.
  subroutine powers(n,x,y,a,b) bind(C, name='rrfs_nssl2m_powers')
    integer(c_int), value :: n
    real(c_float), intent(in) :: x(n)
    real(c_double), intent(in) :: y(n)
    real(c_float), intent(out) :: a(n,8)
    real(c_double), intent(out) :: b(n,8)
    integer :: i
    do i=1,n
      a(i,1)=x(i)**2; b(i,1)=y(i)**2
      a(i,2)=x(i)**3; b(i,2)=y(i)**3
      a(i,3)=x(i)**4; b(i,3)=y(i)**4
      a(i,4)=x(i)**5; b(i,4)=y(i)**5
      a(i,5)=x(i)**6; b(i,5)=y(i)**6
      a(i,6)=x(i)**7; b(i,6)=y(i)**7
      a(i,7)=x(i)**8; b(i,7)=y(i)**8
      a(i,8)=x(i)**9; b(i,8)=y(i)**9
    enddo
  end subroutine powers
end module rrfs_nssl2m_oracle

! The only external framework routine reached by the complete module is fatal
! handling for invalid moment selection. It is inactive for the MPAS 2m call.
subroutine wrf_error_fatal(message)
  implicit none
  character(len=*), intent(in) :: message
  write(0,'(A)') message
  error stop 1
end subroutine wrf_error_fatal
