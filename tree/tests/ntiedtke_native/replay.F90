program run_cu_ntiedtke
#ifndef NT_COLUMN_REPEATS
#define NT_COLUMN_REPEATS 1
#endif
  ! Pristine MPAS-A v8.4.2 module_cu_ntiedtke.F:20-289 full driver replay.
  ! Every driver input/output is recorded as its raw float32 word.
  ! The isolation arm measures the native slab flag's consequences.
  use iso_fortran_env, only: int32
  use ccpp_kind_types, only: kind_phys
  use module_cu_ntiedtke, only: cu_ntiedtke_driver
  use nt_cases
  implicit none

  integer, parameter :: nz  = nt_nz
  integer, parameter :: nc  = nt_ncase * NT_COLUMN_REPEATS
  integer, parameter :: ndx = nt_ndx

  ! One j row, nc columns in i, nz half levels.
  integer, parameter :: ids = 1, ide = nc + 1, jds = 1, jde = 2
  integer, parameter :: kds = 1, kde = nz + 1
  integer, parameter :: ims = 1, ime = nc,     jms = 1, jme = 1
  integer, parameter :: kms = 1, kme = nz + 1
  integer, parameter :: its = 1, ite = nc,     jts = 1, jte = 1
  integer, parameter :: kts = 1, kte = nz

  real(kind=kind_phys), dimension(ims:ime, kms:kme, jms:jme) :: &
       u3d, v3d, w, t3d, qv3d, qc3d, qi3d, pi3d, rho3d, &
       dz8w, pcps, p8w, qvften, thften, &
       rthcuten, rqvcuten, rqccuten, rqicuten, rucuten, rvcuten
  real(kind=kind_phys), dimension(ims:ime, jms:jme) :: &
       raincv, pratec, qfx, hfx, xland, dx
  logical, dimension(ims:ime, jms:jme) :: cu_act_flag

  ! single-column mirrors, for the isolation arm
  real(kind=kind_phys), dimension(1, kms:kme, 1) :: &
       s_u3d, s_v3d, s_w, s_t3d, s_qv3d, s_qc3d, s_qi3d, s_pi3d, s_rho3d, &
       s_dz8w, s_pcps, s_p8w, s_qvften, s_thften, &
       s_rthcuten, s_rqvcuten, s_rqccuten, s_rqicuten, s_rucuten, s_rvcuten
  real(kind=kind_phys), dimension(1, 1) :: &
       s_raincv, s_pratec, s_qfx, s_hfx, s_xland, s_dx
  logical, dimension(1, 1) :: s_cu_act_flag

  ! MPAS bottom-up column scratch
  real, dimension(nz)     :: b_t, b_qv, b_qc, b_qi, b_u, b_v
  real, dimension(nz)     :: b_pcps, b_dz, b_rho, b_pi, b_qvften, b_thften
  real, dimension(nz + 1) :: b_p8w, b_w
  real :: b_xland, b_hfx, b_qfx

  real(kind=kind_phys) :: dt = 60.0_kind_phys
  real(kind=kind_phys) :: dz_scale = 1.0_kind_phys
  real(kind=kind_phys) :: qv_scale = 1.0_kind_phys
  integer :: dz_first = 1, dz_last = nz
  logical :: interface_override = .false.
  real(kind=kind_phys) :: interface_value = 0.0_kind_phys
  integer :: interface_first = 2, interface_last = 2
  integer :: stepcu = 1
  integer :: itimestep = 0      ! MPAS initflag: zero reads forcing.

  ! MPAS always supplies both present condensate flags.
  logical :: p_f_qc = .true., p_f_qi = .true.

  character(len=256) :: errmsg
  integer :: errflg
  integer :: n, m, k, ndiff, flag_mode, variant, input_word
  character(len=32) :: arg
  real :: sf, sf2, dxv
#ifdef NT_MPAS_BINDING
  real :: raw_forcing(6)
#endif

  ! Refuse any build that changes MPAS RKIND from the pinned single recipe.
  if (kind(1.0_kind_phys) /= 4) then
     write(*, '(a,i0,a)') 'FATAL: kind_phys is ', kind(1.0_kind_phys), &
          ' bytes, not 4.  Rebuild with -DSINGLE_PRECISION.'
     stop 3
  end if

  flag_mode = 0
  if (command_argument_count() >= 1) then
     call get_command_argument(1,arg)
     read(arg,*) itimestep
  endif
  if (command_argument_count() >= 2) then
     call get_command_argument(2,arg)
     read(arg,*) stepcu
  endif
  if (command_argument_count() >= 3) then
     call get_command_argument(3,arg)
     read(arg,*) flag_mode
  endif
  p_f_qc = flag_mode /= 1 .and. flag_mode /= 3
  p_f_qi = flag_mode /= 2 .and. flag_mode /= 3
  call nt_build_case_table()
  variant=0
  if(command_argument_count()>=4)then
    call get_command_argument(4,arg)
    read(arg,*)variant
  endif
  ! Additional full-driver stress inputs for branches omitted by the
  ! original analytic fixture. Scheme outputs remain pristine Fortran.
  select case(variant)
  case(1)
    c_wamp=c_wamp*80.; c_vbase=8.; c_ushear=c_ushear*100.
  case(2)
    c_tsfc=258.; c_lapse=0.008; c_hfx=300.; c_qfx=3.e-4
    c_rhsfc=0.98; c_rhmid=0.95; c_rhtop=0.8; c_ztrop=9000.
  case(3)
    c_qcamp=0.01; c_qiamp=0.01; c_rhmid=0.1; c_rhtop=0.01
  case(4)
    c_thften=0.001; c_qvften=-1.e-5; c_wamp=-0.3
  case(5)
    c_tsfc=310.; c_lapse=0.009; c_rhmid=0.99; c_rhtop=0.95
    c_hfx=800.; c_qfx=0.001; c_wamp=2.; c_psfc=75000.
  case(6)
    c_rhsfc=0.3; c_rhmid=0.99; c_rhtop=0.9; c_wamp=5.
    c_zinv=800.; c_dtinv=10.; c_hfx=0.; c_qfx=0.
  case(7)
    c_ushear=c_ushear*1000.; c_vbase=30.; c_wamp=0.6
    c_hfx=500.; c_qfx=8.e-4
  case(8)
    do n=1,nt_ncase
      c_zinv(n)=500.+350.*n; c_dtinv(n)=20.
      c_rhmid(n)=0.2; c_rhtop(n)=0.05; c_rhsfc(n)=0.99
      c_wamp(n)=0.03; c_tsfc(n)=300.; c_lapse(n)=0.008
    enddo
  case(9)
    c_rhmid=0.8; c_rhsfc=0.95; c_rhtop=0.1
    c_qvften=-5.e-5; c_thften=-0.001; c_wamp=0.5
  case(10)
    do n=1,nt_ncase
      c_tsfc(n)=280.+n; c_lapse(n)=0.004+0.00035*n
      c_rhsfc(n)=0.4+0.03*n; c_rhmid(n)=0.2+0.04*n
      c_wamp(n)=0.04*n; c_hfx(n)=20.*n; c_qfx(n)=2.e-5*n
    enddo
  case(11)
    c_rhsfc=0.99; c_rhmid=0.99; c_rhtop=0.99
    c_hfx=0.; c_qfx=0.; c_thften=0.; c_qvften=0.
    do n=1,nt_ncase
      c_wamp(n)=(-1.)**n*0.2*n
    enddo
  case(12)
    c_tsfc=300.; c_lapse=0.01; c_rhsfc=0.99; c_rhmid=0.99
    c_qvften=1.e-6; c_thften=1.e-5; c_wamp=0.03
  case(13:160)
    ! Deterministic varied thermodynamic profiles; no expected output
    ! is prescribed here. The pristine module produces every oracle word.
    ! cuddrafn:2616-2650 initializes itopde only when the bottom
    ! interface gap is <6000 Pa. Coarser ladders leave native itopde
    ! undefined; grid33 with exponent 0.70 produced SIGSEGV in 7/10
    ! pristine strict runs. Keep this oracle sweep in the defined domain.
    if (variant > 32) nt_sigma_power=0.9+0.05*mod(variant,25)
    do n=1,nt_ncase
      c_tsfc(n)=260.+mod(7*n+13*variant,51)
      c_lapse(n)=0.003+0.0005*mod(n+variant,17)
      c_rhsfc(n)=0.2+0.05*mod(3*n+variant,17)
      c_rhmid(n)=0.1+0.05*mod(5*n+3*variant,19)
      c_rhtop(n)=0.1+0.05*mod(7*n+variant,19)
      c_zinv(n)=250.*mod(n+variant,25)
      c_dtinv(n)=real(mod(3*n+variant,21))
      c_wamp(n)=0.1*mod(n+variant,31)-0.3
      c_hfx(n)=20.*mod(5*n+variant,31)
      c_qfx(n)=1.e-5*mod(7*n+variant,61)
      c_ushear(n)=0.001*mod(3*n+variant,21)
      c_thften(n)=1.e-5*(mod(n+variant,11)-5)
      c_qvften(n)=1.e-7*(mod(n+3*variant,11)-5)
      if(variant>96)then
        ! Supersaturation and moist forcing expose adjustment/repair
        ! paths not reached by the subsaturated analytic profiles.
        c_rhsfc(n)=0.8+0.1*mod(n+variant,11)
        c_rhmid(n)=0.8+0.1*mod(3*n+variant,11)
        c_rhtop(n)=0.8+0.1*mod(5*n+variant,11)
        c_qvften(n)=1.e-5*mod(n+variant,11)
        c_qcamp(n)=0.001*mod(n+variant,11)
        c_qiamp(n)=0.001*mod(3*n+variant,11)
      endif
    enddo
  case(161)
    nt_irregular_bottom = .true.
  case(162)
    ! Native-defect witness only: cuddrafn:2616-2650 can leave itopde
    ! unassigned on this coarse bottom ladder. Never an acceptance oracle.
    nt_sigma_power = 0.70
  case(1000:1127)
    call nt_varied_case_table(variant)
  case(1200:1263)
    call nt_boundary_case_table(variant)
  case(2000:3023)
    call nt_melting_inversion_case_table(variant)
  end select

  ! Optional exact input-word search. Only the native full driver's surface
  ! forcing input changes; no plume intermediate or expected output is set.
  ! MPAS cu_ntiedtke.F90:1349-1479,1579-1689.
  if (command_argument_count() >= 5) then
    call get_command_argument(5,arg)
    read(arg,'(z8)') input_word
    c_hfx = transfer(input_word,0.0)
  end if
  if (command_argument_count() >= 6) then
    call get_command_argument(6,arg)
    read(arg,'(z8)') input_word
    c_lapse = transfer(input_word,0.0)
  end if
  ! MPAS cu_ntiedtke.F90:468-469,1920-1925,2012-2042.
  ! Supply a finite exact timestep input to probe native mass-flux caps;
  ! the complete native module computes every intermediate and output.
  if (command_argument_count() >= 7) then
    call get_command_argument(7,arg)
    read(arg,'(z8)') input_word
    dt = transfer(input_word,0.0_kind_phys)
  end if
  ! MPAS cu_ntiedtke_pre.F:117-143 passes geometric layer thickness and
  ! integrates geopotential. Vary only this caller input; the entire native
  ! module computes parcel energetics, mass-flux caps and every output.
  if (command_argument_count() >= 8) then
    call get_command_argument(8,arg)
    read(arg,'(z8)') input_word
    dz_scale = transfer(input_word,0.0_kind_phys)
  end if
  ! MPAS module_cu_ntiedtke.F:158-273,cu_ntiedtke.F90:1585-1689.
  ! Scale only generated caller vapor inputs, never parcel intermediates.
  if (command_argument_count() >= 9) then
    call get_command_argument(9,arg)
    read(arg,'(z8)') input_word
    qv_scale = transfer(input_word,0.0_kind_phys)
  end if
  ! MPAS module_cu_ntiedtke.F:158-204, cu_ntiedtke.F90:1585-1689.
  ! Restrict a geometric input perturbation to a caller-selected layer band;
  ! pressure and all internally generated parcel states remain untouched.
  if (command_argument_count() >= 10) then
    call get_command_argument(10,arg)
    read(arg,*) dz_first
  end if
  if (command_argument_count() >= 11) then
    call get_command_argument(11,arg)
    read(arg,*) dz_last
    if (dz_last == 0) dz_last = nz
  end if
  ! MPAS cu_ntiedtke_pre.F:126-153, cu_ntiedtke.F90:1579-1689.
  ! Probe caller interface-pressure words, never internal plume values.
  ! Nonfinite inputs are diagnostic native-defect probes until independently
  ! repeated; the pristine scheme still computes every expected output word.
  if (command_argument_count() >= 12) then
    call get_command_argument(12,arg)
    read(arg,'(z8)') input_word
    interface_value = transfer(input_word,0.0_kind_phys)
    interface_override = .true.
  end if
  if (command_argument_count() >= 13) then
    call get_command_argument(13,arg)
    read(arg,*) interface_first
    interface_last = interface_first
  end if
  if (command_argument_count() >= 14) then
    call get_command_argument(14,arg)
    read(arg,*) interface_last
  end if

  open(unit=11, file='nt-levels.csv',    status='replace', action='write')
  open(unit=12, file='nt-surface.csv',   status='replace', action='write')
  open(unit=13, file='nt-isolation.csv', status='replace', action='write')
#ifdef NT_MPAS_BINDING
  open(unit=14, file='nt-mpas-forcing.csv', status='replace', action='write')
  write(14,'(a)') 'case,dx,k,rthdynten,rthblten,rthratenlw,rthratensw,rqvdynten,rqvblten'
#endif

  write(11, '(a)') 'case,dx,k,t3d,qv3d,qc3d,qi3d,u3d,v3d,pcps,p8w,dz8w,' // &
       'rho3d,pi3d,w,qvften,thften,rthcuten,rqvcuten,rqccuten,rqicuten,' // &
       'rucuten,rvcuten'
  write(12, '(a)') 'case,dx,scale_fac,scale_fac2,xland,hfx,qfx,psfc,' // &
       'raincv,pratec,cu_act_flag'
  write(13, '(a)') 'case,dx,differing_words'

  do m = 1, ndx
    dxv = nt_dxsweep(m)

    ! scale_fac / scale_fac2 recomputed here exactly as cu_ntiedtke.F90
    ! :230-238 forms them, so the fixture records the dx dependence itself
    ! and a port can be graded on the factor before it is graded on any
    ! physics that consumes it.
    if (dxv < 15000.0) then
      sf  = (1.06133 + log(15000.0 / dxv)) ** 3
      sf2 = sf ** 0.5
    else
      sf  = 1.0 + 1.33e-5 * dxv
      sf2 = 1.0
    end if

    ! ---- pack the slab --------------------------------------------------
    do n = 1, nc
      call nt_build_column(mod(n-1,nt_ncase)+1, nz, b_t, b_qv, b_qc, b_qi, b_u, b_v, &
                           b_pcps, b_p8w, b_dz, b_rho, b_pi, b_w, &
                           b_qvften, b_thften, b_xland, b_hfx, b_qfx)
      b_dz(dz_first:dz_last) = b_dz(dz_first:dz_last) * dz_scale
      b_qv = b_qv * qv_scale
      if (interface_override) b_p8w(interface_first:interface_last) = interface_value
      do k = 1, nz
        t3d(n,k,1)    = b_t(k)
        qv3d(n,k,1)   = b_qv(k)
        qc3d(n,k,1)   = b_qc(k)
        qi3d(n,k,1)   = b_qi(k)
        u3d(n,k,1)    = b_u(k)
        v3d(n,k,1)    = b_v(k)
        pcps(n,k,1)   = b_pcps(k)
        dz8w(n,k,1)   = b_dz(k)
        rho3d(n,k,1)  = b_rho(k)
        pi3d(n,k,1)   = b_pi(k)
        qvften(n,k,1) = b_qvften(k)
        thften(n,k,1) = b_thften(k)
#ifdef NT_MPAS_BINDING
        call mpas_forcing(b_thften(k),b_qvften(k),b_pi(k), &
                          thften(n,k,1),qvften(n,k,1),raw_forcing)
        write(14,'(i0,",",f0.1,",",i0,6(",",a))') n,dxv,k, &
             hexw(raw_forcing(1)),hexw(raw_forcing(2)),hexw(raw_forcing(3)), &
             hexw(raw_forcing(4)),hexw(raw_forcing(5)),hexw(raw_forcing(6))
#endif
      end do
      do k = 1, nz + 1
        p8w(n,k,1) = b_p8w(k)
        w(n,k,1)   = b_w(k)
      end do
      xland(n,1) = b_xland
      hfx(n,1)   = b_hfx
      qfx(n,1)   = b_qfx
      dx(n,1)    = dxv
    end do

    ! Top interface is required by pre conversion; level fields there are
    ! padding only, initialized so the raw-word receipt is deterministic.
    t3d(:,nz+1,:) = 0.0; qv3d(:,nz+1,:) = 0.0
    qc3d(:,nz+1,:) = 0.0; qi3d(:,nz+1,:) = 0.0
    u3d(:,nz+1,:) = 0.0; v3d(:,nz+1,:) = 0.0
    pcps(:,nz+1,:) = 0.0; dz8w(:,nz+1,:) = 0.0
    rho3d(:,nz+1,:) = 0.0; pi3d(:,nz+1,:) = 0.0
    qvften(:,nz+1,:) = 0.0; thften(:,nz+1,:) = 0.0
    raincv = 0.0;  pratec = 0.0;  cu_act_flag = .false.
    rthcuten = 0.0; rqvcuten = 0.0; rqccuten = 0.0003
    rqicuten = -0.0002; rucuten  = 0.0; rvcuten  = 0.0

    call cu_ntiedtke_driver( &
         dt=dt, itimestep=itimestep, stepcu=stepcu, hfx=hfx, &
         raincv=raincv, pratec=pratec, qfx=qfx, &
         u3d=u3d, v3d=v3d, w=w, t3d=t3d, pi3d=pi3d, rho3d=rho3d, &
         qv3d=qv3d, qc3d=qc3d, qi3d=qi3d, &
         dz8w=dz8w, pcps=pcps, p8w=p8w, xland=xland, dx=dx, &
         cu_act_flag=cu_act_flag, &
         ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, &
         ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, &
         its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte, &
         qvften=qvften, thften=thften, &
         f_qc=p_f_qc, &
         f_qi=p_f_qi, &
         rthcuten=rthcuten, rqvcuten=rqvcuten, &
         rqccuten=rqccuten, rqicuten=rqicuten, &
         rucuten=rucuten, rvcuten=rvcuten, &
         grav=real(nt_g,kind_phys), xlf=real(nt_xlf,kind_phys), &
         xls=real(nt_xls,kind_phys), xlv=real(nt_xlv,kind_phys), &
         rd=real(nt_rd,kind_phys), rv=real(nt_rv,kind_phys), &
         cp=real(nt_cp,kind_phys), &
         errmsg=errmsg, errflg=errflg)

    if (errflg /= 0) then
      write(*,'(a,i0,a,a)') 'driver errflg=', errflg, ' : ', trim(errmsg)
      stop 4
    end if

    ! ---- record ---------------------------------------------------------
    do n = 1, nc
      do k = 1, nz + 1
        write(11,'(i0,",",f0.1,",",i0,20(",",a))') n, dxv, k, &
             hexw(t3d(n,k,1)),    hexw(qv3d(n,k,1)),  hexw(qc3d(n,k,1)),  &
             hexw(qi3d(n,k,1)),   hexw(u3d(n,k,1)),   hexw(v3d(n,k,1)),   &
             hexw(pcps(n,k,1)),   hexw(p8w(n,k,1)),   hexw(dz8w(n,k,1)),  &
             hexw(rho3d(n,k,1)),  hexw(pi3d(n,k,1)),  hexw(w(n,k,1)),     &
             hexw(qvften(n,k,1)), hexw(thften(n,k,1)),                    &
             hexw(rthcuten(n,k,1)), hexw(rqvcuten(n,k,1)),                &
             hexw(rqccuten(n,k,1)), hexw(rqicuten(n,k,1)),                &
             hexw(rucuten(n,k,1)),  hexw(rvcuten(n,k,1))
      end do
      write(12,'(i0,",",f0.1,8(",",a),",",l1)') n, dxv, &
           hexw(real(sf,kind_phys)), hexw(real(sf2,kind_phys)), &
           hexw(xland(n,1)), hexw(hfx(n,1)), hexw(qfx(n,1)), &
           hexw(p8w(n,1,1)), hexw(raincv(n,1)), hexw(pratec(n,1)), &
           cu_act_flag(n,1)
    end do

    ! ---- isolation: the same column, alone in its own tile --------------
    ! cuascn has a slab-wide latch. Measure final-word independence before
    ! using packed outputs as an isolated-column oracle.
    do n = 1, nc
      call nt_build_column(mod(n-1,nt_ncase)+1, nz, b_t, b_qv, b_qc, b_qi, b_u, b_v, &
                           b_pcps, b_p8w, b_dz, b_rho, b_pi, b_w, &
                           b_qvften, b_thften, b_xland, b_hfx, b_qfx)
      b_dz(dz_first:dz_last) = b_dz(dz_first:dz_last) * dz_scale
      b_qv = b_qv * qv_scale
      if (interface_override) b_p8w(interface_first:interface_last) = interface_value
      do k = 1, nz
        s_t3d(1,k,1)=b_t(k);      s_qv3d(1,k,1)=b_qv(k)
        s_qc3d(1,k,1)=b_qc(k);    s_qi3d(1,k,1)=b_qi(k)
        s_u3d(1,k,1)=b_u(k);      s_v3d(1,k,1)=b_v(k)
        s_pcps(1,k,1)=b_pcps(k);  s_dz8w(1,k,1)=b_dz(k)
        s_rho3d(1,k,1)=b_rho(k);  s_pi3d(1,k,1)=b_pi(k)
        s_qvften(1,k,1)=b_qvften(k); s_thften(1,k,1)=b_thften(k)
#ifdef NT_MPAS_BINDING
        call mpas_forcing(b_thften(k),b_qvften(k),b_pi(k), &
                          s_thften(1,k,1),s_qvften(1,k,1),raw_forcing)
#endif
      end do
      do k = 1, nz + 1
        s_p8w(1,k,1)=b_p8w(k);    s_w(1,k,1)=b_w(k)
      end do
      s_xland(1,1)=b_xland; s_hfx(1,1)=b_hfx; s_qfx(1,1)=b_qfx
      s_dx(1,1)=dxv
      s_raincv=0.0; s_pratec=0.0; s_cu_act_flag=.false.
      s_rthcuten=0.0; s_rqvcuten=0.0; s_rqccuten=0.0003
      s_rqicuten=-0.0002; s_rucuten=0.0;  s_rvcuten=0.0

      call cu_ntiedtke_driver( &
           dt=dt, itimestep=itimestep, stepcu=stepcu, hfx=s_hfx, &
           raincv=s_raincv, pratec=s_pratec, qfx=s_qfx, &
           u3d=s_u3d, v3d=s_v3d, w=s_w, t3d=s_t3d, pi3d=s_pi3d, &
           rho3d=s_rho3d, qv3d=s_qv3d, qc3d=s_qc3d, qi3d=s_qi3d, &
           dz8w=s_dz8w, pcps=s_pcps, p8w=s_p8w, xland=s_xland, dx=s_dx, &
           cu_act_flag=s_cu_act_flag, &
           ids=1, ide=2, jds=1, jde=2, kds=kds, kde=kde, &
           ims=1, ime=1, jms=1, jme=1, kms=kms, kme=kme, &
           its=1, ite=1, jts=1, jte=1, kts=kts, kte=kte, &
           qvften=s_qvften, thften=s_thften, &
           f_qc=p_f_qc, &
           f_qi=p_f_qi, &
           rthcuten=s_rthcuten, rqvcuten=s_rqvcuten, &
           rqccuten=s_rqccuten, rqicuten=s_rqicuten, &
           rucuten=s_rucuten, rvcuten=s_rvcuten, &
           grav=real(nt_g,kind_phys), xlf=real(nt_xlf,kind_phys), &
           xls=real(nt_xls,kind_phys), xlv=real(nt_xlv,kind_phys), &
           rd=real(nt_rd,kind_phys), rv=real(nt_rv,kind_phys), &
           cp=real(nt_cp,kind_phys), &
           errmsg=errmsg, errflg=errflg)

      ndiff = 0
      do k = 1, nz
        if (wne(s_rthcuten(1,k,1), rthcuten(n,k,1))) ndiff = ndiff + 1
        if (wne(s_rqvcuten(1,k,1), rqvcuten(n,k,1))) ndiff = ndiff + 1
        if (wne(s_rqccuten(1,k,1), rqccuten(n,k,1))) ndiff = ndiff + 1
        if (wne(s_rqicuten(1,k,1), rqicuten(n,k,1))) ndiff = ndiff + 1
        if (wne(s_rucuten(1,k,1),  rucuten(n,k,1)))  ndiff = ndiff + 1
        if (wne(s_rvcuten(1,k,1),  rvcuten(n,k,1)))  ndiff = ndiff + 1
      end do
      if (wne(s_raincv(1,1), raincv(n,1))) ndiff = ndiff + 1
      if (wne(s_pratec(1,1), pratec(n,1))) ndiff = ndiff + 1
      write(13,'(i0,",",f0.1,",",i0)') n, dxv, ndiff
    end do
  end do

  close(11); close(12); close(13)
#ifdef NT_MPAS_BINDING
  close(14)
#endif
  write(*,'(a)') 'run_cu_ntiedtke OK'

#ifdef NT_MPAS_BINDING
contains
  subroutine mpas_forcing(raw_t,raw_q,exner,thermal,vapor,components)
    ! MPAS mpas_atmphys_driver_convection.F:827-851. This synthetic
    ! native oracle records independent pool words before driver folding.
    real,intent(in)::raw_t,raw_q,exner
    real,intent(out)::thermal,vapor,components(6)
    components=(/raw_t,raw_t*0.25,raw_t*(-0.125),raw_t*0.0625,raw_q,raw_q*0.5/)
    thermal=(components(1)+components(2)+components(3)+components(4))*exner
    vapor=components(5)+components(6)
  end subroutine mpas_forcing
#endif
end program run_cu_ntiedtke
