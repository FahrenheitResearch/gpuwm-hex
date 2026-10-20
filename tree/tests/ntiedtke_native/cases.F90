module nt_cases
use iso_fortran_env, only: int32, int64
use ccpp_kind_types, only: kind_phys
implicit none
public
#ifndef NT_NZ
#define NT_NZ 49
#endif
  integer, parameter :: nt_nz    = NT_NZ
  real :: nt_sigma_power = 1.35
  logical :: nt_irregular_bottom = .false.
  integer, parameter :: nt_ncase = 18
  integer, parameter :: nt_ndx   = 6

  ! dx sweep chosen against cu_ntiedtke.F90:228-239, the whole reason this
  ! port exists:
  !
  !     dxref = 15000.
  !     if (dx < dxref) then
  !        scale_fac  = (1.06133 + log(dxref/dx))**3
  !        scale_fac2 = scale_fac**0.5
  !     else
  !        scale_fac  = 1. + 1.33e-5*dx
  !        scale_fac2 = 1.
  !     end if
  !
  ! 4500 and 13500 are the reference tropical-cyclone nests and are mandatory -- they are the
  ! grid spacings the intensity campaign measured Grell-Freitas dying at.
  ! 1500 walks the sub-gray-zone branch out to scale_fac = 38.07.  15000 is
  ! the BRANCH BOUNDARY ITSELF and is the one that earns its place: the test
  ! is `<`, not `<=`, so dx = 15000 takes the ELSE arm and gets
  ! scale_fac = 1.1995, where the limit from below is 1.06133**3 = 1.1956.
  ! The function is DISCONTINUOUS at 15 km by 0.0039, and a port that
  ! transcribes the comparison as `<=` is caught by exactly this column and
  ! by nothing else in the sweep.
  real, parameter :: nt_dxsweep(nt_ndx) = &
       (/ 1500.0, 4500.0, 9000.0, 13500.0, 15000.0, 27000.0 /)

  ! WRF module_model_constants, single-precision build -- what the WRF solver
  ! hands the cumulus driver, and what module_cumulus_driver.F:1404-1405
  ! passes through as grav/xlf/xls/xlv/rd/rv/cp.  cu_ntiedtke_init copies
  ! these into cu_ntiedtke_common, so unlike Grell-Freitas (which takes its
  ! own from module_gfs_physcons and disagrees with WRF's) New Tiedtke runs
  ! on the caller's constants and there is no second set to reconcile.
  real, parameter :: nt_g       = 9.80616
  real, parameter :: nt_rd      = 287.0
  real, parameter :: nt_rv      = 461.6
  real, parameter :: nt_cp      = 7.0 * nt_rd / 2.0
  real, parameter :: nt_xlv     = 2.5e6
  real, parameter :: nt_xlf     = 3.50e5
  real, parameter :: nt_xls     = nt_xlv + nt_xlf
  real, parameter :: nt_rovcp   = nt_rd / nt_cp
  real, parameter :: nt_p1000mb = 100000.0

  real, parameter :: svp1 = 0.6112, svp2 = 17.67, svp3 = 29.65
  real, parameter :: svpt0 = 273.15, ep_2 = nt_rd / nt_rv

  ! per-case scalars
  real, dimension(nt_ncase) :: c_psfc, c_tsfc, c_lapse, c_rhsfc, c_rhmid
  real, dimension(nt_ncase) :: c_rhtop, c_ztrop, c_hfx, c_qfx, c_xland
  real, dimension(nt_ncase) :: c_ubase, c_ushear, c_vbase, c_wamp
  real, dimension(nt_ncase) :: c_thften, c_qvften, c_zinv, c_dtinv
  real, dimension(nt_ncase) :: c_qcamp, c_qiamp

contains

  subroutine nt_varied_case_table(seed)
    ! Additional complete-driver inputs; expected outputs remain pristine
    ! MPAS module_cu_ntiedtke.F:158-273, cu_ntiedtke.F90:148-332.
    integer,intent(in)::seed
    integer(int64)::state
    integer::n
    state=int(seed,int64)
    nt_sigma_power=1.0+next_fraction()*1.0
    do n=1,nt_ncase
      c_psfc(n)=75000.+next_fraction()*28000.
      c_tsfc(n)=260.+next_fraction()*55.
      c_lapse(n)=0.003+next_fraction()*0.004
      c_rhsfc(n)=0.15+next_fraction()*1.15
      c_rhmid(n)=0.10+next_fraction()*1.20
      c_rhtop(n)=0.05+next_fraction()*1.20
      c_ztrop(n)=8000.+next_fraction()*6000.
      c_zinv(n)=next_fraction()*6000.
      c_dtinv(n)=-4.+next_fraction()*24.
      c_hfx(n)=-50.+next_fraction()*850.
      c_qfx(n)=-0.00005+next_fraction()*0.00105
      c_xland(n)=real(1+mod(n,2))
      c_wamp(n)=-0.5+next_fraction()*5.5
      c_ubase(n)=-20.+next_fraction()*40.
      c_ushear(n)=-0.02+next_fraction()*0.04
      c_vbase(n)=-20.+next_fraction()*40.
      c_thften(n)=-0.001+next_fraction()*0.002
      c_qvften(n)=-0.00001+next_fraction()*0.00002
      c_qcamp(n)=next_fraction()*0.002
      c_qiamp(n)=next_fraction()*0.002
    end do
  contains
    real function next_fraction()
      ! Park-Miller arithmetic fits int64; deterministic native input only.
      state=modulo(16807_int64*state,2147483647_int64)
      next_fraction=real(state)/2147483647.
    end function
  end subroutine nt_varied_case_table

  subroutine nt_boundary_case_table(seed)
    ! Complete-driver stress, not prescribed intermediate plume state.
    ! Native authority: cu_ntiedtke.F90:1349-1479,2021-2170.
    integer,intent(in)::seed
    integer::n,group
    call nt_varied_case_table(seed)
    do n=1,nt_ncase
      group=mod(n+seed,6)
      c_tsfc(n)=280.+real(mod(7*n+seed,41))
      c_lapse(n)=0.002+0.001*real(mod(n+seed,11))
      c_hfx(n)=100.*10.**group
      c_qfx(n)=0.0001*real(1+mod(n+seed,10))
      c_wamp(n)=real(mod(3*n+seed,51))
      c_zinv(n)=100.+500.*real(mod(n+seed,10))
      c_dtinv(n)=-10.+5.*real(mod(2*n+seed,9))
      c_rhsfc(n)=0.50+0.05*real(mod(n+seed,11))
      c_rhmid(n)=0.01+0.10*real(mod(2*n+seed,11))
      c_rhtop(n)=0.01+0.10*real(mod(3*n+seed,11))
      c_thften(n)=0.0001*real(mod(n+seed,11)-5)
      c_qvften(n)=0.00001*real(mod(2*n+seed,11)-5)
    end do
  end subroutine nt_boundary_case_table

  subroutine nt_melting_inversion_case_table(seed)
    ! Whole-column melting/evaporation stress. No plume state or output is
    ! assigned: cu_ntiedtke.F90:2941-3026 computes all precipitation repairs.
    integer,intent(in)::seed
    integer::n
    call nt_varied_case_table(seed)
    do n=1,nt_ncase
      c_tsfc(n)=240.+real(mod(7*n+seed,56))
      c_lapse(n)=0.003+0.001*real(mod(3*n+seed,10))
      c_ztrop(n)=10000.+500.*real(mod(n+seed,13))
      c_zinv(n)=250.+250.*real(mod(5*n+seed,17))
      c_dtinv(n)=5.*real(mod(11*n+seed,19))
      c_rhsfc(n)=0.01+0.10*real(mod(3*n+seed,11))
      c_rhmid(n)=0.10+0.10*real(mod(7*n+seed,11))
      c_rhtop(n)=0.10+0.10*real(mod(5*n+seed,11))
      c_wamp(n)=0.5*real(mod(n+seed,31))
      c_hfx(n)=100.*real(mod(7*n+seed,21))
      c_qfx(n)=0.0001*real(mod(5*n+seed,11))
      c_qcamp(n)=0.001*real(mod(3*n+seed,11))
      c_qiamp(n)=0.001*real(mod(7*n+seed,11))
    enddo
  end subroutine nt_melting_inversion_case_table

  ! ==========================================================================
  ! The two helpers every harness needs
  ! ==========================================================================
  ! hexw had FOUR copies and wne THREE.  hexw formats every number in every
  ! oracle CSV and wne decides every proof in the directory: the two
  ! functions with the widest blast radius were the two copied most.  They
  ! agreed, but nothing said so.
  ! Raw IEEE-754 word of a kind_phys scalar, lowercase hex, 8 digits.
  function hexw(x) result(s)
    real(kind=kind_phys), intent(in) :: x
    character(len=8) :: s
    integer(int32) :: wd
    wd = transfer(real(x, kind=4), 1_int32)
    write(s, '(z8.8)') wd
  end function hexw

  ! Bitwise inequality.  NOT `/=`: that is false for +0 vs -0 and true for
  ! NaN vs itself, and both of those are exactly the cases a packing bug
  ! shows up as.
  logical function wne(a, b)
    real(kind=kind_phys), intent(in) :: a, b
    wne = transfer(real(a, kind=4), 1_int32) /= &
          transfer(real(b, kind=4), 1_int32)
  end function wne

  ! ==========================================================================
  ! cu_ntiedtke.F90:3542-3557 and :3566-3573
  ! ==========================================================================
  ! Module-private statement functions in the scheme, so a harness that needs
  ! the SAME WORDS the conversion block gets has to transcribe them.  There
  ! were TWO independent transcriptions -- run_nt_cuinin.F90 and
  ! run_nt_cumastrn.F90 -- written with different variable names and
  ! different line breaks, and nothing compared them.  They agreed, which
  ! was luck rather than a property of the arrangement.
  !
  ! c2es = c1es*rd/rv comes from cu_ntiedtke_common and cu_ntiedtke_init.
  real(kind=kind_phys) function nt_foealfa(tt_in)
    real(kind=kind_phys), intent(in) :: tt_in
    real(kind=kind_phys), parameter :: tmelt_l = 273.16
    real(kind=kind_phys), parameter :: rtwat_l = tmelt_l
    real(kind=kind_phys), parameter :: rtice_l = tmelt_l - 23.
    nt_foealfa = min(1., ((max(rtice_l, min(rtwat_l, tt_in)) - rtice_l) &
                          /(rtwat_l - rtice_l))**2)
  end function nt_foealfa

  real(kind=kind_phys) function nt_foeewm(tt_in)
    real(kind=kind_phys), intent(in) :: tt_in
    real(kind=kind_phys), parameter :: tmelt_l = 273.16
    real(kind=kind_phys), parameter :: c1es_l  = 610.78
    real(kind=kind_phys), parameter :: c3les_l = 17.2693882
    real(kind=kind_phys), parameter :: c3ies_l = 21.875
    real(kind=kind_phys), parameter :: c4les_l = 35.86
    real(kind=kind_phys), parameter :: c4ies_l = 7.66
    real(kind=kind_phys) :: c2es_l
    c2es_l = c1es_l*real(nt_rd,kind_phys)/real(nt_rv,kind_phys)
    nt_foeewm = c2es_l * &
         (nt_foealfa(tt_in)*exp(c3les_l*(tt_in-tmelt_l)/(tt_in-c4les_l)) + &
         (1.-nt_foealfa(tt_in))*exp(c3ies_l*(tt_in-tmelt_l)/(tt_in-c4ies_l)))
  end function nt_foeewm

  ! WRF's own saturation form; used ONLY to build inputs, never to check an
  ! answer.  The scheme computes its own saturation through foeewm
  ! (cu_ntiedtke.F90:3566-3573, the Tetens form over the mixed-phase alpha
  ! ramp) and never sees this.
  real function nt_qsat(t, p)
    real, intent(in) :: t, p
    real :: es
    es = 1000.0 * svp1 * exp(svp2 * (t - svpt0) / (t - svp3))
    nt_qsat = ep_2 * es / max(p - es, 1.0)
  end function nt_qsat

  subroutine nt_build_case_table()
    integer :: n
    ! Defaults: a moist tropical maritime sounding with a deep troposphere,
    ! the state a deep (ktype = 1) plume is expected from.
    do n = 1, nt_ncase
      c_psfc(n)   = 101000.0
      c_tsfc(n)   = 301.5
      c_lapse(n)  = 0.0065
      c_rhsfc(n)  = 0.93
      c_rhmid(n)  = 0.82
      c_rhtop(n)  = 0.30
      c_ztrop(n)  = 16000.0
      c_hfx(n)    = 180.0
      c_qfx(n)    = 3.0e-4
      c_xland(n)  = 2.0          ! water; slimsk = |xland-2| = 0
      c_ubase(n)  = 5.0
      c_ushear(n) = 2.0e-3
      c_vbase(n)  = 0.0
      c_wamp(n)   = 0.06
      c_thften(n) = 2.0e-5
      c_qvften(n) = 2.0e-8
      c_zinv(n)   = 0.0          ! no inversion
      c_dtinv(n)  = 0.0
      c_qcamp(n)  = 0.0
      c_qiamp(n)  = 0.0
    end do

    ! ---- ktype = 1, deep -------------------------------------------------
    ! 1  the default tropical maritime column -- deliberately
    !    left just under the deep threshold, so the fixture
    !    keeps a ktype = 2 control beside the ktype = 1 arm
    ! 2  hotter and wetter: the TC-eyewall-like arm this port exists for
    c_tsfc(2)  = 302.5;  c_rhsfc(2) = 0.95;  c_rhmid(2) = 0.85
    c_hfx(2)   = 250.0;  c_qfx(2)   = 4.0e-4
    c_ubase(2) = 18.0;   c_ushear(2) = 4.0e-4
    ! 3  deep with strong large-scale forcing (nonequil arm reads ptte/pqte)
    c_tsfc(3) = 302.5;  c_rhsfc(3) = 0.95;  c_rhmid(3) = 0.86
    c_hfx(3) = 260.0;   c_qfx(3) = 4.2e-4
    c_thften(3) = 6.0e-5;  c_qvften(3) = 6.0e-8
    ! 4  deep with strong subsidence forcing of the opposite sign
    c_tsfc(4) = 302.8;  c_rhsfc(4) = 0.96;  c_rhmid(4) = 0.87
    c_hfx(4) = 270.0;   c_qfx(4) = 4.4e-4
    c_thften(4) = -6.0e-5; c_qvften(4) = -6.0e-8
    ! 5  deep over land: slimsk = 1 changes the entrainment/trigger path
    c_xland(5) = 1.0;   c_tsfc(5) = 303.0;  c_rhsfc(5) = 0.94
    c_rhmid(5) = 0.85;  c_hfx(5) = 320.0;   c_qfx(5) = 3.6e-4
    ! 6  deep, cold-topped: a low tropopause shortens the plume
    c_tsfc(6) = 302.5;  c_rhsfc(6) = 0.95;  c_rhmid(6) = 0.86
    c_hfx(6) = 255.0;   c_qfx(6) = 4.1e-4
    c_ztrop(6) = 12500.0
    ! 7  deep with pre-existing condensate on the column
    c_tsfc(7) = 302.6;  c_rhsfc(7) = 0.95;  c_rhmid(7) = 0.87
    c_hfx(7) = 265.0;   c_qfx(7) = 4.3e-4
    c_qcamp(7) = 4.0e-4;  c_qiamp(7) = 1.0e-4

    ! ---- ktype = 2, shallow ---------------------------------------------
    ! 8  trade cumulus: strong subsidence inversion, dry aloft
    c_wamp(8) = 0.015;  c_thften(8) = 5.0e-6;  c_zinv(8) = 2000.0;  c_dtinv(8) = 6.0
    c_rhmid(8) = 0.7;   c_rhtop(8) = 0.3;  c_hfx(8) = 140.0
    ! 9  shallower and drier still
    c_wamp(9) = 0.015;  c_thften(9) = 5.0e-6;  c_zinv(9) = 1200.0;  c_dtinv(9) = 8.0
    c_rhmid(9) = 0.66;   c_rhtop(9) = 0.26
    c_qfx(9)  = 2.4e-4;  c_hfx(9) = 120.0
    ! 10 shallow over land with a hot dry boundary layer
    c_xland(10) = 1.0;   c_wamp(10) = 0.015;  c_thften(10) = 5.0e-6;  c_zinv(10) = 2500.0; c_dtinv(10) = 5.0
    c_tsfc(10) = 305.0;  c_rhsfc(10) = 0.85;  c_rhmid(10) = 0.64
    c_hfx(10)  = 350.0;  c_qfx(10) = 2.6e-4
    ! 11 shallow, weak fluxes -- near the trigger's lower edge
    ! 11 THE CLOSURE'S SHALLOW ARM.  :716 -- zmfub1/scale_fac2 -- is the
    !    only use of scale_fac2 in the whole scheme, and only a column that
    !    is still ktype = 2 AT THE CLOSURE reaches it.  cumastrn:566
    !    promotes anything whose cloud is >= zdnoprc = 2e4 Pa deep, and
    !    every earlier "shallow" case rose 885 hPa and was promoted.
    !
    !    Two things are needed at once, which is why three earlier rounds of
    !    tuning missed it: case 1's TRIGGER strength so cutypen accepts it
    !    (cases 8-11 failed here -- wamp 0.015 and thften 5.0e-6 are too
    !    weak and cutypen rejected them outright), AND a hard cap just above
    !    cloud base so cuascn's plume terminates inside 200 hPa.  Full
    !    surface forcing, strong low inversion, dry above it.
    c_zinv(11) = 1400.0; c_dtinv(11) = 12.0
    c_hfx(11) = 180.0;   c_qfx(11) = 3.0e-4
    c_rhmid(11) = 0.35;  c_rhtop(11) = 0.15

    ! ---- ktype = 3, mid-level -------------------------------------------
    ! cubasmcn (cu_ntiedtke.F90:3406-3485) starts a mid-level plume from an
    ! elevated moist layer when the surface parcel is capped.  These three
    ! put the moisture aloft and cap the boundary layer.
    ! 12 elevated moist layer over a capped, dry boundary layer
    c_rhsfc(12) = 0.40;  c_rhmid(12) = 0.92;  c_rhtop(12) = 0.30
    c_zinv(12)  = 900.0; c_dtinv(12) = 9.0
    c_hfx(12)   = 30.0;  c_qfx(12) = 4.0e-5
    ! 13 the same with stronger flow (mid-level closure is shear-sensitive)
    c_rhsfc(13) = 0.40;  c_rhmid(13) = 0.90;  c_rhtop(13) = 0.35
    c_zinv(13)  = 800.0; c_dtinv(13) = 10.0
    c_ubase(13) = 20.0;  c_ushear(13) = 3.0e-3;  c_vbase(13) = 8.0
    c_hfx(13)   = 25.0;  c_qfx(13) = 3.0e-5
    ! 14 elevated moist layer over land
    c_xland(14) = 1.0;   c_rhsfc(14) = 0.35;  c_rhmid(14) = 0.88
    c_zinv(14)  = 1000.0; c_dtinv(14) = 8.0
    c_hfx(14)   = 45.0;  c_qfx(14) = 4.0e-5

    ! ---- no convection: ldcum = .false. ---------------------------------
    ! 15 dry and stable throughout
    c_rhsfc(15) = 0.20;  c_rhmid(15) = 0.10;  c_rhtop(15) = 0.05
    c_lapse(15) = 0.0035; c_hfx(15) = 5.0;    c_qfx(15) = 1.0e-6
    c_thften(15) = 0.0;  c_qvften(15) = 0.0;  c_wamp(15) = 0.0
    ! 16 saturated but absolutely stable -- no buoyancy anywhere
    c_rhsfc(16) = 0.95;  c_rhmid(16) = 0.95;  c_rhtop(16) = 0.60
    c_lapse(16) = 0.0020; c_hfx(16) = 0.0;    c_qfx(16) = 0.0
    c_thften(16) = 0.0;  c_qvften(16) = 0.0;  c_wamp(16) = 0.0
    ! 17 cold polar column: everything below freezing, ice-phase paths live
    c_tsfc(17) = 258.0;  c_lapse(17) = 0.0055; c_ztrop(17) = 9000.0
    c_rhsfc(17) = 0.80;  c_rhmid(17) = 0.60;   c_rhtop(17) = 0.20
    c_xland(17) = 1.0;   c_hfx(17) = -20.0;    c_qfx(17) = 1.0e-6
    ! 18 near-freezing maritime: straddles the foealfa mixed-phase ramp
    !    (rtice = tmelt-23 .. rtwat = tmelt), where fliq/fice both matter
    c_tsfc(18) = 274.0;  c_lapse(18) = 0.0060; c_ztrop(18) = 10000.0
    c_rhsfc(18) = 0.92;  c_rhmid(18) = 0.80;   c_rhtop(18) = 0.30
    c_hfx(18)  = 40.0;   c_qfx(18) = 6.0e-5
    c_qcamp(18) = 2.0e-4; c_qiamp(18) = 2.0e-4
  end subroutine nt_build_case_table

  ! Build one column in WRF order (k = 1 surface .. nz top).
  !
  ! Constructed hydrostatically off a prescribed temperature profile so that
  ! p8w, pcps, dz8w and rho3d are mutually consistent to single precision --
  ! cumastrn differences half-level geopotential against pressure in several
  ! places (the zdp/zro/zdz group at :1000-1050) and an inconsistent column
  ! makes those diagnostics meaningless rather than merely different.
  subroutine nt_build_column(n, nz, t3d, qv3d, qc3d, qi3d, u3d, v3d, &
                             pcps, p8w, dz8w, rho3d, pi3d, w, &
                             qvften, thften, xland, hfx, qfx)
    integer, intent(in) :: n, nz
    real, intent(out), dimension(nz)   :: t3d, qv3d, qc3d, qi3d, u3d, v3d
    real, intent(out), dimension(nz)   :: pcps, dz8w, rho3d, pi3d
    real, intent(out), dimension(nz)   :: qvften, thften
    real, intent(out), dimension(nz+1) :: p8w, w
    real, intent(out) :: xland, hfx, qfx

    integer :: k
    real :: ptop, sigma, zfull, tv, rh, zhalf, tk, qs, dz
    real, dimension(nz+1) :: zi

    ptop = 5000.0

    ! Full-level pressures on a stretched sigma ladder: fine near the
    ! surface, coarse aloft, which is the shape a real eta ladder has and
    ! the shape the trigger's 50 hPa departure-layer search assumes.
    do k = 1, nz + 1
      sigma  = real(k - 1) / real(nz)
      sigma  = sigma ** nt_sigma_power
      p8w(k) = c_psfc(n) + sigma * (ptop - c_psfc(n))
    end do
    if (nt_irregular_bottom) then
      ! cutypen:1575-1595: the first two mixing layers exceed 50 hPa
      ! while the departure interface remains within 60 hPa of surface.
      ! The bottom gap stays defined for cuddrafn:2616-2650.
      p8w(2) = c_psfc(n) - 400.
      p8w(3) = c_psfc(n) - 3400.
      p8w(4) = c_psfc(n) - 5900.
      do k = 5, nz + 1
        sigma = real(k - 4) / real(nz - 3)
        p8w(k) = p8w(4) + sigma**1.35 * (ptop - p8w(4))
      end do
    end if

    do k = 1, nz
      pcps(k) = 0.5 * (p8w(k) + p8w(k + 1))
    end do

    ! Temperature: constant lapse to the tropopause, isothermal above, with
    ! an optional inversion of c_dtinv K at c_zinv m.  Integrate the
    ! hypsometric equation upward so z and p agree.
    zi(1) = 0.0
    do k = 1, nz
      ! first guess of layer-mean height from the previous interface
      zhalf = zi(k)
      tk    = c_tsfc(n) - c_lapse(n) * min(zhalf, c_ztrop(n))
      if (c_zinv(n) > 0.0 .and. zhalf > c_zinv(n)) then
        tk = tk + c_dtinv(n)
      end if
      t3d(k) = tk

      rh = c_rhsfc(n)
      if (zhalf > 1000.0) then
        rh = c_rhsfc(n) + (c_rhmid(n) - c_rhsfc(n)) * &
             min(1.0, (zhalf - 1000.0) / 3000.0)
      end if
      if (zhalf > 4000.0) then
        rh = c_rhmid(n) + (c_rhtop(n) - c_rhmid(n)) * &
             min(1.0, (zhalf - 4000.0) / 6000.0)
      end if
      if (c_zinv(n) > 0.0 .and. zhalf > c_zinv(n)) then
        rh = min(rh, c_rhmid(n))
      end if

      qs      = nt_qsat(t3d(k), pcps(k))
      qv3d(k) = max(1.0e-8, rh * qs)

      tv       = t3d(k) * (1.0 + 0.608 * qv3d(k))
      dz       = nt_rd * tv / nt_g * log(p8w(k) / p8w(k + 1))
      dz8w(k)  = dz
      zi(k + 1) = zi(k) + dz

      rho3d(k) = pcps(k) / (nt_rd * tv)
      pi3d(k)  = (pcps(k) / nt_p1000mb) ** nt_rovcp

      qc3d(k) = 0.0
      qi3d(k) = 0.0
      if (c_qcamp(n) > 0.0 .and. zhalf > 500.0 .and. zhalf < 6000.0) then
        qc3d(k) = c_qcamp(n)
      end if
      if (c_qiamp(n) > 0.0 .and. zhalf > 6000.0 .and. zhalf < 12000.0) then
        qi3d(k) = c_qiamp(n)
      end if

      u3d(k) = c_ubase(n) + c_ushear(n) * zhalf
      v3d(k) = c_vbase(n)

      ! Advective + PBL forcing tendencies, tapered above the troposphere.
      ! Native initflag=0 in the baseline so these are actually read.
      ! initflag=1 zeroes them (cu_ntiedtke_pre.F:161-165), changing
      ! the nonequilibrium closure's zcape2 term.
      thften(k) = c_thften(n) * max(0.0, 1.0 - zhalf / c_ztrop(n))
      qvften(k) = c_qvften(n) * max(0.0, 1.0 - zhalf / c_ztrop(n))
    end do

    ! Vertical velocity on full levels: a half-sine bump through the
    ! troposphere.  omg = -0.5*g*rho*(w(k)+w(k+1)) is what the driver forms
    ! from it, so this is the only path by which large-scale ascent reaches
    ! the scheme.
    do k = 1, nz + 1
      zfull = zi(k)
      if (zfull < c_ztrop(n)) then
        w(k) = c_wamp(n) * sin(3.14159265 * zfull / c_ztrop(n))
      else
        w(k) = 0.0
      end if
    end do

    xland = c_xland(n)
    hfx   = c_hfx(n)
    qfx   = c_qfx(n)
  end subroutine nt_build_column

end module nt_cases
