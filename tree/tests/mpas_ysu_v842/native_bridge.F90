! Test-only ISO_C_BINDING adapter; numerical routines remain pristine.
! MPAS v8.4.2 physics_mmm/bl_ysu.F90:20-50,56-1420
! MPAS v8.4.2 physics_wrf/module_bl_ysu.F:16-475
subroutine run_native(arr, scalars, flags) bind(c)
use iso_c_binding
use bl_ysu, only: bl_ysu_run
implicit none
type(c_ptr), intent(in) :: arr(60)
real(c_float), intent(in) :: scalars(11)
integer(c_int), intent(in) :: flags(9)
character(len=256) :: errmsg
integer :: errflg,nz,nmix
real(c_float), pointer :: dz8w2d(:,:)
real(c_float), pointer :: pi2d(:,:)
real(c_float), pointer :: tx(:,:)
real(c_float), pointer :: qvx(:,:)
real(c_float), pointer :: qcx(:,:)
real(c_float), pointer :: qix(:,:)
real(c_float), pointer :: qmix(:,:,:)
real(c_float), pointer :: utnp(:,:)
real(c_float), pointer :: vtnp(:,:)
real(c_float), pointer :: ttnp(:,:)
real(c_float), pointer :: qvtnp(:,:)
real(c_float), pointer :: qctnp(:,:)
real(c_float), pointer :: qitnp(:,:)
real(c_float), pointer :: qmixtnp(:,:,:)
real(c_float), pointer :: p2di(:,:)
real(c_float), pointer :: p2d(:,:)
real(c_float), pointer :: hpbl(:)
real(c_float), pointer :: dusfc(:)
real(c_float), pointer :: dvsfc(:)
real(c_float), pointer :: dtsfc(:)
real(c_float), pointer :: dqsfc(:)
real(c_float), pointer :: ust(:)
real(c_float), pointer :: znt(:)
real(c_float), pointer :: xland(:)
real(c_float), pointer :: hfx(:)
real(c_float), pointer :: qfx(:)
real(c_float), pointer :: wspd(:)
real(c_float), pointer :: br(:)
real(c_float), pointer :: psim(:)
real(c_float), pointer :: psih(:)
real(c_float), pointer :: psfcpa(:)
integer(c_int), pointer :: kpbl1d(:)
real(c_float), pointer :: ux(:,:)
real(c_float), pointer :: vx(:,:)
real(c_float), pointer :: rthraten(:,:)
real(c_float), pointer :: ctopo(:)
real(c_float), pointer :: ctopo2(:)
real(c_float), pointer :: a_u(:,:)
real(c_float), pointer :: a_v(:,:)
real(c_float), pointer :: a_t(:,:)
real(c_float), pointer :: a_q(:,:)
real(c_float), pointer :: a_e(:,:)
real(c_float), pointer :: b_u(:,:)
real(c_float), pointer :: b_v(:,:)
real(c_float), pointer :: b_t(:,:)
real(c_float), pointer :: b_q(:,:)
real(c_float), pointer :: b_e(:,:)
real(c_float), pointer :: sfk(:,:)
real(c_float), pointer :: vlk(:,:)
real(c_float), pointer :: dlu(:,:)
real(c_float), pointer :: dlg(:,:)
real(c_float), pointer :: frcurb(:)
real(c_float), pointer :: exch_hx(:,:)
real(c_float), pointer :: exch_mx(:,:)
real(c_float), pointer :: u10(:)
real(c_float), pointer :: v10(:)
real(c_float), pointer :: uox(:)
real(c_float), pointer :: vox(:)
real(c_float), pointer :: wstar(:)
real(c_float), pointer :: delta(:)
nz=flags(3)
nmix=flags(6)
nullify(dz8w2d)
if(c_associated(arr(1))) call c_f_pointer(arr(1),dz8w2d,[1,nz])
nullify(pi2d)
if(c_associated(arr(2))) call c_f_pointer(arr(2),pi2d,[1,nz])
nullify(tx)
if(c_associated(arr(3))) call c_f_pointer(arr(3),tx,[1,nz])
nullify(qvx)
if(c_associated(arr(4))) call c_f_pointer(arr(4),qvx,[1,nz])
nullify(qcx)
if(c_associated(arr(5))) call c_f_pointer(arr(5),qcx,[1,nz])
nullify(qix)
if(c_associated(arr(6))) call c_f_pointer(arr(6),qix,[1,nz])
nullify(qmix)
if(c_associated(arr(7))) call c_f_pointer(arr(7),qmix,[1,nz,nmix])
nullify(utnp)
if(c_associated(arr(8))) call c_f_pointer(arr(8),utnp,[1,nz])
nullify(vtnp)
if(c_associated(arr(9))) call c_f_pointer(arr(9),vtnp,[1,nz])
nullify(ttnp)
if(c_associated(arr(10))) call c_f_pointer(arr(10),ttnp,[1,nz])
nullify(qvtnp)
if(c_associated(arr(11))) call c_f_pointer(arr(11),qvtnp,[1,nz])
nullify(qctnp)
if(c_associated(arr(12))) call c_f_pointer(arr(12),qctnp,[1,nz])
nullify(qitnp)
if(c_associated(arr(13))) call c_f_pointer(arr(13),qitnp,[1,nz])
nullify(qmixtnp)
if(c_associated(arr(14))) call c_f_pointer(arr(14),qmixtnp,[1,nz,nmix])
nullify(p2di)
if(c_associated(arr(15))) call c_f_pointer(arr(15),p2di,[1,nz+1])
nullify(p2d)
if(c_associated(arr(16))) call c_f_pointer(arr(16),p2d,[1,nz])
nullify(hpbl)
if(c_associated(arr(17))) call c_f_pointer(arr(17),hpbl,[1])
nullify(dusfc)
if(c_associated(arr(18))) call c_f_pointer(arr(18),dusfc,[1])
nullify(dvsfc)
if(c_associated(arr(19))) call c_f_pointer(arr(19),dvsfc,[1])
nullify(dtsfc)
if(c_associated(arr(20))) call c_f_pointer(arr(20),dtsfc,[1])
nullify(dqsfc)
if(c_associated(arr(21))) call c_f_pointer(arr(21),dqsfc,[1])
nullify(ust)
if(c_associated(arr(22))) call c_f_pointer(arr(22),ust,[1])
nullify(znt)
if(c_associated(arr(23))) call c_f_pointer(arr(23),znt,[1])
nullify(xland)
if(c_associated(arr(24))) call c_f_pointer(arr(24),xland,[1])
nullify(hfx)
if(c_associated(arr(25))) call c_f_pointer(arr(25),hfx,[1])
nullify(qfx)
if(c_associated(arr(26))) call c_f_pointer(arr(26),qfx,[1])
nullify(wspd)
if(c_associated(arr(27))) call c_f_pointer(arr(27),wspd,[1])
nullify(br)
if(c_associated(arr(28))) call c_f_pointer(arr(28),br,[1])
nullify(psim)
if(c_associated(arr(29))) call c_f_pointer(arr(29),psim,[1])
nullify(psih)
if(c_associated(arr(30))) call c_f_pointer(arr(30),psih,[1])
nullify(psfcpa)
if(c_associated(arr(31))) call c_f_pointer(arr(31),psfcpa,[1])
nullify(kpbl1d)
if(c_associated(arr(32))) call c_f_pointer(arr(32),kpbl1d,[1])
nullify(ux)
if(c_associated(arr(33))) call c_f_pointer(arr(33),ux,[1,nz])
nullify(vx)
if(c_associated(arr(34))) call c_f_pointer(arr(34),vx,[1,nz])
nullify(rthraten)
if(c_associated(arr(35))) call c_f_pointer(arr(35),rthraten,[1,nz])
nullify(ctopo)
if(c_associated(arr(36))) call c_f_pointer(arr(36),ctopo,[1])
nullify(ctopo2)
if(c_associated(arr(37))) call c_f_pointer(arr(37),ctopo2,[1])
nullify(a_u)
if(c_associated(arr(38))) call c_f_pointer(arr(38),a_u,[1,nz])
nullify(a_v)
if(c_associated(arr(39))) call c_f_pointer(arr(39),a_v,[1,nz])
nullify(a_t)
if(c_associated(arr(40))) call c_f_pointer(arr(40),a_t,[1,nz])
nullify(a_q)
if(c_associated(arr(41))) call c_f_pointer(arr(41),a_q,[1,nz])
nullify(a_e)
if(c_associated(arr(42))) call c_f_pointer(arr(42),a_e,[1,nz])
nullify(b_u)
if(c_associated(arr(43))) call c_f_pointer(arr(43),b_u,[1,nz])
nullify(b_v)
if(c_associated(arr(44))) call c_f_pointer(arr(44),b_v,[1,nz])
nullify(b_t)
if(c_associated(arr(45))) call c_f_pointer(arr(45),b_t,[1,nz])
nullify(b_q)
if(c_associated(arr(46))) call c_f_pointer(arr(46),b_q,[1,nz])
nullify(b_e)
if(c_associated(arr(47))) call c_f_pointer(arr(47),b_e,[1,nz])
nullify(sfk)
if(c_associated(arr(48))) call c_f_pointer(arr(48),sfk,[1,nz])
nullify(vlk)
if(c_associated(arr(49))) call c_f_pointer(arr(49),vlk,[1,nz])
nullify(dlu)
if(c_associated(arr(50))) call c_f_pointer(arr(50),dlu,[1,nz])
nullify(dlg)
if(c_associated(arr(51))) call c_f_pointer(arr(51),dlg,[1,nz])
nullify(frcurb)
if(c_associated(arr(52))) call c_f_pointer(arr(52),frcurb,[1])
nullify(exch_hx)
if(c_associated(arr(53))) call c_f_pointer(arr(53),exch_hx,[1,nz])
nullify(exch_mx)
if(c_associated(arr(54))) call c_f_pointer(arr(54),exch_mx,[1,nz])
nullify(u10)
if(c_associated(arr(55))) call c_f_pointer(arr(55),u10,[1])
nullify(v10)
if(c_associated(arr(56))) call c_f_pointer(arr(56),v10,[1])
nullify(uox)
if(c_associated(arr(57))) call c_f_pointer(arr(57),uox,[1])
nullify(vox)
if(c_associated(arr(58))) call c_f_pointer(arr(58),vox,[1])
nullify(wstar)
if(c_associated(arr(59))) call c_f_pointer(arr(59),wstar,[1])
nullify(delta)
if(c_associated(arr(60))) call c_f_pointer(arr(60),delta,[1])
call bl_ysu_run( &
  its=flags(1), &
  ite=flags(2), &
  kte=flags(3), &
  kme=flags(4), &
  ysu_topdown_pblmix=flags(5)/=0, &
  nmix=flags(6), &
  dt=scalars(1), &
  cp=scalars(2), &
  g=scalars(3), &
  rovcp=scalars(4), &
  rovg=scalars(5), &
  rd=scalars(6), &
  xlv=scalars(7), &
  rv=scalars(8), &
  ep1=scalars(9), &
  ep2=scalars(10), &
  karman=scalars(11), &
  f_qc=flags(7)/=0, &
  f_qi=flags(8)/=0, &
  dz8w2d=dz8w2d, &
  pi2d=pi2d, &
  tx=tx, &
  qvx=qvx, &
  qcx=qcx, &
  qix=qix, &
  qmix=qmix, &
  utnp=utnp, &
  vtnp=vtnp, &
  ttnp=ttnp, &
  qvtnp=qvtnp, &
  qctnp=qctnp, &
  qitnp=qitnp, &
  qmixtnp=qmixtnp, &
  p2di=p2di, &
  p2d=p2d, &
  hpbl=hpbl, &
  dusfc=dusfc, &
  dvsfc=dvsfc, &
  dtsfc=dtsfc, &
  dqsfc=dqsfc, &
  ust=ust, &
  znt=znt, &
  xland=xland, &
  hfx=hfx, &
  qfx=qfx, &
  wspd=wspd, &
  br=br, &
  psim=psim, &
  psih=psih, &
  psfcpa=psfcpa, &
  kpbl1d=kpbl1d, &
  ux=ux, &
  vx=vx, &
  rthraten=rthraten, &
  ctopo=ctopo, &
  ctopo2=ctopo2, &
  flag_bep=flags(9)/=0, &
  a_u=a_u, &
  a_v=a_v, &
  a_t=a_t, &
  a_q=a_q, &
  a_e=a_e, &
  b_u=b_u, &
  b_v=b_v, &
  b_t=b_t, &
  b_q=b_q, &
  b_e=b_e, &
  sfk=sfk, &
  vlk=vlk, &
  dlu=dlu, &
  dlg=dlg, &
  frcurb=frcurb, &
  errmsg=errmsg, &
  errflg=errflg, &
  exch_hx=exch_hx, &
  exch_mx=exch_mx, &
  u10=u10, &
  v10=v10, &
  uox=uox, &
  vox=vox, &
  wstar=wstar, &
  delta=delta)
end subroutine

! Standalone YSU driver slice, with common framework carriers supplied by
! the caller. MPAS v8.4.2 mpas_atmphys_driver_pbl.F:354-416,599-628,832-852.
! The pristine wrapper remains the callee; only pool lookup is replaced by ABI.
subroutine run_driver(arr, scalars, flags, lw, sw, diagnostic_exch_h) bind(c)
use iso_c_binding
implicit none
type(c_ptr), intent(in) :: arr(60)
real(c_float), intent(in) :: scalars(11)
integer(c_int), intent(in) :: flags(9)
real(c_float), intent(in) :: lw(flags(3)), sw(flags(3))
real(c_float), intent(out) :: diagnostic_exch_h(flags(3))
type(c_ptr) :: local_arr(60)
real(c_float), allocatable, target :: radiation(:)
real(c_float), target :: topo(1), topo2(1), ocean_u(1), ocean_v(1), wind_u(1), wind_v(1)
real(c_float), pointer :: original_u(:), original_v(:)
integer :: k
interface
  subroutine run_wrapper(arr, scalars, flags) bind(c)
    use iso_c_binding
    type(c_ptr), intent(in) :: arr(60)
    real(c_float), intent(in) :: scalars(11)
    integer(c_int), intent(in) :: flags(9)
  end subroutine
end interface
local_arr = arr
allocate(radiation(flags(3)+1))
! MPAS driver:378; strict Fortran addition before YSU.
do k = 1,flags(3)
  radiation(k) = lw(k) + sw(k)
enddo
local_arr(35) = c_loc(radiation(1))
! MPAS driver:398-407. u10/v10 are local INOUT copies, not written to pools.
call c_f_pointer(arr(55),original_u,[1])
call c_f_pointer(arr(56),original_v,[1])
wind_u = original_u
wind_v = original_v
local_arr(55) = c_loc(wind_u(1))
local_arr(56) = c_loc(wind_v(1))
topo = 1.0_c_float
topo2 = 1.0_c_float
ocean_u = 0.0_c_float
ocean_v = 0.0_c_float
local_arr(36) = c_loc(topo(1))
local_arr(37) = c_loc(topo2(1))
local_arr(57) = c_loc(ocean_u(1))
local_arr(58) = c_loc(ocean_v(1))
! MPAS driver:412-416,628. Diagnostic exch_p remains zero; kzh is distinct.
diagnostic_exch_h = 0.0_c_float
call run_wrapper(local_arr,scalars,flags)
end subroutine run_driver
subroutine run_wrapper(arr, scalars, flags) bind(c)
use iso_c_binding
use module_bl_ysu, only: ysu
implicit none
type(c_ptr), intent(in) :: arr(60)
real(c_float), intent(in) :: scalars(11)
integer(c_int), intent(in) :: flags(9)
character(len=256) :: errmsg
integer :: errflg,nz,nmix
real(c_float), pointer :: dz8w2d(:,:,:)
real(c_float), pointer :: pi2d(:,:,:)
real(c_float), pointer :: tx(:,:,:)
real(c_float), pointer :: qvx(:,:,:)
real(c_float), pointer :: qcx(:,:,:)
real(c_float), pointer :: qix(:,:,:)
real(c_float), pointer :: qmix(:,:,:)
real(c_float), pointer :: utnp(:,:,:)
real(c_float), pointer :: vtnp(:,:,:)
real(c_float), pointer :: ttnp(:,:,:)
real(c_float), pointer :: qvtnp(:,:,:)
real(c_float), pointer :: qctnp(:,:,:)
real(c_float), pointer :: qitnp(:,:,:)
real(c_float), pointer :: qmixtnp(:,:,:)
real(c_float), pointer :: p2di(:,:,:)
real(c_float), pointer :: p2d(:,:,:)
real(c_float), pointer :: hpbl(:,:)
real(c_float), pointer :: dusfc(:,:)
real(c_float), pointer :: dvsfc(:,:)
real(c_float), pointer :: dtsfc(:,:)
real(c_float), pointer :: dqsfc(:,:)
real(c_float), pointer :: ust(:,:)
real(c_float), pointer :: znt(:,:)
real(c_float), pointer :: xland(:,:)
real(c_float), pointer :: hfx(:,:)
real(c_float), pointer :: qfx(:,:)
real(c_float), pointer :: wspd(:,:)
real(c_float), pointer :: br(:,:)
real(c_float), pointer :: psim(:,:)
real(c_float), pointer :: psih(:,:)
real(c_float), pointer :: psfcpa(:,:)
integer(c_int), pointer :: kpbl1d(:,:)
real(c_float), pointer :: ux(:,:,:)
real(c_float), pointer :: vx(:,:,:)
real(c_float), pointer :: rthraten(:,:,:)
real(c_float), pointer :: ctopo(:,:)
real(c_float), pointer :: ctopo2(:,:)
real(c_float), pointer :: a_u(:,:,:)
real(c_float), pointer :: a_v(:,:,:)
real(c_float), pointer :: a_t(:,:,:)
real(c_float), pointer :: a_q(:,:,:)
real(c_float), pointer :: a_e(:,:,:)
real(c_float), pointer :: b_u(:,:,:)
real(c_float), pointer :: b_v(:,:,:)
real(c_float), pointer :: b_t(:,:,:)
real(c_float), pointer :: b_q(:,:,:)
real(c_float), pointer :: b_e(:,:,:)
real(c_float), pointer :: sfk(:,:,:)
real(c_float), pointer :: vlk(:,:,:)
real(c_float), pointer :: dlu(:,:,:)
real(c_float), pointer :: dlg(:,:,:)
real(c_float), pointer :: frcurb(:,:)
real(c_float), pointer :: exch_hx(:,:,:)
real(c_float), pointer :: exch_mx(:,:,:)
real(c_float), pointer :: u10(:,:)
real(c_float), pointer :: v10(:,:)
real(c_float), pointer :: uox(:,:)
real(c_float), pointer :: vox(:,:)
real(c_float), pointer :: wstar(:,:)
real(c_float), pointer :: delta(:,:)
nz=flags(3)
nmix=flags(6)
nullify(dz8w2d)
if(c_associated(arr(1))) call c_f_pointer(arr(1),dz8w2d,[1,nz+1,1])
nullify(pi2d)
if(c_associated(arr(2))) call c_f_pointer(arr(2),pi2d,[1,nz+1,1])
nullify(tx)
if(c_associated(arr(3))) call c_f_pointer(arr(3),tx,[1,nz+1,1])
nullify(qvx)
if(c_associated(arr(4))) call c_f_pointer(arr(4),qvx,[1,nz+1,1])
nullify(qcx)
if(c_associated(arr(5))) call c_f_pointer(arr(5),qcx,[1,nz+1,1])
nullify(qix)
if(c_associated(arr(6))) call c_f_pointer(arr(6),qix,[1,nz+1,1])
nullify(qmix)
if(c_associated(arr(7))) call c_f_pointer(arr(7),qmix,[1,nz,nmix])
nullify(utnp)
if(c_associated(arr(8))) call c_f_pointer(arr(8),utnp,[1,nz+1,1])
nullify(vtnp)
if(c_associated(arr(9))) call c_f_pointer(arr(9),vtnp,[1,nz+1,1])
nullify(ttnp)
if(c_associated(arr(10))) call c_f_pointer(arr(10),ttnp,[1,nz+1,1])
nullify(qvtnp)
if(c_associated(arr(11))) call c_f_pointer(arr(11),qvtnp,[1,nz+1,1])
nullify(qctnp)
if(c_associated(arr(12))) call c_f_pointer(arr(12),qctnp,[1,nz+1,1])
nullify(qitnp)
if(c_associated(arr(13))) call c_f_pointer(arr(13),qitnp,[1,nz+1,1])
nullify(qmixtnp)
if(c_associated(arr(14))) call c_f_pointer(arr(14),qmixtnp,[1,nz,nmix])
nullify(p2di)
if(c_associated(arr(15))) call c_f_pointer(arr(15),p2di,[1,nz+1,1])
nullify(p2d)
if(c_associated(arr(16))) call c_f_pointer(arr(16),p2d,[1,nz+1,1])
nullify(hpbl)
if(c_associated(arr(17))) call c_f_pointer(arr(17),hpbl,[1,1])
nullify(dusfc)
if(c_associated(arr(18))) call c_f_pointer(arr(18),dusfc,[1,1])
nullify(dvsfc)
if(c_associated(arr(19))) call c_f_pointer(arr(19),dvsfc,[1,1])
nullify(dtsfc)
if(c_associated(arr(20))) call c_f_pointer(arr(20),dtsfc,[1,1])
nullify(dqsfc)
if(c_associated(arr(21))) call c_f_pointer(arr(21),dqsfc,[1,1])
nullify(ust)
if(c_associated(arr(22))) call c_f_pointer(arr(22),ust,[1,1])
nullify(znt)
if(c_associated(arr(23))) call c_f_pointer(arr(23),znt,[1,1])
nullify(xland)
if(c_associated(arr(24))) call c_f_pointer(arr(24),xland,[1,1])
nullify(hfx)
if(c_associated(arr(25))) call c_f_pointer(arr(25),hfx,[1,1])
nullify(qfx)
if(c_associated(arr(26))) call c_f_pointer(arr(26),qfx,[1,1])
nullify(wspd)
if(c_associated(arr(27))) call c_f_pointer(arr(27),wspd,[1,1])
nullify(br)
if(c_associated(arr(28))) call c_f_pointer(arr(28),br,[1,1])
nullify(psim)
if(c_associated(arr(29))) call c_f_pointer(arr(29),psim,[1,1])
nullify(psih)
if(c_associated(arr(30))) call c_f_pointer(arr(30),psih,[1,1])
nullify(psfcpa)
if(c_associated(arr(31))) call c_f_pointer(arr(31),psfcpa,[1,1])
nullify(kpbl1d)
if(c_associated(arr(32))) call c_f_pointer(arr(32),kpbl1d,[1,1])
nullify(ux)
if(c_associated(arr(33))) call c_f_pointer(arr(33),ux,[1,nz+1,1])
nullify(vx)
if(c_associated(arr(34))) call c_f_pointer(arr(34),vx,[1,nz+1,1])
nullify(rthraten)
if(c_associated(arr(35))) call c_f_pointer(arr(35),rthraten,[1,nz+1,1])
nullify(ctopo)
if(c_associated(arr(36))) call c_f_pointer(arr(36),ctopo,[1,1])
nullify(ctopo2)
if(c_associated(arr(37))) call c_f_pointer(arr(37),ctopo2,[1,1])
nullify(a_u)
if(c_associated(arr(38))) call c_f_pointer(arr(38),a_u,[1,nz+1,1])
nullify(a_v)
if(c_associated(arr(39))) call c_f_pointer(arr(39),a_v,[1,nz+1,1])
nullify(a_t)
if(c_associated(arr(40))) call c_f_pointer(arr(40),a_t,[1,nz+1,1])
nullify(a_q)
if(c_associated(arr(41))) call c_f_pointer(arr(41),a_q,[1,nz+1,1])
nullify(a_e)
if(c_associated(arr(42))) call c_f_pointer(arr(42),a_e,[1,nz+1,1])
nullify(b_u)
if(c_associated(arr(43))) call c_f_pointer(arr(43),b_u,[1,nz+1,1])
nullify(b_v)
if(c_associated(arr(44))) call c_f_pointer(arr(44),b_v,[1,nz+1,1])
nullify(b_t)
if(c_associated(arr(45))) call c_f_pointer(arr(45),b_t,[1,nz+1,1])
nullify(b_q)
if(c_associated(arr(46))) call c_f_pointer(arr(46),b_q,[1,nz+1,1])
nullify(b_e)
if(c_associated(arr(47))) call c_f_pointer(arr(47),b_e,[1,nz+1,1])
nullify(sfk)
if(c_associated(arr(48))) call c_f_pointer(arr(48),sfk,[1,nz+1,1])
nullify(vlk)
if(c_associated(arr(49))) call c_f_pointer(arr(49),vlk,[1,nz+1,1])
nullify(dlu)
if(c_associated(arr(50))) call c_f_pointer(arr(50),dlu,[1,nz+1,1])
nullify(dlg)
if(c_associated(arr(51))) call c_f_pointer(arr(51),dlg,[1,nz+1,1])
nullify(frcurb)
if(c_associated(arr(52))) call c_f_pointer(arr(52),frcurb,[1,1])
nullify(exch_hx)
if(c_associated(arr(53))) call c_f_pointer(arr(53),exch_hx,[1,nz+1,1])
nullify(exch_mx)
if(c_associated(arr(54))) call c_f_pointer(arr(54),exch_mx,[1,nz+1,1])
nullify(u10)
if(c_associated(arr(55))) call c_f_pointer(arr(55),u10,[1,1])
nullify(v10)
if(c_associated(arr(56))) call c_f_pointer(arr(56),v10,[1,1])
nullify(uox)
if(c_associated(arr(57))) call c_f_pointer(arr(57),uox,[1,1])
nullify(vox)
if(c_associated(arr(58))) call c_f_pointer(arr(58),vox,[1,1])
nullify(wstar)
if(c_associated(arr(59))) call c_f_pointer(arr(59),wstar,[1,1])
nullify(delta)
if(c_associated(arr(60))) call c_f_pointer(arr(60),delta,[1,1])
call ysu( &
  dz8w=dz8w2d, &
  pi3d=pi2d, &
  t3d=tx, &
  qv3d=qvx, &
  qc3d=qcx, &
  qi3d=qix, &
  rublten=utnp, &
  rvblten=vtnp, &
  rthblten=ttnp, &
  rqvblten=qvtnp, &
  rqcblten=qctnp, &
  rqiblten=qitnp, &
  p3di=p2di, &
  p3d=p2d, &
  hpbl=hpbl, &
  ust=ust, &
  znt=znt, &
  xland=xland, &
  hfx=hfx, &
  qfx=qfx, &
  wspd=wspd, &
  br=br, &
  psim=psim, &
  psih=psih, &
  psfc=psfcpa, &
  kpbl2d=kpbl1d, &
  u3d=ux, &
  v3d=vx, &
  rthraten=rthraten, &
  ctopo=ctopo, &
  ctopo2=ctopo2, &
  a_u_bep=a_u, &
  a_v_bep=a_v, &
  a_t_bep=a_t, &
  a_q_bep=a_q, &
  a_e_bep=a_e, &
  b_u_bep=b_u, &
  b_v_bep=b_v, &
  b_t_bep=b_t, &
  b_q_bep=b_q, &
  b_e_bep=b_e, &
  sf_bep=sfk, &
  vl_bep=vlk, &
  dl_u_bep=dlu, &
  dlg_bep=dlg, &
  frc_urb2d=frcurb, &
  exch_h=exch_hx, &
  exch_m=exch_mx, &
  u10=u10, &
  v10=v10, &
  uoce=uox, &
  voce=vox, &
  wstar=wstar, &
  delta=delta, &
  dt=scalars(1), &
  cp=scalars(2), &
  g=scalars(3), &
  rovcp=scalars(4), &
  rovg=scalars(5), &
  rd=scalars(6), &
  xlv=scalars(7), &
  rv=scalars(8), &
  ep1=scalars(9), &
  ep2=scalars(10), &
  karman=scalars(11), &
  flag_qc=flags(7)/=0, &
  flag_qi=flags(8)/=0, &
  flag_bep=flags(9)/=0, &
  ysu_topdown_pblmix=flags(5), &
  idiff=0, &
  ids=1, &
  ide=1, &
  jds=1, &
  jde=1, &
  kds=1, &
  ims=1, &
  ime=1, &
  jms=1, &
  jme=1, &
  kms=1, &
  its=1, &
  ite=1, &
  jts=1, &
  jte=1, &
  kts=1, &
  kde=nz+1, &
  kme=nz+1, &
  kte=nz, &
  errmsg=errmsg, &
  errflg=errflg)
end subroutine run_wrapper

subroutine native_lifecycle(flags) bind(c)
use iso_c_binding
use bl_ysu, only: bl_ysu_init,bl_ysu_finalize
implicit none
integer(c_int), intent(out) :: flags(2)
character(len=256) :: errmsg
call bl_ysu_init(errmsg,flags(1))
call bl_ysu_finalize(errmsg,flags(2))
end subroutine native_lifecycle
