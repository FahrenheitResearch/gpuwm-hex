! RRFS MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_sf_mynn.F:223-388 @1d6e370e
! Whole-wrapper test ABI; argument mapping reused from the proven p10 bridge.
subroutine native_wrapper(f,constants,profiles,step,flux,spp,water,land,n,levels) bind(c)
use iso_c_binding
use module_sf_mynn
implicit none
integer(c_int),value :: n,levels,step,flux,spp,water,land
real(c_float),intent(inout) :: f(n,54)
real(c_float),intent(in) :: constants(12),profiles(n,levels,1,8)
character(len=256) :: errmsg
integer :: errflg
call sfclay_mynn( &
psfcpa=f(:,11), &
chs=f(:,12), &
chs2=f(:,13), &
cqs2=f(:,14), &
cpm=f(:,15), &
pblh=f(:,16), &
rmol=f(:,17), &
znt=f(:,18), &
ust=f(:,19), &
mavail=f(:,20), &
zol=f(:,21), &
mol=f(:,22), &
regime=f(:,23), &
psim=f(:,24), &
psih=f(:,25), &
xland=f(:,26), &
hfx=f(:,27), &
qfx=f(:,28), &
tsk=f(:,29), &
u10=f(:,30), &
v10=f(:,31), &
th2=f(:,32), &
t2=f(:,33), &
q2=f(:,34), &
flhc=f(:,35), &
flqc=f(:,36), &
snowh=f(:,37), &
qgh=f(:,38), &
qsfc=f(:,39), &
lh=f(:,40), &
gz1oz0=f(:,41), &
wspd=f(:,42), &
br=f(:,43), &
dx=f(:,44), &
ch=f(:,45), &
qcg=f(:,46), &
ustm=f(:,49), &
ck=f(:,50), &
cka=f(:,51), &
cd=f(:,52), &
cda=f(:,53), &
u3d=profiles(:,:,:,1), &
v3d=profiles(:,:,:,2), &
t3d=profiles(:,:,:,3), &
qv3d=profiles(:,:,:,4), &
qc3d=profiles(:,:,:,5), &
p3d=profiles(:,:,:,6), &
rho3d=profiles(:,:,:,7), &
dz8w=profiles(:,:,:,8), &
th3d=profiles(:,:,:,3), &
pi3d=profiles(:,:,:,3), &
cp=constants(1), &
g=constants(2), &
rovcp=constants(3), &
r=constants(4), &
xlv=constants(5), &
svp1=constants(6), &
svp2=constants(7), &
svp3=constants(8), &
svpt0=constants(9), &
ep1=constants(10), &
ep2=constants(11), &
karman=constants(12), &
itimestep=step, &
isfflx=flux, &
spp_pbl=spp, &
isftcflx=water, &
iz0tlnd=land, &
errmsg=errmsg, &
errflg=errflg, &
ids=1, &
ims=1, &
its=1, &
jds=1, &
jms=1, &
jts=1, &
kds=1, &
kms=1, &
kts=1, &
jde=1, &
jme=1, &
jte=1, &
ide=n, &
ime=n, &
ite=n, &
kde=levels, &
kme=levels, &
kte=levels)
end subroutine
