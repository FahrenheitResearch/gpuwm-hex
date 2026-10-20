! Native black-box oracle: pristine pinned RRFS module.
! MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_ra_rrtmg_sw.F:10026-10674 @ 1d6e370e
! Input text per case: flags, time/solar scalars, 7 surface values,
! nz rows of 12 fields, nz+1 rows of pressure/temperature interfaces.
program sw_fixture_driver
  use module_ra_rrtmg_sw, only: rrtmg_swrad
  use module_mp_thompson_aerosols, only: gt_aod
  use module_ra_rrtmg_sw_aerosols, only: calc_aerosol_rrtmg_sw
  use mpas_atmphys_rrtmg_swinit, only: rrtmg_initsw_forMPAS
  use mpas_derived_types, only: dm_info
  implicit none
  type(dm_info) :: dminfo
  integer :: iu,ou,ncase,nz,npin,icase,k,band,icloud,cldovrlp,idcor,o3input,has_reqc,has_reqi,has_reqs,aer_opt
  real :: xtime,gmt,radt,degrad,declin,solcon,julian
  integer :: julday
  character(len=1024) :: infile,outfile
  character(len=32) :: mode
  character(len=8) :: pre
  real, allocatable :: p3d(:,:,:)
  real, allocatable :: t3d(:,:,:)
  real, allocatable :: dz8w(:,:,:)
  real, allocatable :: pi3d(:,:,:)
  real, allocatable :: qv3d(:,:,:)
  real, allocatable :: qc3d(:,:,:)
  real, allocatable :: qi3d(:,:,:)
  real, allocatable :: qs3d(:,:,:)
  real, allocatable :: cldfra3d(:,:,:)
  real, allocatable :: re_cloud(:,:,:)
  real, allocatable :: re_ice(:,:,:)
  real, allocatable :: re_snow(:,:,:)
  real, allocatable :: p8w(:,:,:)
  real, allocatable :: t8w(:,:,:)
  real, allocatable :: rthratensw(:,:,:)
  real, allocatable :: swupflx(:,:,:)
  real, allocatable :: swupflxc(:,:,:)
  real, allocatable :: swdnflx(:,:,:)
  real, allocatable :: swdnflxc(:,:,:)
  real :: latcell(1),loncell(1),ht(1,1),aod2(1,1)
  integer :: aer_type(1,1)
  real, allocatable :: original_qv(:,:,:),nwfa(:,:,:),nifa(:,:,:),aod3(:,:,:)
  real :: xlat(1,1)
  real :: xlong(1,1)
  real :: albedo(1,1)
  real :: tsk(1,1)
  real :: xland(1,1)
  real :: xice(1,1)
  real :: snow(1,1)
  real :: coszr(1,1)
  real :: gsw(1,1)
  real :: swcf(1,1)
  real :: swupt(1,1)
  real :: swuptc(1,1)
  real :: swdnt(1,1)
  real :: swdntc(1,1)
  real :: swupb(1,1)
  real :: swupbc(1,1)
  real :: swdnb(1,1)
  real :: swdnbc(1,1)
  real :: swddir(1,1)
  real :: swddni(1,1)
  real :: swddif(1,1)
  real, allocatable :: tauaer3d(:,:,:,:),ssaaer3d(:,:,:,:),asyaer3d(:,:,:,:)
  real, allocatable :: pin(:),o3clim(:,:,:)
  call get_command_argument(1,infile)
  call get_command_argument(2,outfile)
  call rrtmg_initsw_forMPAS(dminfo)
  open(newunit=iu,file=trim(infile),status='old')
  open(newunit=ou,file=trim(outfile),status='replace',access='stream',form='unformatted')
  write(ou) 'SWD1'
  call get_command_argument(3,mode)
  npin=6
  if(trim(mode)=='real')then
    read(iu,*) ncase,nz,npin
  else
    read(iu,*) ncase,nz
  endif
  allocate(pin(npin),o3clim(1,npin,1))
  allocate(p3d(1,nz+1,1)); p3d=0.
  allocate(t3d(1,nz+1,1)); t3d=0.
  allocate(dz8w(1,nz+1,1)); dz8w=0.
  allocate(pi3d(1,nz+1,1)); pi3d=0.
  allocate(qv3d(1,nz+1,1)); qv3d=0.
  allocate(qc3d(1,nz+1,1)); qc3d=0.
  allocate(qi3d(1,nz+1,1)); qi3d=0.
  allocate(qs3d(1,nz+1,1)); qs3d=0.
  allocate(cldfra3d(1,nz+1,1)); cldfra3d=0.
  allocate(re_cloud(1,nz+1,1)); re_cloud=0.
  allocate(re_ice(1,nz+1,1)); re_ice=0.
  allocate(re_snow(1,nz+1,1)); re_snow=0.
  allocate(p8w(1,nz+1,1)); p8w=0.
  allocate(t8w(1,nz+1,1)); t8w=0.
  allocate(rthratensw(1,nz+1,1)); rthratensw=0.
  allocate(swupflx(1,nz+3,1)); swupflx=0.
  allocate(swupflxc(1,nz+3,1)); swupflxc=0.
  allocate(swdnflx(1,nz+3,1)); swdnflx=0.
  allocate(swdnflxc(1,nz+3,1)); swdnflxc=0.
  if(trim(mode)/='real')pin=[0.01,1.,10.,100.,500.,1100.]
  call get_command_argument(3,mode)
  ! Native vinterp_ozn:148-158, extrapolation above and below the table.
  if(trim(mode)=='ozone-edges') pin=[10.,30.,100.,300.,500.,600.]
  if(trim(mode)/='real')o3clim(1,:,1)=[1.e-7,8.e-6,6.e-6,1.e-6,1.e-7,4.e-8]
  allocate(tauaer3d(1,nz+1,1,14),ssaaer3d(1,nz+1,1,14),asyaer3d(1,nz+1,1,14))
  allocate(original_qv(1,nz+1,1),nwfa(1,nz+1,1),nifa(1,nz+1,1),aod3(1,nz+1,1))
  do icase=1,ncase
    write(pre,'("c",I5.5,"/")') icase
    read(iu,*) icloud,cldovrlp,idcor,o3input,has_reqc,has_reqi,has_reqs,aer_opt,julday
    read(iu,*) xtime,gmt,radt,degrad,declin,solcon,julian
    read(iu,*) xlat(1,1),xlong(1,1),albedo(1,1),tsk(1,1),xland(1,1),xice(1,1),snow(1,1)
    if(trim(mode)=='real')then
      read(iu,*)pin
      read(iu,*)o3clim(1,:,1)
    endif
    do k=1,nz
      read(iu,*) p3d(1,k,1),t3d(1,k,1),dz8w(1,k,1),pi3d(1,k,1),qv3d(1,k,1),qc3d(1,k,1),qi3d(1,k,1),qs3d(1,k,1),cldfra3d(1,k,1),re_cloud(1,k,1),re_ice(1,k,1),re_snow(1,k,1)
    enddo
    do k=1,nz+1
      read(iu,*) p8w(1,k,1),t8w(1,k,1)
    enddo
    tauaer3d=0.;ssaaer3d=1.;asyaer3d=0.
    ! Pristine TEMPO driver ordering (SW:627-679), with distinct qv_p/qvrad_p.
    latcell=xlat(:,1)*degrad;loncell=xlong(:,1)*degrad
    if(trim(mode)/='real')then
      xlat(:,1)=latcell/degrad;xlong(:,1)=loncell/degrad
    endif
    original_qv=qv3d*1.25+0.00002;nwfa=1.e8;nifa=2.e5;ht=1234.5
    aod3=0.;aod2=0.;aer_type=1
    if(xland(1,1)==2.)aer_type=3
    if(aer_opt==3)then
      call gt_aod(p_phy=p3d,dz8w=dz8w,t_phy=t3d,qvapor=original_qv,nwfa=nwfa,nifa=nifa, &
        taod5503d=aod3,ims=1,ime=1,jms=1,jme=1,kms=1,kme=nz+1,its=1,ite=1,jts=1,jte=1,kts=1,kte=nz)
      do k=1,nz
        aod2(1,1)=aod2(1,1)+aod3(1,k,1)
      enddo
      call calc_aerosol_rrtmg_sw(ht=ht,dz8w=dz8w,p=p3d,t3d=t3d,qv3d=original_qv,aer_type=aer_type, &
        aer_aod550_opt=2,aer_angexp_opt=3,aer_ssa_opt=3,aer_asy_opt=3, &
        aer_aod550_val=.12,aer_angexp_val=1.3,aer_ssa_val=.85,aer_asy_val=.9, &
        aod5502d=aod2,aod5503d=aod3,tauaer=tauaer3d,ssaaer=ssaaer3d,asyaer=asyaer3d, &
        ims=1,ime=1,jms=1,jme=1,kms=1,kme=nz+1,its=1,ite=1,jts=1,jte=1,kts=1,kte=nz)
    endif
    call wr1(trim(pre)//'in/lat_cell',latcell)
    call wr1(trim(pre)//'in/lon_cell',loncell)
    call wr1(trim(pre)//'in/nwfa',nwfa(1,:,1))
    call wr1(trim(pre)//'in/nifa',nifa(1,:,1))
    call wr1(trim(pre)//'in/aerosol_qv',original_qv(1,:,1))
    call wr0(trim(pre)//'in/ht',ht(1,1))
    call wr0(trim(pre)//'in/taod5502d',aod2(1,1))
    call wr1(trim(pre)//'in/taod5503d',aod3(1,1:nz,1))
    call wr2(trim(pre)//'in/tauaer',tauaer3d(1,:,1,:))
    call wr2(trim(pre)//'in/ssaaer',ssaaer3d(1,:,1,:))
    call wr2(trim(pre)//'in/asyaer',asyaer3d(1,:,1,:))
    coszr=0.
    gsw=0.
    swcf=0.
    swupt=0.
    swuptc=0.
    swdnt=0.
    swdntc=0.
    swupb=0.
    swupbc=0.
    swdnb=0.
    swdnbc=0.
    swddir=0.
    swddni=0.
    swddif=0.
    rthratensw=0.; swupflx=0.; swupflxc=0.; swdnflx=0.; swdnflxc=0.
    call wr1(trim(pre)//'in/p3d',p3d(1,:,1))
    call wr1(trim(pre)//'in/t3d',t3d(1,:,1))
    call wr1(trim(pre)//'in/dz8w',dz8w(1,:,1))
    call wr1(trim(pre)//'in/pi3d',pi3d(1,:,1))
    call wr1(trim(pre)//'in/qv3d',qv3d(1,:,1))
    call wr1(trim(pre)//'in/qc3d',qc3d(1,:,1))
    call wr1(trim(pre)//'in/qi3d',qi3d(1,:,1))
    call wr1(trim(pre)//'in/qs3d',qs3d(1,:,1))
    call wr1(trim(pre)//'in/cldfra3d',cldfra3d(1,:,1))
    call wr1(trim(pre)//'in/re_cloud',re_cloud(1,:,1))
    call wr1(trim(pre)//'in/re_ice',re_ice(1,:,1))
    call wr1(trim(pre)//'in/re_snow',re_snow(1,:,1))
    call wr1(trim(pre)//'in/p8w',p8w(1,:,1))
    call wr1(trim(pre)//'in/t8w',t8w(1,:,1))
    call wr0(trim(pre)//'in/xlat',xlat(1,1))
    call wr0(trim(pre)//'in/xlong',xlong(1,1))
    call wr0(trim(pre)//'in/albedo',albedo(1,1))
    call wr0(trim(pre)//'in/tsk',tsk(1,1))
    call wr0(trim(pre)//'in/xland',xland(1,1))
    call wr0(trim(pre)//'in/xice',xice(1,1))
    call wr0(trim(pre)//'in/snow',snow(1,1))
    call wr0(trim(pre)//'in/julian',julian)
    call wr0(trim(pre)//'in/xtime',xtime)
    call wr0(trim(pre)//'in/gmt',gmt)
    call wr0(trim(pre)//'in/radt',radt)
    call wr0(trim(pre)//'in/degrad',degrad)
    call wr0(trim(pre)//'in/declin',declin)
    call wr0(trim(pre)//'in/solcon',solcon)
    call wi0(trim(pre)//'in/icloud',icloud)
    call wi0(trim(pre)//'in/cldovrlp',cldovrlp)
    call wi0(trim(pre)//'in/idcor',idcor)
    call wi0(trim(pre)//'in/o3input',o3input)
    call wi0(trim(pre)//'in/has_reqc',has_reqc)
    call wi0(trim(pre)//'in/has_reqi',has_reqi)
    call wi0(trim(pre)//'in/has_reqs',has_reqs)
    call wi0(trim(pre)//'in/aer_opt',aer_opt)
    call wi0(trim(pre)//'in/julday',julday)
    call wi0(trim(pre)//'in/noznlevels',npin)
    call wr1(trim(pre)//'in/pin',pin)
    call wr1(trim(pre)//'in/o3clim',o3clim(1,:,1))
    call rrtmg_swrad( &
      p3d=p3d, &
      t3d=t3d, &
      dz8w=dz8w, &
      pi3d=pi3d, &
      qv3d=qv3d, &
      qc3d=qc3d, &
      qi3d=qi3d, &
      qs3d=qs3d, &
      cldfra3d=cldfra3d, &
      re_cloud=re_cloud, &
      re_ice=re_ice, &
      re_snow=re_snow, &
      xlat=xlat, &
      xlong=xlong, &
      albedo=albedo, &
      tsk=tsk, &
      xland=xland, &
      xice=xice, &
      snow=snow, &
      coszr=coszr, &
      gsw=gsw, &
      swcf=swcf, &
      swupt=swupt, &
      swuptc=swuptc, &
      swdnt=swdnt, &
      swdntc=swdntc, &
      swupb=swupb, &
      swupbc=swupbc, &
      swdnb=swdnb, &
      swdnbc=swdnbc, &
      swddir=swddir, &
      swddni=swddni, &
      swddif=swddif, &
      p8w=p8w, &
      t8w=t8w, &
      rthratensw=rthratensw, &
      swupflx=swupflx, &
      swupflxc=swupflxc, &
      swdnflx=swdnflx, &
      swdnflxc=swdnflxc, &
      xtime=xtime, &
      gmt=gmt, &
      radt=radt, &
      degrad=degrad, &
      declin=declin, &
      solcon=solcon, &
      icloud=icloud, &
      cldovrlp=cldovrlp, &
      idcor=idcor, &
      o3input=o3input, &
      has_reqc=has_reqc, &
      has_reqi=has_reqi, &
      has_reqs=has_reqs, &
      aer_opt=aer_opt, tauaer3d=tauaer3d, ssaaer3d=ssaaer3d, asyaer3d=asyaer3d, &
      julday=julday, julian=julian, &
      pin=pin, &
      o3clim=o3clim, &
      noznlevels=npin, &
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
      kte=nz)
    call wr0(trim(pre)//'out/coszr',coszr(1,1))
    call wr0(trim(pre)//'out/gsw',gsw(1,1))
    call wr0(trim(pre)//'out/swcf',swcf(1,1))
    call wr0(trim(pre)//'out/swupt',swupt(1,1))
    call wr0(trim(pre)//'out/swuptc',swuptc(1,1))
    call wr0(trim(pre)//'out/swdnt',swdnt(1,1))
    call wr0(trim(pre)//'out/swdntc',swdntc(1,1))
    call wr0(trim(pre)//'out/swupb',swupb(1,1))
    call wr0(trim(pre)//'out/swupbc',swupbc(1,1))
    call wr0(trim(pre)//'out/swdnb',swdnb(1,1))
    call wr0(trim(pre)//'out/swdnbc',swdnbc(1,1))
    call wr0(trim(pre)//'out/swddir',swddir(1,1))
    call wr0(trim(pre)//'out/swddni',swddni(1,1))
    call wr0(trim(pre)//'out/swddif',swddif(1,1))
    call wr1(trim(pre)//'out/rthratensw',rthratensw(1,1:nz,1))
    call wr1(trim(pre)//'out/swupflx',swupflx(1,1:nz+2,1))
    call wr1(trim(pre)//'out/swupflxc',swupflxc(1,1:nz+2,1))
    call wr1(trim(pre)//'out/swdnflx',swdnflx(1,1:nz+2,1))
    call wr1(trim(pre)//'out/swdnflxc',swdnflxc(1,1:nz+2,1))
  enddo
  close(iu); close(ou)
contains
  subroutine wname(name, dtype, dims)
    character(len=*), intent(in) :: name
    integer, intent(in) :: dtype
    integer, intent(in) :: dims(:)
    write(ou) int(len_trim(name),4)
    write(ou) trim(name)
    write(ou) int(dtype,4)
    write(ou) int(size(dims),4)
    write(ou) int(dims,4)
  end subroutine wname

  subroutine wr0(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a
    call wname(name, 0, [integer ::])
    write(ou) a
  end subroutine wr0

  subroutine wi0(name, a)
    character(len=*), intent(in) :: name
    integer, intent(in) :: a
    call wname(name, 1, [integer ::])
    write(ou) int(a,4)
  end subroutine wi0

  subroutine wr1(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:)
    call wname(name, 0, shape(a))
    write(ou) a
  end subroutine wr1

  subroutine wi1(name, a)
    character(len=*), intent(in) :: name
    integer, intent(in) :: a(:)
    call wname(name, 1, shape(a))
    write(ou) int(a,4)
  end subroutine wi1

  subroutine wr2(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:,:)
    call wname(name, 0, shape(a))
    write(ou) a
  end subroutine wr2

  subroutine wr3(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:,:,:)
    call wname(name, 0, shape(a))
    write(ou) a
  end subroutine wr3

  subroutine wr4(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:,:,:,:)
    call wname(name, 0, shape(a))
    write(ou) a
  end subroutine wr4

end program sw_fixture_driver
