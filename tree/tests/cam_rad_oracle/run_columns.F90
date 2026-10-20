! Full native CAM oracle. MPAS camrad argument boundary, no radiation replacement.
program cam_columns
  use mpas_kind_types
  use mpas_pool_routines
  use mpas_dmpar
  use mpas_atmphys_constants, only: degrad
  use mpas_atmphys_camrad_init, only: camradinit
  use module_ra_cam, only: camrad
  use module_ra_cam_support, only: indirect, yrdata
  use, intrinsic :: ieee_arithmetic
  implicit none
  integer, parameter :: nc=6, nv=26, no=59, na=29, ns=12, nm=12
  type(mpas_pool_type) :: mesh, input, diag, phys, state
  type(dm_info) :: info
  integer, target :: nct=nc, nvt=nv, nat=na, not=no, nmt=nm
  integer, target :: isul=1,isalt=2,idust=3,iocpho=7,ibcpho=8,iocphi=9,ibcphi=10,ibg=11,ivolc=12
  real, target :: lat(nc),lon(nc),pin(no),ps(nc),hybi(na,nc),pressure(nv,nc)
  real, target :: ozone(nm,no,nc),aerp(ns,na,nc),aern(ns,na,nc)
  real :: t(nc,nv+1,1),qv(nc,nv+1,1),qc(nc,nv+1,1),qr(nc,nv+1,1)
  real :: qi(nc,nv+1,1),qs(nc,nv+1,1),qg(nc,nv+1,1),cld(nc,nv+1,1)
  real :: p(nc,nv+1,1),p8w(nc,nv+1,1),pi(nc,nv+1,1),rho(nc,nv+1,1)
  real :: z(nc,nv+1,1),dz(nc,nv+1,1),fice(nc,nv+1,1),frain(nc,nv+1,1)
  real :: xlat(nc,1),xlon(nc,1),xland(nc,1),xice(nc,1),snow(nc,1)
  real :: emiss(nc,1),tsk(nc,1),albedo(nc,1),oz(nc,no,1,nm)
  real :: acp(nc,na,1,ns),acn(nc,na,1,ns)
  real :: heatlw(nc,nv+1,1),heatsw(nc,nv+1,1)
  real :: glw(nc,1),gsw(nc,1),swcf(nc,1),lwcf(nc,1),olr(nc,1),coszr(nc,1)
  real :: cemiss(nc,nv+1,1),taucldc(nc,nv+1,1),taucldi(nc,nv+1,1)
  real :: flux(nc,1,16)
  real :: abstot(nc,nv+1,nv+1,1),absnxt(nc,nv+1,4,1),emstot(nc,nv+1,1)
  real :: base_t(nc,nv+1,1),base_qv(nc,nv+1,1)
  real :: base_p(nc,nv+1,1),base_p8w(nc,nv+1,1),base_pi(nc,nv+1,1)
  real :: base_acp(nc,na,1,ns),base_acn(nc,na,1,ns),base_ps(nc),base_xlat(nc,1)
  logical :: doabs, fqc, fqr, fqi, fqs, dolw, dosw, diagnostic
  integer :: i,k,m,n,variant,mode,record,u,year,variant_count,mode_count,first_variant
  real :: julian,gmt,xtime
  character(len=256) :: filename

  mesh%nc=>nct; mesh%nv=>nvt; mesh%na=>nat; mesh%no=>not; mesh%nm=>nmt
  mesh%lat=>lat; mesh%lon=>lon
  input%pin=>pin; input%ozone=>ozone
  diag%pressure=>pressure
  phys%hybi=>hybi; phys%ps=>ps
  state%sul=>isul; state%salt=>isalt; state%dust=>idust
  state%ocpho=>iocpho; state%bcpho=>ibcpho; state%ocphi=>iocphi; state%bcphi=>ibcphi
  state%bg=>ibg; state%volc=>ivolc; state%aerp=>aerp; state%aern=>aern
  do i=1,nc
    lat(i)=(-75.0+(i-1)*30.0)*degrad
    lon(i)=0.
  enddo
  call get_command_argument(3,filename)
  if(trim(filename)=='polar') lat=[-90.,-45.,0.,45.,90.,30.]*degrad
  do k=1,nv
    pressure(k,:)=100000.-(k-1)*95000./(nv-1)
  enddo
  aerp=0.; aern=0.
  call camradinit(info,mesh,input,diag,phys,state,1)
  ! Last year in the pinned CO2 table is 2150; choose a supported synthetic year.
  year=yrdata(1)+50
  xlat(:,1)=lat/degrad; xlon(:,1)=lon/degrad
  xland(:,1)=[1.,2.,1.,2.,1.,2.]
  xice(:,1)=[0.,0.,1.,0.5,0.,0.]
  snow(:,1)=[0.,0.,100.,20.,1.,0.]
  emiss=0.97; tsk(:,1)=[290.,300.,265.,275.,285.,295.]
  albedo(:,1)=[0.05,0.1,0.8,0.4,0.2,0.15]
  do m=1,nm
    do k=1,no
      oz(:,k,1,m)=ozone(m,k,:)
    enddo
  enddo
  do n=1,ns
    do k=1,na
      acp(:,k,1,n)=aerp(n,k,:); acn(:,k,1,n)=aern(n,k,:)
    enddo
  enddo
  do k=1,nv+1
    p8w(:,k,1)=100000.*exp(-5.5*real(k-1)/nv)
  enddo
  do k=1,nv
    p(:,k,1)=0.5*(p8w(:,k,1)+p8w(:,k+1,1))
    t(:,k,1)=290.-80.*real(k-1)/(nv-1)
    qv(:,k,1)=0.015*exp(-6.*real(k-1)/(nv-1))
    pi(:,k,1)=(p(:,k,1)/100000.)**(287./1004.5)
    rho(:,k,1)=p(:,k,1)/(287.*t(:,k,1))
    z(:,k,1)=700.*(k-1); dz(:,k,1)=700.
  enddo
  p(:,nv+1,1)=p8w(:,nv+1,1); t(:,nv+1,1)=210.; qv(:,nv+1,1)=1.e-6
  pi(:,nv+1,1)=1.; rho(:,nv+1,1)=1.; z(:,nv+1,1)=18000.; dz(:,nv+1,1)=700.
  fice=0.6; frain=0.2
  base_t=t; base_qv=qv; base_p=p; base_p8w=p8w; base_pi=pi
  base_acp=acp; base_acn=acn; base_ps=ps; base_xlat=xlat
  variant_count=6
  call get_command_argument(2,filename)
  if(len_trim(filename)>0) read(filename,*) variant_count
  if(variant_count<6.or.variant_count>91) error stop 'unsupported synthetic variant count'
  mode_count=5
  if(variant_count>6) mode_count=7
  if(variant_count>24) mode_count=8
  first_variant=1
  call get_command_argument(4,filename)
  if(len_trim(filename)>0) read(filename,*) first_variant
  if(first_variant<1.or.first_variant>variant_count) error stop 'unsupported first variant'
  call get_command_argument(5,filename)
  diagnostic=trim(filename)=='diagnostic'
  call get_command_argument(1,filename)
  if (len_trim(filename)==0) filename='native-columns.bin'
  open(newunit=u,file=trim(filename),form='unformatted',access='stream',status='replace')
  write(u) int(z'43414d38'),nc,nv,no,na,ns,nm
  record=0
  do variant=first_variant,variant_count
    t=base_t; qv=base_qv; p=base_p; p8w=base_p8w; pi=base_pi
    acp=base_acp; acn=base_acn; ps=base_ps; xlat=base_xlat
    qc=0.; qr=0.; qi=0.; qs=0.; qg=0.; cld=0.
    fqc=.true.; fqr=.true.; fqi=.true.; fqs=.true.
    if (variant/=1) then
      cld(:,4:14,1)=0.5
      qc(:,4:14,1)=2.e-4
      qi(:,15:19,1)=1.e-4; qs(:,15:19,1)=5.e-5; cld(:,15:19,1)=0.8
    endif
    select case(variant)
    case(7)
      ! Many cloud fractions/regions exercise overlap ordering and sorting.
      do k=1,nv
        do i=1,nc
          cld(i,k,1)=real(mod(3*k+i,9))/10.
          qc(i,k,1)=real(mod(k+i,7))*1.e-4
          qi(i,k,1)=real(mod(2*k+i,5))*1.e-5
        enddo
      enddo
    case(8)
      cld=1.; qc=0.002; qi=0.0005; qs=0.0001
    case(9)
      cld=0.00001; qc=1.e-12; qi=1.e-12; qs=0.
    case(10)
      qv=0.; qc=0.; qi=0.; qs=0.; cld=0.
    case(11)
      qv=0.1
    case(12)
      t=190.; qv=1.e-8
    case(13)
      t=320.; qv=0.03
    case(14)
      ! Two distinct cloud fractions in each overlap region.
      do k=1,nv
        cld(:,k,1)=0.3+0.4*real(mod(k,2))
      enddo
    case(15)
      ! Three layers/regions with separate sparse cloud fractions.
      cld=0.
      cld(:,3,1)=0.2; cld(:,7,1)=0.6; cld(:,12,1)=0.4
      cld(:,17,1)=0.9; cld(:,22,1)=0.3
    case(16)
      fqc=.false.; fqr=.false.; fqi=.true.; fqs=.true.
    case(17)
      fqc=.true.; fqr=.false.; fqi=.false.; fqs=.true.
      t=280.
    case(18)
      fqc=.true.; fqr=.false.; fqi=.false.; fqs=.true.
      t=260.
    case(20,21)
      ! Reach below/above the native ozone pressure interpolation table.
      do k=1,nv+1
        p8w(:,k,1)=100000.*exp(-9.5*real(k-1)/nv)
        if(variant==21) p8w(:,k,1)=1.2*p8w(:,k,1)
      enddo
      do k=1,nv
        p(:,k,1)=0.5*(p8w(:,k,1)+p8w(:,k+1,1))
      enddo
      p(:,nv+1,1)=p8w(:,nv+1,1)
      pi=(p/100000.)**(287./1004.5)
    case(22)
      t=273.15; fqi=.false.; fqs=.false.
    case(23,24)
      ! Distinct cloud fractions between columns and disjoint vertical regions.
      cld=0.
      do i=1,nc
        do k=1,nv
          if(mod(k,5)/=0) cld(i,k,1)=real(mod(i+k,4)+1)*0.2
        enddo
      enddo
      if(variant==24) cld=1.-cld
    case(25)
      ! Every column is dark for the combined midnight call.
      xlat=0.
    case(26)
      ! All first two ice/cloud tests true, last snow-presence test false.
      fqs=.false.
    case(27)
      fqc=.false.; fqi=.false.; fqs=.true.
    case(28)
      ! Exactly two adjacent cloudy layers, in both fraction orderings.
      cld=0.; cld(:,8,1)=0.2; cld(:,9,1)=0.8
      cld(2:6:2,8,1)=0.8; cld(2:6:2,9,1)=0.2
    case(29)
      ! Native clips negative cumulative aerosol mass before differencing.
      acp=-1.e-14; acn=-2.e-14
    case(30)
      ! Aerosol pressure matching spans unlike columns and table endpoints.
      ps=[1000.,5000.,50000.,100000.,150000.,200000.]
    case(31)
      ! A small increasing cumulative sea-salt profile reaches negative mmr.
      do k=1,na
        acp(:,k,1,2)=1.e-12*real(k)
        acn(:,k,1,2)=8.e-13*real(k)
      enddo
    case(32)
      fqc=.true.; fqr=.false.; fqi=.false.; fqs=.false.
    case(33)
      ! Mixed ozone interpolation and extrapolation in the same native chunk.
      do i=1,nc
        p(i,:,1)=p(i,:,1)*(1.+0.12*real(mod(i,3)))
        p8w(i,:,1)=p8w(i,:,1)*(1.+0.12*real(mod(i,3)))
      enddo
      pi=(p/100000.)**(287./1004.5)
    case(34,35,36)
      ! More than fifteen configurations; native pruning and maximum-overlap retry.
      cld=0.;qc=0.;qi=0.;qs=0.
      do k=3,31,4
        cld(:,k,1)=0.5
        if(variant==35) cld(:,k,1)=0.4+0.025*real(k)
        qc(:,k,1)=2.e-4
        if(variant/=36.and.k>=19) exit
      enddo
    case(39)
      ! A monotone aerosol grid whose top interface is above zero reaches
      ! native extrapolation above the first aerosol level.
      hybi(1,:)=0.02
      cld=0.;qc=0.;qi=0.;qs=0.
    case(40:43)
      ! Native warns and continues for negative interpolated aerosol mass.
      ! Mixed signs can reach its g/f optical-property diagnostic checks.
      ! These explicit diagnostic inputs are not physical case defaults.
      acp=0.;acn=0.
      do k=1,na
        acp(:,k,1,2)=1.e-5*real(k)*real(variant-39)
        acp(:,k,1,8)=1.e-5*real(na-k+1)
        if(variant>=42) then
          acp(:,k,1,2)=1.e-5*real(na-k+1)*real(variant-41)
          acp(:,k,1,8)=1.e-5*real(k)
        endif
      enddo
      acn=acp
      cld=0.;qc=0.;qi=0.;qs=0.
    case(44:67)
      ! Scan cancellation between sea-salt and black-carbon scattering.
      ! Negative cumulative gradients are explicit diagnostic inputs only.
      acp=0.;acn=0.
      do k=1,na
        if(variant<=55) then
          acp(:,k,1,2)=1.e-5*real(k)*10.**(-4.+0.5*real(variant-44))
          acp(:,k,1,8)=1.e-5*real(na-k+1)
        else
          acp(:,k,1,2)=1.e-5*real(na-k+1)
          acp(:,k,1,8)=1.e-5*real(k)*10.**(-4.+0.5*real(variant-56))
        endif
      enddo
      acn=acp
      cld=0.;qc=0.;qi=0.;qs=0.
    case(68:91)
      ! Unlike sea salt, native does not clamp organic-carbon paths at6216.
      ! Scan signed scattering cancellation with hydrophilic organic carbon
      ! and black carbon; these are explicit nonphysical diagnostic inputs.
      acp=0.;acn=0.
      do k=1,na
        if(variant<=79) then
          acp(:,k,1,9)=1.e-5*real(k)*10.**(-4.+0.5*real(variant-68))
          acp(:,k,1,8)=1.e-5*real(na-k+1)
        else
          acp(:,k,1,9)=1.e-5*real(na-k+1)
          acp(:,k,1,8)=1.e-5*real(k)*10.**(-4.+0.5*real(variant-80))
        endif
      enddo
      acn=acp
      cld=0.;qc=0.;qi=0.;qs=0.
    case(37,38)
      ! Nonuniform columns with weak and strong regions. The compiled LW
      ! threshold is 1.e-80, so every positive RKIND cloud remains cloudy.
      cld=0.;qc=0.;qi=0.;qs=0.
      do i=1,nc
        cld(i,3+mod(i,3),1)=0.2+0.1*real(i)
        qc(i,3+mod(i,3),1)=2.e-4
        cld(i,12+mod(i,2),1)=1.e-8
        qc(i,12+mod(i,2),1)=1.e-8
        cld(i,22+mod(i,4),1)=0.7
        qc(i,22+mod(i,4),1)=2.e-4
        if(variant==38.and.mod(i,2)==0) then
          cld(i,12+mod(i,2),1)=0.4
          qc(i,12+mod(i,2),1)=2.e-4
        endif
      enddo
    end select
    rho=p/(287.*t)
    if (variant==3) then
      fqi=.false.; fqs=.false.
    else if (variant==4) then
      fqi=.false.; fqr=.false.
    else if (variant==5) then
      fqc=.false.; fqr=.false.; fqi=.false.; fqs=.false.
    endif
    indirect=variant/=6
    julian=real((variant-1)*70)
    if(variant>6) julian=real(mod(variant*31,365))
    if(variant==19) julian=364.
    do mode=1,mode_count
      dolw=mode/=4.and.mode/=8; dosw=mode==4.or.mode==5.or.mode==8
      gmt=12.; if(mode==5) gmt=0.
      xtime=0.
      doabs=mode/=2
      if(mode==1.or.mode==3) then
        abstot=0.; absnxt=0.; emstot=0.
      endif
      if(mode==3) doabs=.false. ! zero-cache restart recovery
      ! Native tests the first column once, rather than every column.
      if(mode==6) then
        doabs=.false.; abstot(nc,1,1,1)=0.
      else if(mode==7) then
        doabs=.false.; abstot(1,1,1,1)=0.
      else if(mode==8) then
        doabs=.false. ! SW-only call with absorption update disabled.
        abstot(1,1,1,1)=0.
      endif
      heatlw=0.; heatsw=0.; flux=0.; glw=0.; gsw=0.
      swcf=0.; lwcf=0.; olr=0.; coszr=0.; cemiss=0.; taucldc=0.; taucldi=0.
      call camrad(heatlw,heatsw,dolw,dosw, &
        flux(:,:,1),flux(:,:,2),flux(:,:,3),flux(:,:,4), &
        flux(:,:,5),flux(:,:,6),flux(:,:,7),flux(:,:,8), &
        flux(:,:,9),flux(:,:,10),flux(:,:,11),flux(:,:,12), &
        flux(:,:,13),flux(:,:,14),flux(:,:,15),flux(:,:,16), &
        swcf,lwcf,olr,cemiss,taucldc,taucldi,coszr,gsw,glw,xlat,xlon, &
        albedo,t,tsk,emiss,qv,qc,qr,qi,qs,qg,fqc,fqr,fqi,fqs,fice,frain, &
        p,p8w,z,pi,rho,dz,cld,xland,xice,snow,oz,pin,no,nm, &
        ps,ps,acp,acn,hybi(:,1),4,nv+1,na,ns, &
        gmt,int(julian),julian,year,300.,xtime,0.2,1370.,30.,degrad,3, &
        abstot,absnxt,emstot,doabs, &
        1,nc,1,1,1,nv+1,1,nc,1,1,1,nv+1,1,nc,1,1,1,nv)
      if (.not.diagnostic) then
        if (.not.all(ieee_is_finite(heatlw)).or..not.all(ieee_is_finite(heatsw))) &
          error stop 'oracle nonfinite heating'
        if (.not.all(ieee_is_finite(flux))) error stop 'oracle nonfinite flux'
      endif
      record=record+1
      write(u) record,variant,mode,merge(1,0,doabs)
      write(u) heatlw,heatsw,flux,gsw,glw,swcf,lwcf,olr,coszr,cemiss,taucldc,taucldi,abstot,absnxt,emstot
    enddo
  enddo
  close(u)
  write(*,'(a,i0)') 'CAM_NATIVE_RECORDS=',record
end program
