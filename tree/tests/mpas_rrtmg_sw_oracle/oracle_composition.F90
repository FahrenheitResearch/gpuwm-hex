! Pristine MPAS module_ra_rrtmg_sw.F:8867-9622 composition interface.
program oracle_composition
 use rrtmg_sw_rad,only:rrtmg_sw,earth_sun
 use mcica_subcol_gen_sw,only:mcica_subcol_sw
 use mpas_atmphys_rrtmg_swinit,only:rrtmg_initsw_forMPAS
 use mpas_derived_types,only:dm_info
 implicit none
 type(dm_info)::dminfo
 integer,parameter::nc=2,nl=12,ng=112
 integer::u,c,k,b,a,icld,day,aer,zero,rng,native_icld,profile,last_profile,selected_aer
 integer,parameter::unsupported_aerosols(4)=[-1,4,99,0]
 character(len=128)::filename
 character(len=32)::mode
 real::play(nc,nl),plev(nc,nl+1),t(nc,nl),te(nc,nl+1),ts(nc),gas(nc,nl,6),basegas(6)
 real::alb(nc),cosz(nc),adjes(nc),scon
 real::cf(nc,nl),ci(nc,nl),cl(nc,nl),cs(nc,nl),height(nc,nl),rei(nc,nl),rel(nc,nl),res(nc,nl)
 real::tau(14,nc,nl),ssa(14,nc,nl),asy(14,nc,nl),fsf(14,nc,nl)
 real::ocf(ng,nc,nl),oci(ng,nc,nl),ocl(ng,nc,nl),ocs(ng,nc,nl),otau(ng,nc,nl),ossa(ng,nc,nl),oasy(ng,nc,nl),ofsf(ng,nc,nl)
 real::orei(nc,nl),orel(nc,nl),ores(nc,nl),ta(nc,nl,14),sa(nc,nl,14),ga(nc,nl,14),ec(nc,nl,6)
 real::uf(nc,nl+1),df(nc,nl+1),hr(nc,nl),ufc(nc,nl+1),dfc(nc,nl+1),hrc(nc,nl),ufn(nc,nl+1),dfn(nc,nl+1)
 real::vd(nc,nl+1),vf(nc,nl+1),nd(nc,nl+1),nf(nc,nl+1),dd(nc,nl+1),dif(nc,nl+1),ddc(nc,nl+1)
 call rrtmg_initsw_forMPAS(dminfo)
 call get_command_argument(1,mode)
 last_profile=3
 if(trim(mode)=='unsupported-aerosol')last_profile=1
 do profile=1,last_profile
 filename='composition.bin'
 if(profile==2)filename='composition-extreme.bin'
 if(profile==3)filename='composition-dense.bin'
 if(trim(mode)=='unsupported-aerosol')filename='composition-unsupported-aerosol.bin'
 open(newunit=u,file=trim(filename),access='stream',form='unformatted',status='replace')
 write(u)nc,nl,ng,32
 scon=1371.25;alb=[.18,.73];cosz=[.38,.00000000001];adjes=.995;ts=[298.,265.]
 do c=1,nc
 do k=1,nl+1
  plev(c,k)=1100.*exp(-real(k-1)*.91)
  if(profile==2)plev(c,k)=5000.*exp(-real(k-1)*1.87)
  ! Repeated pressure indices below the solar reference layer, then crossing.
  if(profile==3)plev(c,k)=1100.*exp(-real(k-1)*.1)
  te(c,k)=290.-real(k-1)*5.
 enddo
 do k=1,nl
  play(c,k)=.5*(plev(c,k)+plev(c,k+1));t(c,k)=.5*(te(c,k)+te(c,k+1))
  cf(c,k)=real(mod(k+c,5))*.2;ci(c,k)=real(mod(k,3))*1.3;cl(c,k)=real(mod(k,4))*2.7;cs(c,k)=0.
  height(c,k)=250.+real(k*k)*235.7;rei(c,k)=30.+real(k);rel(c,k)=10.+real(k);res(c,k)=50.+real(k)
  do b=1,14
   tau(b,c,k)=0.;ssa(b,c,k)=1.;asy(b,c,k)=0.;fsf(b,c,k)=0.
   ta(c,k,b)=real(b)*.003+real(k)*.002;sa(c,k,b)=.8+real(b)*.01;ga(c,k,b)=.55+real(b)*.01
  enddo
  do a=1,6
   ec(c,k,a)=real(a+c+k)*.001
   if(profile==2.and.(c==1.or.mod(k,2)==0))ec(c,k,a)=0.
  enddo
 enddo
 enddo
 rng=0
 call mcica_subcol_sw(1,nc,nl,2,1,rng,play,cf,ci,cl,cs,rei,rel,res,tau,ssa,asy,fsf, &
  height,1,279,41.25,ocf,oci,ocl,ocs,orei,orel,ores,otau,ossa,oasy,ofsf)
 write(u)play,plev,t,te,ts,alb,cosz,adjes,scon
 write(u)ocf,otau,ossa,oasy,ofsf,oci,ocl,ocs,orei,orel,ores,ta,sa,ga,ec
 basegas=[.004,1.e-6,379.e-6,1.7e-6,319.e-9,.209]
 do zero=0,1
 do day=0,279,279
 do icld=0,2,2
 do aer=0,3
  selected_aer=aer
  if(trim(mode)=='unsupported-aerosol')selected_aer=unsupported_aerosols(aer+1)
  do a=1,6
   gas(:,:,a)=basegas(a)
  enddo
  if(zero==1)gas(:,:,3:6)=0.
  write(u)icld,day,selected_aer,zero
  write(u)gas
  native_icld=icld
  call rrtmg_sw(nc,nl,native_icld,play,plev,t,te,ts,gas(:,:,1),gas(:,:,2),gas(:,:,3),gas(:,:,4),gas(:,:,5),gas(:,:,6), &
   alb,alb,alb,alb,cosz,adjes(1),day,scon,2,2,1,ocf,otau,ossa,oasy,ofsf,oci,ocl,ocs,orei,orel,ores, &
   ta,sa,ga,ec,uf,df,hr,ufc,dfc,hrc,ufn,dfn,selected_aer,vd,vf,nd,nf,dd,dif,ddc,0)
  write(u)uf,df,hr,ufc,dfc,hrc,ufn,dfn,vd,vf,nd,nf,dd,dif,ddc
 enddo
 enddo
 enddo
 enddo
 close(u)
 enddo
 open(newunit=u,file='earth_sun.bin',access='stream',form='unformatted',status='replace')
 do day=0,366
  write(u)day,earth_sun(day)
 enddo
 close(u)
end program oracle_composition
