! Genuine MPAS module_ra_rrtmg_sw.F:2088-2509, all defined cloud methods.
program oracle_cloud
 use rrtmg_sw_cldprmc,only:cldprmc_sw
 use mpas_atmphys_rrtmg_swinit,only:rrtmg_initsw_forMPAS
 use mpas_derived_types,only:dm_info
 implicit none
 type(dm_info)::dminfo
 integer,parameter::nl=8,ng=112
 integer::caseid,ice,inflag,liq,i,k,u
 character(32)::mode
 integer::firstcase,lastcase
 real::cf(ng,nl),ci(ng,nl),cl(ng,nl),cs(ng,nl),tau(ng,nl),ssa(ng,nl),asy(ng,nl),fsf(ng,nl),taor(ng,nl)
 real::rei(nl),rel(nl),res(nl)
 call get_command_argument(1,mode)
 firstcase=0;lastcase=11
 if(len_trim(mode)>0)then
  firstcase=1;lastcase=1
 endif
 call rrtmg_initsw_forMPAS(dminfo)
 if(trim(mode)=='large-ice')then
  open(newunit=u,file='cloud-large-ice.bin',access='stream',form='unformatted',status='replace')
 else if(trim(mode)=='zero-path')then
  open(newunit=u,file='cloud-zero-path.bin',access='stream',form='unformatted',status='replace')
 else if(len_trim(mode)>0)then
  open(newunit=u,file='cloud-'//trim(mode)//'.bin',access='stream',form='unformatted',status='replace')
 else
  open(newunit=u,file='cloud.bin',access='stream',form='unformatted',status='replace')
 endif
 write(u)nl,ng,lastcase-firstcase+1
 do caseid=firstcase,lastcase
 ice=max(1,mod(caseid,6));inflag=2;liq=1
 if(mod(caseid,6)==0)inflag=0
 do k=1,nl
  rei(k)=20.+real(k)*10.;rel(k)=3.5+real(k)*5.;res(k)=50.+real(k)*10.
  if(caseid>=6)then
   rel(k)=1.5;if(mod(k,2)==0)rel(k)=60.
   rei(k)=5.;if(mod(k,2)==0)rei(k)=131.
   if(ice>=3.and.mod(k,2)==0)rei(k)=140.
   res(k)=5.;if(mod(k,2)==0)res(k)=140.
  endif
  do i=1,ng
   cf(i,k)=1.;if(mod(i+k,7)==0)cf(i,k)=0.
   ci(i,k)=real(mod(i+k,5))*2.1
   cl(i,k)=real(mod(i+k,4))*3.7
   cs(i,k)=0.;if(ice==5)cs(i,k)=real(mod(i+k,6))*4.1
   tau(i,k)=real(mod(i+k,5))*.2
   ssa(i,k)=.91;asy(i,k)=.71;fsf(i,k)=.12
  enddo
  if(trim(mode)=='large-ice')rei(k)=400.
 enddo
 if(trim(mode)=='zero-path')then
  ice=3;cf=1.;ci=0.;cl=0.;cs=0.;tau=.2
 endif
 select case(trim(mode))
 case('other-ice');ice=6;cs=2.
 case('snow-other-ice');ice=3;cs=2.
 case('no-ice');ice=0;ci=0.;cs=0.
 case('no-liquid');liq=0;cl=0.
 case('negative-inflag');inflag=-1
 end select
 write(u)inflag,ice
 write(u)cf,ci,cl,cs,rei,rel,res,tau,ssa,asy,fsf
 call cldprmc_sw(nl,inflag,ice,liq,cf,ci,cl,cs,rei,rel,res,taor,tau,ssa,asy,fsf)
 write(u)taor,tau,ssa,asy
 enddo
 close(u)
end program oracle_cloud
