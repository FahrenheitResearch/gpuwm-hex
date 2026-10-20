program deck
 use mpas_atm_dissipation_models
 use mpas_atm_dimensions
 use mpas_constants, only: rvord
 implicit none
#ifndef D1_LEVELS
#define D1_LEVELS 8
#endif
 integer,parameter::nc=4,ne=6,nv=4,nl=D1_LEVELS,ns=6
 integer::e,c,k,j,s,b,sub,es,cs
 integer::coe(2,ne+1),voe(2,ne+1),eoc(3,nc+1),eov(3,nv+1),count(nc+1)
 real::ecs(3,nc+1),evs(3,nv+1),area(nc+1),at(nv+1),dc(ne+1),dv(ne+1),idc(ne+1),idv(ne+1)
 real::density(nc+1),d2(ne+1),d4(ne+1),angle(ne+1),z(nl+1,nc+1),eddy(nl,nc+1),zero(nl,nc+1),one(nl,nc+1)
 real::rho(nl,nc+1),re(nl,ne+1),u(nl,ne+1),v(nl,ne+1),w(nl+1,nc+1),theta(nl,nc+1),initial(nl,nc+1)
 real::real_zz(nl,nc)
 real::div(nl,nc+1),vort(nl,nv+1),ui(nl),vi(nl),zw(nl),zu(nl),fzm(nl),fzp(nl)
 real::q(ns,nl,nc+1),tq(ns,nl,nc+1),tq0(ns,nl,nc+1)
 real::tu(nl,ne+1),tw(nl+1,nc+1),tt(nl,nc+1),lapu(nl,ne+1),lapd(nl,nc+1),lapv(nl,nv+1),lapw(nl,nc+1),lapt(nl,nc+1)
 real::defc2(3,nc+1),defs2(3,nc+1),defcs(3,nc+1),h2,u_seed,diagsign,h4m,h4t,vm,vt,ca(nl,nc+1),ray(nl,ne+1),visc,inverse,coef
 character(len=1024)::name,real_columns,state_mode
 type(MPAS_Clock_type)::clock
 logical::full,mix
 h2=37.5
 call get_command_argument(1,name)
 if(len_trim(name)>0) read(name,*) h2
 u_seed=0.000001
 call get_command_argument(2,name)
 if(len_trim(name)>0) read(name,*) u_seed
 diagsign=1.0
 call get_command_argument(3,name)
 if(len_trim(name)>0) read(name,*) diagsign
 open(20,file='u-seed.bin',access='stream',form='unformatted',convert='little_endian')
 write(20) u_seed
 close(20)
 nVertLevels=nl;maxEdges=3;num_scalars=ns
 coe=1;voe=1;eoc=1;eov=1;count=0;ecs=0;evs=0
 coe(:,1)=[1,2];coe(:,2)=[1,3];coe(:,3)=[1,4];coe(:,4)=[2,3];coe(:,5)=[2,4];coe(:,6)=[3,4]
 voe(:,1)=[3,4];voe(:,2)=[2,4];voe(:,3)=[2,3];voe(:,4)=[1,4];voe(:,5)=[1,3];voe(:,6)=[1,2]
 do c=1,nc
 j=0
 do e=1,ne
 if(any(coe(:,e)==c))then
 j=j+1;eoc(j,c)=e
 ecs(j,c)=merge(1.0,-1.0,coe(1,e)==c)
 end if
 end do
 count(c)=j
 end do
 do c=1,nv
 j=0
 do e=1,ne
 if(any(voe(:,e)==c))then
 j=j+1;eov(j,c)=e
 evs(j,c)=merge(-1.0,1.0,voe(1,e)==c)
 end if
 end do
 end do
 do e=1,ne+1
 dc(e)=12000.0+float(e)*23.5;dv(e)=8000.0+float(e)*12.75
 idc(e)=1.0/dc(e);idv(e)=1.0/dv(e)
 angle(e)=0.13*float(e)-0.4;d2(e)=0.8+float(e)*0.03;d4(e)=1.1+float(e)*0.02
 do k=1,nl
 re(k,e)=0.8+float(e)*0.021+float(k)*0.014
 u(k,e)=3.0+float(e)*0.31+float(k*k)*0.017
 v(k,e)=-1.0+float(e*k)*0.023
 end do
 end do
 do k=1,nl
 ui(k)=2.0+float(k)*0.07;vi(k)=-0.3+float(k*k)*0.02
 zw(k)=1.0/(210.0+float(k)*31.0);zu(k)=1.0/(205.0+float(k)*28.0)
 fzm(k)=0.5;fzp(k)=0.5
 end do
 do c=1,nc+1
 area(c)=1.0/(1.e8+float(c)*127.0);at(c)=1.0/(8.e7+float(c)*237.0)
 do k=1,nl+1
 z(k,c)=float(k-1)*300.0+float(k*k)*8.0+float(c)*9.0
 w(k,c)=float(k*k*c)*0.013-0.24
 end do
 do k=1,nl
 eddy(k,c)=7.0+float(k*c)*0.31
 rho(k,c)=0.82+float(k*c)*0.023
 theta(k,c)=281.0+float(k*k)*0.033+float(c*k)*0.12
 initial(k,c)=279.0+float(k*k*c)*0.027
 div(k,c)=diagsign*float(k*c)*1.7e-5;vort(k,c)=diagsign*float(k-c)*1.3e-5
 do s=1,ns
 q(s,k,c)=float(s*k*c)*0.000017
 tq0(s,k,c)=float(s+c-k)*1.e-7
 end do
 end do
 end do
 zero=0;one=1
 eddy=h2;d2=1.0;d4=1.0
 do c=1,nc+1
 density(c)=0.17+float(c*c)*0.39
 end do
 do c=1,nc+1
 do j=1,3
 defc2(j,c)=float(c*j)*1.e-5
 defs2(j,c)=float(c-j)*0.7e-5
 defcs(j,c)=float(c+j)*0.4e-5
 end do
 end do
 ! Optional Rust-selected real state columns. Geometry and diagnostic
 ! carriers retain the synthetic stencil; this is never a forecast deck.
 call get_command_argument(4,real_columns)
 if(len_trim(real_columns)>0)then
 open(20,file=trim(real_columns),access='stream',form='unformatted',convert='little_endian')
 read(20)z(:,1:nc),rho(:,1:nc),real_zz,u(:,1:ne),w(:,1:nc),theta(:,1:nc),initial(:,1:nc),ui,vi,zw,zu,q(:,:,1:nc)
 close(20)
 call get_command_argument(5,state_mode)
 if(trim(state_mode)=='dry')q(:,:,1:nc)=0
 ! Exact coupled-state driver slice, mpas_atm_time_integration.F:7546-7547.
 do c=1,nc
 do k=1,nl
 theta(k,c)=theta(k,c)*(1.0+rvord*q(1,k,c))
 rho(k,c)=rho(k,c)/real_zz(k,c)
 end do
 end do
 ! Same density staggering as mpas_atm_time_integration.F:7142.
 do e=1,ne
 do k=1,nl
 re(k,e)=0.5*(rho(k,coe(1,e))+rho(k,coe(2,e)))
 end do
 end do
 end if
 ! Synthetic branch corpus with subnormal prognostic/diagnostic inputs.
 ! This changes harness input only; the native module stays pristine.
 call get_command_argument(6,name)
 if(trim(name)=='subnormal')then
 do c=1,nc+1
 do k=1,nl
 div(k,c)=transfer(1048576+k*c,0.0)
 vort(k,c)=transfer(-2146435072+k*c,0.0)
 do s=1,ns
 q(s,k,c)=transfer(1048576+s*k*c,0.0)
 tq0(s,k,c)=0.0
 end do
 end do
 end do
 u=transfer(1048576,0.0);v=transfer(-2146435072,0.0)
 open(20,file='subnormal.txt');write(20,*)'subnormal input corpus';close(20)
 end if
 open(20,file='levels.txt')
 write(20,*)nl
 close(20)
 ! Dump all original inputs in a fixed stream record order for device replay.
 open(20,file='inputs.bin',access='stream',form='unformatted',convert='little_endian')
 write(20)coe,voe,eoc,eov,count,ecs,evs,area,at,dc,dv,idc,idv,d2,d4,angle,z,eddy,rho,re,u,v,w,theta,initial,div,vort,ui,vi,zw,zu,q,tq0,defc2,defs2,defcs,density
 close(20)
 do b=1,256
 d2=1;d4=1
 if(.not.btest(b-1,6))then
 do e=1,ne
 d2(e)=1.0/((density(coe(1,e))+density(coe(2,e)))/2.0)**0.25
 d4(e)=1.0/((density(coe(1,e))+density(coe(2,e)))/2.0)**0.75
 end do
 end if
 full=btest(b-1,0);mix=btest(b-1,1);sub=merge(2,1,btest(b-1,2))
 h4m=9.0e8;h4t=2.0e9;vm=3.7;vt=1.9;es=ne-1;cs=nc-1
 eddy=h2
 if(btest(b-1,3))then
 h4m=0;h4t=0
 end if
 if(btest(b-1,4))then
 vm=0;vt=0
 end if
 if(btest(b-1,5))then
 call smagorinsky_2d(eddy,u,v,0.125,12000.0,defc2,defs2,defcs,0.05,h4m, &
 merge(0.0,0.05,btest(b-1,3)),h4t,1,nc,count,eoc,nc,ne)
 end if
 if(btest(b-1,7))then
 do c=1,nc
 do k=nl-4+1,nl
 visc=4.0*2.0833*12000.0*2.0
 visc=visc*(1.0-real(nl-k)/real(4))
 eddy(k,c)=max(eddy(k,c),visc)
 end do
 end do
 end if
 tu=u_seed;tw=0.000002;tt=0.000003;tq=tq0
 ! Caller-owned scratch is zero-seeded, including branches that leave it untouched.
 ! This defines the harness input without changing the pristine native routine.
 lapu=0;lapd=0;lapv=0;lapw=0;lapt=0
 call u_dissipation_3d(1,ne,1,es,1,nv,1,nc,nc,ne,nv,3,coe,voe,eoc,eov,count,ecs,evs, &
 area,at,idv,idc,angle,dc,dv,d2,d4,full,h4m,vm,10.0,z,eddy,zero,one,zu,zw,fzm,fzp, &
 LES_MODEL_NONE,LES_SURFACE_NONE,0.0,lapu,lapv,lapd,u,v,div,vort,re,rho,ui,vi,[0.0],tu)
 call w_dissipation_3d(1,nc,1,cs,nc,ne,count,eoc,coe,ecs,area,idc,dv,d2,d4,zw,zu,vm,h4m, &
 lapw,w,re,rho,one,div,eddy,zero,LES_MODEL_NONE,LES_SURFACE_NONE,tw)
 call scalar_dissipation_3d_les(1,nc,1,cs,nc,ne,count,eoc,coe,ecs,area,idc,dv,d2,d4,full,initial,z, &
 zw,zu,fzm,fzp,vt,h4t,1.0,one,lapt,theta,re,rho,one,eddy,zero,zero,12000.0,q,tq,0,1,ns,mix, &
 LES_MODEL_NONE,LES_SURFACE_NONE,clock,20.0,0.0,0.0,zero,zero,[0.0],[0.0],tt,sub)
 write(name,'(a,i0,a)')'output-',b,'.bin'
 open(20,file=trim(name),access='stream',form='unformatted',convert='little_endian')
 write(20)tu,tw,tt,tq,lapu,lapd,lapv,lapw,lapt,eddy
 close(20)
 end do
 ! Exact extracted driver blocks: time_integration.F:6390-6408,6610-6629.
 do b=1,3
 ca=37.5;ray=u
 select case(b)
 case(1)
 call native_upper(nl,nc,ne,ca,ray,re,u,12000.0,2.0,4,6,5.0,ne-1)
 case(2)
 call native_upper(nl,nc,ne,ca,ray,re,u,12000.0,0.00001,8,8,0.03125,ne)
 case(3)
 call native_upper(nl,nc,ne,ca,ray,re,u,12000.0,0.0,1,1,2.7,0)
 end select
 write(name,'(a,i0,a)')'upper-',b,'.bin'
 open(20,file=trim(name),access='stream',form='unformatted',convert='little_endian')
 write(20)ca,ray
 close(20)
 end do
end program
