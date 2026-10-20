program deck
 use mpas_atm_dissipation_models
 use mpas_atm_dimensions
 use mpas_constants, only: rvord
 implicit none
#ifndef D1_LEVELS
#define D1_LEVELS 8
#endif
 integer,parameter::nc=D1_NC,ne=D1_NE,nv=D1_NV,nl=D1_LEVELS,ns=D1_NS,me=D1_ME
 integer::e,c,k,j,s,b,sub,es,cs
 integer::coe(2,ne+1),voe(2,ne+1),eoc(me,nc+1),eov(3,nv+1),count(nc+1)
 real::ecs(me,nc+1),evs(3,nv+1),area(nc+1),at(nv+1),dc(ne+1),dv(ne+1),idc(ne+1),idv(ne+1)
 real::density(nc+1),d2(ne+1),d4(ne+1),angle(ne+1),z(nl+1,nc+1),eddy(nl,nc+1),zero(nl,nc+1),one(nl,nc+1)
 real::rho(nl,nc+1),re(nl,ne+1),u(nl,ne+1),v(nl,ne+1),w(nl+1,nc+1),theta(nl,nc+1),initial(nl,nc+1)
 real::real_zz(nl,nc),actual_area(nc+1),actual_at(nv+1),length,dt
 real::div(nl,nc+1),vort(nl,nv+1),ui(nl),vi(nl),zw(nl),zu(nl),fzm(nl),fzp(nl)
 real::q(ns,nl,nc+1),tq(ns,nl,nc+1),tq0(ns,nl,nc+1)
 real::tu(nl,ne+1),tw(nl+1,nc+1),tt(nl,nc+1),lapu(nl,ne+1),lapd(nl,nc+1),lapv(nl,nv+1),lapw(nl,nc+1),lapt(nl,nc+1)
 real::defc2(me,nc+1),defs2(me,nc+1),defcs(me,nc+1),h2,u_seed,diagsign,h4m,h4t,vm,vt,ca(nl,nc+1),ray(nl,ne+1),visc,inverse,coef
 character(len=1024)::name,real_columns,state_mode
 type(MPAS_Clock_type)::clock
 logical::full,mix
 ! All state and geometry words below come from f1 native PRE captures.
 ! Native callers: mpas_atm_time_integration.F:6319-6410,6585-6598,6880-6906.
 nVertLevels=nl;maxEdges=me;num_scalars=ns
 open(20,file='tap-input.bin',access='stream',form='unformatted',convert='little_endian')
 read(20)coe,voe,eoc,eov,count,ecs,evs,area,at,dc,dv,idc,idv,density,angle,z,rho,re,u,v,w,theta,initial,div,vort,ui,vi,zw,zu,q,defc2,defs2,defcs,actual_area,actual_at,fzm,fzp,length
 close(20)
 h2=37.5;u_seed=0;zero=0;one=1;tq0=0;dt=1200.0
 open(20,file='levels.txt');write(20,*)nl;close(20)
 do b=1,256
 d2=1;d4=1
 if(.not.btest(b-1,6))then
 do e=1,ne
 d2(e)=1.0/((density(coe(1,e))+density(coe(2,e)))/2.0)**0.25
 d4(e)=1.0/((density(coe(1,e))+density(coe(2,e)))/2.0)**0.75
 end do
 end if
 full=btest(b-1,0);mix=btest(b-1,1);sub=merge(2,1,btest(b-1,2))
 h4m=9.0e8;h4t=2.0e9;vm=3.7;vt=1.9;es=ne;cs=nc
 eddy=h2
 if(btest(b-1,3))then
 h4m=0;h4t=0
 end if
 if(btest(b-1,4))then
 vm=0;vt=0
 end if
 if(btest(b-1,5))then
 call smagorinsky_2d(eddy,u,v,0.125,length,defc2,defs2,defcs,1.0/dt,h4m, &
 merge(0.0,0.05,btest(b-1,3)),h4t,1,nc,count,eoc,nc,ne)
 end if
 if(btest(b-1,7))then
 do c=1,nc
 do k=nl-4+1,nl
 visc=4.0*2.0833*length*2.0
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
 zw,zu,fzm,fzp,vt,h4t,1.0,one,lapt,theta,re,rho,one,eddy,zero,zero,length,q,tq,0,1,ns,mix, &
 LES_MODEL_NONE,LES_SURFACE_NONE,clock,dt,0.0,0.0,zero,zero,[0.0],[0.0],tt,sub)
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
 call native_upper(nl,nc,ne,ca,ray,re,u,length,2.0,4,6,5.0,ne-1)
 case(2)
 call native_upper(nl,nc,ne,ca,ray,re,u,length,0.00001,8,8,0.03125,ne)
 case(3)
 call native_upper(nl,nc,ne,ca,ray,re,u,length,0.0,1,1,2.7,0)
 end select
 write(name,'(a,i0,a)')'upper-',b,'.bin'
 open(20,file=trim(name),access='stream',form='unformatted',convert='little_endian')
 write(20)ca,ray
 close(20)
 end do
end program
