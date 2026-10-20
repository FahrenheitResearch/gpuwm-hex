! Synthetic branch deck for unchanged v8.4.2 regional routines.
program fixture
 use direct_regional
 use direct_lbc
 use mpas_atm_dimensions, only: model_nlev => nVertLevels
 use iso_fortran_env, only: int32
 implicit none
 integer, parameter :: nc=10, ne=10, nv=10, nl=4, nt=3
 integer, target :: ncptr=nc, vd=2
 type(mpas_pool_type) :: mesh, state, diag, tend, config
 integer, target :: bc(nc+1),be(ne+1),bv(nv+1),nearest(nc+1),counts(nc+1)
 integer, target :: coc(2,nc+1),coe(2,ne+1),eoc(2,nc+1),voe(2,ne+1),eov(2,nv+1)
 real, target :: smc(nc+1),sme(ne+1),smv(nv+1),x(nc+1),y(nc+1),z(nc+1)
 real, target :: scale_c(nc+1),scale_e(ne+1),dc(ne+1),dv(ne+1),idc(ne+1),idv(ne+1),iac(nc+1),iat(nv+1)
 real, target :: esc(2,nc+1),esv(2,nv+1),divdamp
 real, target :: tr(nl,nc+1),tt(nl,nc+1),tu(nl,ne+1),tw(nl+1,nc+1),diabatic(nl,nc+1)
 real, target :: rho(nl,nc+1),theta(nl,nc+1),ru(nl,ne+1),rp(nl,nc+1),base(nl,nc+1)
 real :: rd(nl,nc+1),td(nl,nc+1),ud(nl,ne+1),q(nt,nl,nc+1),qd(nt,nl,nc+1)
 real :: dt,dt_rk,interval,remaining
 real :: zz(nl,nc+1),lbc_rho(nl,nc+1),lbc_rt(nl,nc+1),lbc_re(nl,ne+1),lbc_ru(nl,ne+1)
 real :: old_u(nl,ne+1),old_ru(nl,ne+1),old_re(nl,ne+1),old_w(nl+1,nc+1)
 real :: old_theta(nl,nc+1),old_rt(nl,nc+1),old_rho(nl,nc+1),old_r(nl,nc+1),old_q(nt,nl,nc+1)
 real :: at_q(nt,nl,nc+1),density(nc+1),scaling_c(nc+1),scaling_e(ne+1)
 real :: reset_u(nl,ne+1),reset_ru(nl,ne+1),drive_ru(nl,ne+1),clamp_values(nt,nl,nc+1)
 integer(int32),parameter :: clamp_bits(11)=[int(z'80000000',int32),int(z'00000000',int32), &
   int(z'80000001',int32),int(z'00000001',int32),int(z'BF800000',int32),int(z'3F800000',int32), &
   int(z'FF800000',int32),int(z'7F800000',int32),int(z'7FC01234',int32),int(z'FFC01234',int32),int(z'7F801234',int32)]
 integer(int32) :: clamp_raw(nt*nl*(nc+1))
 integer :: lbc_coe(2,ne+1)
 integer :: i,k,t,caseid,previous,following
 character(32) :: directory
 model_nlev=nl
 mesh%nCells=>ncptr; mesh%vertexDegree=>vd
 mesh%bdyMaskCell=>bc; mesh%bdyMaskEdge=>be; mesh%bdyMaskVertex=>bv
 mesh%nearestRelaxationCell=>nearest; mesh%nEdgesOnCell=>counts; mesh%cellsOnCell=>coc
 mesh%cellsOnEdge=>coe; mesh%edgesOnCell=>eoc; mesh%verticesOnEdge=>voe; mesh%edgesOnVertex=>eov
 mesh%specZoneMaskCell=>smc; mesh%specZoneMaskEdge=>sme; mesh%specZoneMaskVertex=>smv
 mesh%xCell=>x; mesh%yCell=>y; mesh%zCell=>z
 mesh%meshScalingRegionalCell=>scale_c; mesh%meshScalingRegionalEdge=>scale_e
 mesh%dcEdge=>dc; mesh%dvEdge=>dv; mesh%invDcEdge=>idc; mesh%invDvEdge=>idv
 mesh%invAreaCell=>iac; mesh%invAreaTriangle=>iat
 mesh%edgesOnCell_sign=>esc; mesh%edgesOnVertex_sign=>esv
 state%rho_zz=>rho; state%theta_m=>theta
 diag%ru=>ru; diag%rtheta_p=>rp; diag%rtheta_base=>base
 tend%rho_zz=>tr; tend%theta_m=>tt; tend%u=>tu; tend%w=>tw; tend%rt_diabatic_tend=>diabatic
 config%config_relax_zone_divdamp_coef=>divdamp
 do caseid=1,24
   write(directory,'("case-",i2.2)') caseid
   call execute_command_line('mkdir -p '//trim(directory))
   dt=real(caseid)*7.3; dt_rk=dt/3.; divdamp=real(mod(caseid,4))*2.0
   bc=[0,1,2,3,4,5,6,7,5,6,0]; be=bc; bv=bc
   counts=2; counts(nc+1)=0
   coc=nc+1; coe=nc+1; eoc=ne+1; voe=nv+1; eov=ne+1
   do i=1,nc
     previous=modulo(i-2,nc)+1; following=modulo(i,nc)+1
     coc(:,i)=[previous,following]; coe(:,i)=[i,following]
     eoc(:,i)=[previous,i]; voe(:,i)=[i,following]; eov(:,i)=[previous,i]
   end do
   do i=1,nc+1
     x(i)=6370000.+real(i)*0.3; y(i)=real(i)*0.7; z(i)=real(i)*0.11
     scale_c(i)=1.+real(i)/37.; scale_e(i)=1.+real(i)/43.
     dc(i)=210000.+real(i)*3100.; dv(i)=90000.+real(i)*2700.
     idc(i)=1./dc(i); idv(i)=1./dv(i)
     iac(i)=1./(dc(i)*dv(i)); iat(i)=1./(dc(i)*dv(i)/3.)
     esc(:,i)=[-1.,1.]; esv(:,i)=[1.,-1.]
     do k=1,nl
       rho(k,i)=0.3+real(k+3*i)/71.; theta(k,i)=280.+real(k*3+i)/13.
       ru(k,i)=real(k*7-i*3)/17.; rd(k,i)=rho(k,i)*1.01; td(k,i)=rho(k,i)*theta(k,i)*0.99
       ud(k,i)=ru(k,i)+real(i+k)/211.
       tr(k,i)=real(k-i)/113.; tt(k,i)=real(k+i)/59.; tu(k,i)=real(k*2-i)/79.
       diabatic(k,i)=real(i+k)/31.; base(k,i)=200.; rp(k,i)=rho(k,i)*theta(k,i)-base(k,i)
       do t=1,nt
         q(t,k,i)=real(t*17+k*7+i*3+caseid)/997.; qd(t,k,i)=q(t,k,i)*0.973
       end do
     end do
     do k=1,nl+1
       tw(k,i)=real(k-i)/37.
     end do
   end do
   ! Endpoints are already zero after native recover; boundary helper writes internal levels only.
   tw(1,:)=0.; tw(nl+1,:)=0.
   coe(2,ne)=nc+1
   ! Equal single-precision squared distances: strict '<' must retain the first candidate.
   coc(:,7)=[6,9]
   x(7)=0.; y(7)=0.; z(7)=0.; x(8)=0.; y(8)=0.; z(8)=0.
   x(6)=1.; y(6)=0.0001; z(6)=0.; x(9)=1.; y(9)=0.; z(9)=0.
   ! Additional branch decks retain the same array ABI for direct replay.
   select case(caseid)
   case(17)
     bc=0; be=0; bv=0 ! Empty regional selections.
   case(18)
     bc(:nc)=6; be(:ne)=6; bv(:nv)=6 ! No relaxation neighbors.
   case(19)
     bc(:nc)=5; be(:ne)=5; bv(:nv)=5 ! No specified selections.
   case(20)
     counts=0 ! Searches and cell filter adjacency loops are empty.
   case(21)
     rd=rho; td=rho*theta; ud=ru ! Zero deviations in relaxation filters.
   case(22)
     x(9)=0.5 ! Second candidate wins rather than retaining a rounded tie.
   case(23)
     coc(:,7)=[6,4] ! Outer search rejects a non-relaxation second-hop neighbor.
     q(1,:,:)=0.; qd(1,:,:)=0. ! Dry LBC thermodynamic derivation.
   case(24)
     coc(:,7)=nc+1; coc(:,8)=nc+1 ! Stored garbage neighbors.
     dv=0.125*dc; idv=1./dv ! Select the capped 4*invDc vorticity length.
     iac=1./(dc*dv); iat=1./(dc*dv/3.)
   end select
   smc=0.; sme=0.; smv=0.
   open(10,file=trim(directory)//'/input.bin',access='stream',form='unformatted',status='replace')
   write(10) dt,dt_rk,divdamp,bc,be,bv,counts,coc,coe,eoc,voe,eov,x,y,z,scale_c,scale_e,dc,dv,idc,idv,iac,iat,esc,esv
   write(10) rho,theta,ru,rd,td,ud,tr,tt,tu,tw,diabatic,rp,base,q,qd
   close(10)
   ! LBC numeric pool: persisted one-cell edge, explicit garbage, and all fields.
   lbc_coe=coe; lbc_coe(2,ne)=0
   if (caseid==23) lbc_coe(1,1)=0 ! Missing first as well as second cell.
   zz=1.3; zz(:,nc+1)=0.
   lbc_rho=0.; lbc_rt=0.; lbc_re=0.; lbc_re(:,ne)=0.25; lbc_ru=0.
   old_u=ru*0.991; old_ru=ru*0.983; old_re=0.3; old_w=tw*0.97
   old_theta=theta*0.993; old_rt=td*0.91; old_rho=rho*0.999; old_r=rho*0.99; old_q=q*0.95
   interval=10800.; remaining=interval-dt_rk
   if (caseid==24) remaining=-dt_rk ! Numeric interpolation beyond endpoint.
   do i=1,nc
     density(i)=1.+real(caseid+i)/19.
   end do
   density(nc+1)=0.
   ! Save exact native inputs so the replay does not reproduce initialization formulas.
   open(10,file=trim(directory)//'/lbc-input.bin',access='stream',form='unformatted',status='replace')
   write(10) lbc_coe,zz,lbc_re,old_u,old_ru,old_re,old_w,old_theta,old_rt,old_rho,old_r,old_q,interval,remaining,density
   close(10)
   call derive_lbc(nc,ne,nl,nt,1,lbc_coe,zz,rho,theta,lbc_rho,lbc_rt,lbc_re,lbc_ru,ru,tw,q)
   open(10,file=trim(directory)//'/lbc-derived.bin',access='stream',form='unformatted',status='replace')
   write(10) zz,lbc_rho,lbc_rt,lbc_re,lbc_ru
   close(10)
   call tend_lbc(nc,ne,nl,nt,1,lbc_coe,zz,rho,theta,lbc_rho,lbc_rt,lbc_re,lbc_ru,ru,tw,q,interval,old_u,old_ru,old_re,old_w,old_theta,old_rt,old_rho,old_r,old_q)
   open(10,file=trim(directory)//'/lbc-tendency.bin',access='stream',form='unformatted',status='replace')
   write(10) old_u,old_ru,old_re,old_w,old_theta,old_rt,old_rho,old_r,old_q
   close(10)
   call state_lbc(nt,nl,nc,q,old_q,remaining,at_q)
   open(10,file=trim(directory)//'/lbc-state.bin',access='stream',form='unformatted',status='replace')
   write(10) at_q
   close(10)
   call scaling_lbc(nc,ne,coe,density,mod(caseid,2)==0,scaling_c,scaling_e)
   open(10,file=trim(directory)//'/mesh-scaling.bin',access='stream',form='unformatted',status='replace')
   write(10) scaling_c,scaling_e
   close(10)
   call mpas_atm_setup_bdy_masks(mesh,config)
   call atm_bdy_adjust_dynamics_speczone_tend(tend,mesh,config,nl,ud,td,rd,1,nc,1,ne,1,nc,1,ne)
   open(10,file=trim(directory)//'/spec.bin',access='stream',form='unformatted',status='replace')
   write(10) tr,tt,tu,tw,diabatic,smc,sme,smv,nearest
   close(10)
   call atm_bdy_adjust_dynamics_relaxzone_tend(config,tend,state,diag,mesh,nl,dt,ud,td,rd,1,nc,1,ne,1,nc,1,ne)
   open(10,file=trim(directory)//'/relax.bin',access='stream',form='unformatted',status='replace')
   write(10) tr,tt,tu
   close(10)
   call atm_zero_gradient_w_bdy_work(tw,bc,nearest,nc,1,nc)
   call atm_bdy_reset_speczone_values(state,diag,mesh,nl,td,rd,1,nc,1,nc)
   open(10,file=trim(directory)//'/reset.bin',access='stream',form='unformatted',status='replace')
   write(10) tw,theta,rp
   close(10)
   call atm_bdy_adjust_scalars_work(q,qd,dt,dt_rk,nl,nc,nt,counts,eoc,esc,coe,dv,idc,bc,scale_c,1,nc,1,nc)
   open(10,file=trim(directory)//'/scalar-adjust.bin',access='stream',form='unformatted',status='replace')
   write(10) q
   close(10)
   call atm_bdy_set_scalars_work(qd,q,nl,nc,nt,bc,1,nc,1,nc)
   open(10,file=trim(directory)//'/scalar-set.bin',access='stream',form='unformatted',status='replace')
   write(10) q
   close(10)
   reset_u=ru;reset_ru=ru;drive_ru=ud+0.75
   open(10,file=trim(directory)//'/velocity-input.bin',access='stream',form='unformatted',status='replace')
   write(10) reset_u,reset_ru,ud,drive_ru;close(10)
   call overwrite_native(nl,ne,ne,be,reset_u,ud)
   call overwrite_native(nl,ne,ne,be,reset_ru,drive_ru)
   open(10,file=trim(directory)//'/velocity-reset.bin',access='stream',form='unformatted',status='replace')
   write(10) reset_u,reset_ru;close(10)
   do i=1,size(clamp_raw)
     clamp_raw(i)=clamp_bits(modulo(i+caseid-2,size(clamp_bits))+1)
   end do
   clamp_values=reshape(transfer(clamp_raw,[(0.,i=1,size(clamp_raw))]),shape(clamp_values))
   open(10,file=trim(directory)//'/clamp-input.bin',access='stream',form='unformatted',status='replace')
   write(10) clamp_values;close(10)
   call clamp_native(nt,nl,nc,clamp_values)
   open(10,file=trim(directory)//'/clamp-output.bin',access='stream',form='unformatted',status='replace')
   write(10) clamp_values;close(10)
 end do
 ! Native 3d interpolation on signed subnormal input/output words. Arithmetic
 ! is the unchanged mpas_atm_boundaries.F:631-642 slice in direct_lbc.
 do i=1,size(clamp_raw)
   select case(modulo(i,4))
   case(0)
     clamp_raw(i)=int(z'00000003',int32)
   case(1)
     clamp_raw(i)=int(z'80000003',int32)
   case(2)
     clamp_raw(i)=int(z'007FFFFF',int32)
   case(3)
     clamp_raw(i)=int(z'807FFFFF',int32)
   end select
 end do
 q=reshape(transfer(clamp_raw,[(0.,i=1,size(clamp_raw))]),shape(q))
 qd=q;remaining=0.5
 open(10,file='subnormal-input.bin',access='stream',form='unformatted',status='replace')
 write(10) q,qd,remaining;close(10)
 call state_lbc(nt,nl,nc,q,qd,remaining,at_q)
 open(10,file='subnormal-output.bin',access='stream',form='unformatted',status='replace')
 write(10) at_q;close(10)
end program
