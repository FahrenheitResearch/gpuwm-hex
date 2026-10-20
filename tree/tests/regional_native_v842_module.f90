! Exercise every public routine of pristine mpas_atm_boundaries.F.
program whole_boundary_fixture
 use mpas_atm_boundaries
 use mpas_derived_types
 implicit none
 integer,parameter::nc=4,ne=4,nl=3,ns=2
 integer,target::ncp=nc,nep=ne,nlp=nl,nsp=ns,qvp=1,qcp=2
 type(mpas_pool_type),target::mesh,state,lbc,structs,configs,file
 type(block_type),target::block
 type(block_type),pointer::bp
 type(dm_info),target::info
 type(dm_info),pointer::ip
 type(MPAS_streamManager_type),target::stream
 type(MPAS_streamManager_type),pointer::sp
 type(mpas_clock_type)::clock
 logical,target::apply=.true.
 integer,target::bc(nc+1),be(ne+1),bv(ne+1),nearest(nc+1),counts(nc+1)
 integer,target::coc(2,nc+1),coe(2,ne+1)
 real(RKIND),target::smc(nc+1),sme(ne+1),smv(ne+1),x(nc+1),y(nc+1),z(nc+1),zz(nl,nc+1)
 real(RKIND)::tend_r(nl,nc+1),tend_q(nl,nc+1),at_r(nl,nc+1),at_q(nl,nc+1),at_s(ns,nl,nc+1)
 integer::caseid,i,j,k,rows,ierr,status(9)
 character(32)::directory
 mesh%nCells=>ncp;mesh%nEdges=>nep;mesh%nVertLevels=>nlp;mesh%nCellsSolve=>ncp
 mesh%bdyMaskCell=>bc;mesh%bdyMaskEdge=>be;mesh%bdyMaskVertex=>bv
 mesh%nearestRelaxationCell=>nearest;mesh%nEdgesOnCell=>counts;mesh%cellsOnCell=>coc;mesh%cellsOnEdge=>coe
 mesh%specZoneMaskCell=>smc;mesh%specZoneMaskEdge=>sme;mesh%specZoneMaskVertex=>smv
 mesh%xCell=>x;mesh%yCell=>y;mesh%zCell=>z;mesh%zz=>zz
 state%num_scalars=>nsp;lbc%index_qv=>qvp;lbc%index_qc=>qcp;lbc%num_scalars=>nsp
 structs%mesh=>mesh;structs%state=>state;structs%lbc=>lbc
 configs%config_apply_lbcs=>apply;block%structs=>structs;block%configs=>configs
 stream%pool=>lbc;stream%file=>file;bp=>block;ip=>info;sp=>stream
 do j=1,8
   rows=nl
   if(j==4) rows=nl+1
   allocate(lbc%fields(j)%current(rows,nc+1),lbc%fields(j)%previous(rows,nc+1))
   allocate(file%fields(j)%current(rows,nc+1))
 end do
 allocate(lbc%scalars(ns,nl,nc+1),lbc%old_scalars(ns,nl,nc+1),file%scalars(ns,nl,nc+1))
 do caseid=1,6
   write(directory,'("whole-",i2.2)') caseid
   call execute_command_line('mkdir -p '//trim(directory))
   bc=[0,5,6,7,0];be=bc;bv=bc;counts=2;counts(nc+1)=0
   do i=1,nc
     coc(:,i)=[modulo(i-2,nc)+1,modulo(i,nc)+1];coe(:,i)=[i,modulo(i,nc)+1]
   end do
   coc(:,nc+1)=nc+1;coe(:,ne+1)=nc+1;coe(2,ne)=nc+1
   if(caseid>=4) coe(1,1)=nc+1
   if(caseid==6) counts=0
   do i=1,nc+1
     x(i)=real(i);y(i)=real(i)/7.;z(i)=0.
   end do
   smc=0.;sme=0.;smv=0.;zz=1.3;zz(:,nc+1)=0.
   do j=1,8
     lbc%fields(j)%current=0.;lbc%fields(j)%previous=-7.
   end do
   lbc%fields(3)%current(:,ne)=0.25;lbc%fields(3)%current(:,1)=0.125
   lbc%scalars=0.;lbc%old_scalars=-7.
   call set_file(1)
   call write_input('first-input.bin')
   clock%now%ticks=0;stream%read_time='0';stream%read_error=1
   call mpas_atm_update_bdy_tend(clock,stream,block,.true.,ierr);status(1)=ierr
   stream%read_error=0
   call mpas_atm_update_bdy_tend(clock,stream,block,.true.,ierr);status(2)=ierr
   status(3)=stream%last_whence
   call write_pool('first-derived.bin')
   call set_file(2)
   call write_input('next-input.bin')
   stream%read_time='10800250000'
   call mpas_atm_update_bdy_tend(clock,stream,block,.false.,ierr);status(4)=ierr
   status(5)=stream%last_whence
   call write_pool('next-derived.bin')
   open(10,file=trim(directory)//'/tendencies.bin',access='stream',form='unformatted',status='replace')
   do j=1,8
     write(10) lbc%fields(j)%previous
   end do
   write(10) lbc%old_scalars;close(10)
   clock%now%ticks=7200125000_int64
   call mpas_atm_get_bdy_tend(clock,block,nl,nc,'rho_zz',0.375_RKIND,tend_r)
   call mpas_atm_get_bdy_tend(clock,block,nl,nc,'qc',0.375_RKIND,tend_q)
   open(10,file=trim(directory)//'/retrieved.bin',access='stream',form='unformatted',status='replace')
   write(10) tend_r,tend_q;close(10)
   call mpas_atm_get_bdy_state(clock,block,nl,nc,'rho_zz',0.375_RKIND,at_r)
   call mpas_atm_get_bdy_state(clock,block,nl,nc,'qc',0.375_RKIND,at_q)
   call mpas_atm_get_bdy_state(clock,block,ns,nl,nc,'scalars',0.375_RKIND,at_s)
   open(10,file=trim(directory)//'/interpolated.bin',access='stream',form='unformatted',status='replace')
   write(10) at_r,at_q,at_s;close(10)
   call mpas_atm_setup_bdy_masks(mesh,configs)
   open(10,file=trim(directory)//'/mesh-input.bin',access='stream',form='unformatted',status='replace')
   write(10) bc,be,bv,counts,coc,x,y,z;close(10)
   open(10,file=trim(directory)//'/masks.bin',access='stream',form='unformatted',status='replace')
   write(10) smc,sme,smv,nearest;close(10)
   stream%read_error=1
   call mpas_atm_update_bdy_tend(clock,stream,block,.false.,ierr);status(6)=ierr
   stream%read_error=0
   apply=.true.;stream%property_error=0;stream%input_interval='03:00:00'
   call mpas_atm_bdy_checks(ip,bp,sp,ierr);status(7)=ierr
   stream%input_interval='none'
   call mpas_atm_bdy_checks(ip,bp,sp,ierr);status(8)=ierr
   stream%property_error=1
   call mpas_atm_bdy_checks(ip,bp,sp,ierr);status(9)=ierr
   open(10,file=trim(directory)//'/status.bin',access='stream',form='unformatted',status='replace')
   write(10) status
   stream%property_error=0;stream%input_interval='03:00:00';apply=.false.
   call mpas_atm_bdy_checks(ip,bp,sp,ierr);write(10) ierr
   bc=0
   call mpas_atm_bdy_checks(ip,bp,sp,ierr);write(10) ierr
   apply=.true.
   call mpas_atm_bdy_checks(ip,bp,sp,ierr);write(10) ierr
   close(10)
 end do
contains
 subroutine set_file(phase)
   integer,intent(in)::phase
   do j=1,8
     file%fields(j)%current=0.
     do i=1,nc
       do k=1,size(file%fields(j)%current,1)
         file%fields(j)%current(k,i)=real(j*13+i*7+k+caseid+phase)/31.
       end do
     end do
   end do
   file%fields(5)%current(:,:nc)=file%fields(5)%current(:,:nc)+280.
   file%fields(8)%current(:,:nc)=file%fields(8)%current(:,:nc)/5.
   file%scalars=0.
   do j=1,ns
     do i=1,nc
       do k=1,nl
         file%scalars(j,k,i)=real(j*9+i+k+phase+caseid)/1713.
       end do
     end do
   end do
   if(caseid==5) file%scalars(1,:,:)=0.
 end subroutine
 subroutine write_input(filename)
   character(*),intent(in)::filename
   open(10,file=trim(directory)//'/'//filename,access='stream',form='unformatted',status='replace')
   write(10) coe,zz,lbc%fields(3)%current,file%fields(1)%current,file%fields(4)%current,file%fields(5)%current,file%fields(8)%current,file%scalars
   close(10)
 end subroutine
 subroutine write_pool(filename)
   character(*),intent(in)::filename
   open(10,file=trim(directory)//'/'//filename,access='stream',form='unformatted',status='replace')
   do j=1,8
     write(10) lbc%fields(j)%current
   end do
   write(10) lbc%scalars;close(10)
 end subroutine
end program
