! Real initialized regional columns through pristine mpas_atm_boundaries.F.
! The stream fixture repeats the same initial state at two prescribed times.
! Optional argument 2 supplies published native LBC fields. The fixture
! Optional argument 3 admits the next actual LBC at a 3h interval.
! This whole-module component replay does not prove a forecast.
program real_boundary_fixture
 use mpas_atm_boundaries
 use mpas_derived_types
 use netcdf
 implicit none
 integer,target::nc,ne,nl,ns=2,qvp=1,qcp=2
 integer::nv,me,ncid,j,rows,cols,ierr
 integer,allocatable,target::bc(:),be(:),bv(:),nearest(:),counts(:),coc(:,:),coe(:,:)
 real(RKIND),allocatable,target::smc(:),sme(:),smv(:),x(:),y(:),z(:),zz(:,:)
 real(RKIND),allocatable::at_r(:,:),at_q(:,:),at_s(:,:,:),tend_r(:,:),tend_q(:,:)
 type(mpas_pool_type),target::mesh,state,lbc,structs,configs,file
 type(block_type),target::block
 type(block_type),pointer::bp
 type(dm_info),target::info
 type(dm_info),pointer::ip
 type(MPAS_streamManager_type),target::stream
 type(MPAS_streamManager_type),pointer::sp
 type(mpas_clock_type)::clock
 logical,target::apply=.true.
 character(1024)::input_path,lbc_path,next_lbc_path
 character(4)::prefix
 character(32)::scalar_count,omit_scalar,retrieved_name
 character(8),parameter::scalar_names(12)=[character(8)::"qv","qc","qr","qi","qs","qg","ni","nr","nc","nifa","nwfa","tke"]
 call get_command_argument(1,input_path)
 call get_command_argument(2,lbc_path)
 call get_command_argument(3,next_lbc_path)
 if(len_trim(lbc_path)>0) ns=3
 call get_command_argument(4,scalar_count)
 call get_command_argument(5,omit_scalar)
 if(len_trim(scalar_count)>0) read(scalar_count,*) ns
 if(ns<1.or.ns>12) error stop 'scalar inventory'
 retrieved_name='qc'
 if(ns==1) retrieved_name='qv'
 call check(nf90_open(trim(input_path),nf90_nowrite,ncid))
 call dimension('nCells',nc);call dimension('nEdges',ne)
 call dimension('nVertices',nv);call dimension('nVertLevels',nl);call dimension('maxEdges',me)
 allocate(bc(nc+1),be(ne+1),bv(nv+1),nearest(nc+1),counts(nc+1),coc(me,nc+1),coe(2,ne+1))
 allocate(smc(nc+1),sme(ne+1),smv(nv+1),x(nc+1),y(nc+1),z(nc+1),zz(nl,nc+1))
 allocate(at_r(nl,nc+1),at_q(nl,nc+1),at_s(ns,nl,nc+1),tend_r(nl,nc+1),tend_q(nl,nc+1))
 bc=0;be=0;bv=0;counts=0;coc=nc+1;coe=0;x=0.;y=0.;z=0.;zz=1.
 smc=0.;sme=0.;smv=0.
 call int1('bdyMaskCell',bc(:nc));call int1('bdyMaskEdge',be(:ne));call int1('bdyMaskVertex',bv(:nv))
 call int1('nEdgesOnCell',counts(:nc));call int2('cellsOnCell',coc(:,:nc));call int2('cellsOnEdge',coe(:,:ne))
 ! Missing mask-search neighbors use the framework's padded garbage cell.
 ! This association has bdyMaskCell=0; no native numerical code is changed.
 where(coc==0) coc=nc+1
 ! MPAS-A v8.4.2 framework/mpas_block_creator.F:1490-1496 applies
 ! the same missing-neighbor reindexing to cellsOnEdge before boundary math.
 where(coe==0) coe=nc+1
 call real1('xCell',x(:nc));call real1('yCell',y(:nc));call real1('zCell',z(:nc))
 call real2('zz',zz(:,:nc))
 mesh%nCells=>nc;mesh%nEdges=>ne;mesh%nVertLevels=>nl;mesh%nCellsSolve=>nc
 mesh%bdyMaskCell=>bc;mesh%bdyMaskEdge=>be;mesh%bdyMaskVertex=>bv
 mesh%nearestRelaxationCell=>nearest;mesh%nEdgesOnCell=>counts;mesh%cellsOnCell=>coc;mesh%cellsOnEdge=>coe
 mesh%specZoneMaskCell=>smc;mesh%specZoneMaskEdge=>sme;mesh%specZoneMaskVertex=>smv
 mesh%xCell=>x;mesh%yCell=>y;mesh%zCell=>z;mesh%zz=>zz
 state%num_scalars=>ns;lbc%index_qv=>qvp;lbc%index_qc=>qcp;lbc%num_scalars=>ns
 structs%mesh=>mesh;structs%state=>state;structs%lbc=>lbc
 configs%config_apply_lbcs=>apply;block%structs=>structs;block%configs=>configs
 stream%pool=>lbc;stream%file=>file;bp=>block;ip=>info;sp=>stream
 do j=1,8
   rows=nl;cols=nc
   if(j<=3) cols=ne
   if(j==4) rows=nl+1
   allocate(lbc%fields(j)%current(rows,cols+1),lbc%fields(j)%previous(rows,cols+1))
   allocate(file%fields(j)%current(rows,cols+1))
   lbc%fields(j)%current=0.;lbc%fields(j)%previous=0.;file%fields(j)%current=0.
 end do
 allocate(lbc%scalars(ns,nl,nc+1),lbc%old_scalars(ns,nl,nc+1),file%scalars(ns,nl,nc+1))
 lbc%scalars=0.;lbc%old_scalars=0.;file%scalars=0.
 prefix=''
 if(len_trim(lbc_path)>0) then
   call check(nf90_close(ncid))
   call check(nf90_open(trim(lbc_path),nf90_nowrite,ncid))
   prefix='lbc_'
 end if
 call timed_real2(trim(prefix)//'u',file%fields(1)%current(:,:ne))
 call timed_real2(trim(prefix)//'w',file%fields(4)%current(:,:nc))
 call timed_real2(trim(prefix)//'theta',file%fields(5)%current(:,:nc))
 call timed_real2(trim(prefix)//'rho',file%fields(8)%current(:,:nc))
 do j=1,ns
   call read_scalar(j,trim(prefix)//trim(scalar_names(j)))
 end do
 call check(nf90_close(ncid))
 open(10,file='dimensions.bin',access='stream',form='unformatted',status='replace')
 write(10) nc,ne,nv,nl,me,ns;close(10)
 open(10,file='primary.bin',access='stream',form='unformatted',status='replace')
 write(10) file%fields(1)%current,file%fields(4)%current,file%fields(5)%current,file%fields(8)%current,file%scalars
 close(10)
 open(10,file='mesh.bin',access='stream',form='unformatted',status='replace')
 write(10) bc,be,bv,counts,coc,coe,x,y,z,zz;close(10)
 ! MPAS-A v8.4.2 mpas_atm_boundaries.F:659-753, full unchanged routine.
 call mpas_atm_setup_bdy_masks(mesh,configs)
 open(10,file='masks.bin',access='stream',form='unformatted',status='replace')
 write(10) smc,sme,smv,nearest;close(10)
 call mpas_atm_bdy_checks(ip,bp,sp,ierr)
 if(ierr/=0) error stop 'regional admission'
 ! MPAS-A v8.4.2 mpas_atm_boundaries.F:78-339, full unchanged routine.
 clock%now%ticks=0;stream%read_time='0'
 call mpas_atm_update_bdy_tend(clock,stream,block,.true.,ierr)
 if(ierr/=0) error stop 'first admission'
 call write_pool('first-derived.bin',.false.)
 stream%read_time='10800250000'
 if(len_trim(next_lbc_path)>0) then
   call check(nf90_open(trim(next_lbc_path),nf90_nowrite,ncid))
   call timed_real2('lbc_u',file%fields(1)%current(:,:ne))
   call timed_real2('lbc_w',file%fields(4)%current(:,:nc))
   call timed_real2('lbc_theta',file%fields(5)%current(:,:nc))
   call timed_real2('lbc_rho',file%fields(8)%current(:,:nc))
   do j=1,ns
     call read_scalar(j,'lbc_'//trim(scalar_names(j)))
   end do
   call check(nf90_close(ncid))
   open(10,file='next-primary.bin',access='stream',form='unformatted',status='replace')
   write(10) file%fields(1)%current,file%fields(4)%current,file%fields(5)%current,file%fields(8)%current,file%scalars
   close(10)
   stream%read_time='10800000000'
 end if
 if(len_trim(omit_scalar)>0) file%scalar_present(2)=.false.
 call mpas_atm_update_bdy_tend(clock,stream,block,.false.,ierr)
 if(ierr/=0) error stop 'next admission'
 call write_pool('next-derived.bin',.false.);call write_pool('tendencies.bin',.true.)
 if(len_trim(omit_scalar)>0) then
   ! Synthetic stream-presence branch, retaining genuine input column words.
   ! The second missing admission recycles the preceding tendency buffer.
   stream%read_time='21600000000'
   call mpas_atm_update_bdy_tend(clock,stream,block,.false.,ierr)
   if(ierr/=0) error stop 'third admission'
   call write_pool('third-derived.bin',.false.);call write_pool('tendencies.bin',.true.)
 end if
 clock%now%ticks=7200125000_int64
 ! MPAS-A v8.4.2 mpas_atm_boundaries.F:375-426,462-553,587-642.
 call mpas_atm_get_bdy_tend(clock,block,nl,nc,'rho_zz',0.375_RKIND,tend_r)
 call mpas_atm_get_bdy_tend(clock,block,nl,nc,trim(retrieved_name),0.375_RKIND,tend_q)
 open(10,file='retrieved.bin',access='stream',form='unformatted',status='replace')
 write(10) tend_r,tend_q;close(10)
 call mpas_atm_get_bdy_state(clock,block,nl,nc,'rho_zz',0.375_RKIND,at_r)
 call mpas_atm_get_bdy_state(clock,block,nl,nc,trim(retrieved_name),0.375_RKIND,at_q)
 call mpas_atm_get_bdy_state(clock,block,ns,nl,nc,'scalars',0.375_RKIND,at_s)
 open(10,file='interpolated.bin',access='stream',form='unformatted',status='replace')
 write(10) at_r,at_q,at_s;close(10)
contains
 subroutine read_scalar(index,name)
   integer,intent(in)::index
   character(*),intent(in)::name
   integer::id,status
   status=nf90_inq_varid(ncid,name,id)
   file%scalar_present(index)=status==nf90_noerr
   if(status==nf90_enotvar) return
   call check(status)
   call timed_real2(name,file%scalars(index,:,:nc))
 end subroutine
 subroutine check(status)
   integer,intent(in)::status
   if(status/=nf90_noerr) then
     print *,trim(nf90_strerror(status));error stop 'netcdf'
   end if
 end subroutine
 subroutine dimension(name,out)
   character(*),intent(in)::name
   integer,intent(out)::out
   integer::id
   call check(nf90_inq_dimid(ncid,name,id));call check(nf90_inquire_dimension(ncid,id,len=out))
 end subroutine
 subroutine int1(name,out)
   character(*),intent(in)::name
   integer,intent(out)::out(:)
   integer::id
   call check(nf90_inq_varid(ncid,name,id));call check(nf90_get_var(ncid,id,out))
 end subroutine
 subroutine int2(name,out)
   character(*),intent(in)::name
   integer,intent(out)::out(:,:)
   integer::id
   call check(nf90_inq_varid(ncid,name,id));call check(nf90_get_var(ncid,id,out))
 end subroutine
 subroutine real1(name,out)
   character(*),intent(in)::name
   real(RKIND),intent(out)::out(:)
   integer::id
   call check(nf90_inq_varid(ncid,name,id));call check(nf90_get_var(ncid,id,out))
 end subroutine
 subroutine real2(name,out)
   character(*),intent(in)::name
   real(RKIND),intent(out)::out(:,:)
   integer::id
   call check(nf90_inq_varid(ncid,name,id));call check(nf90_get_var(ncid,id,out))
 end subroutine
 subroutine timed_real2(name,out)
   character(*),intent(in)::name
   real(RKIND),intent(out)::out(:,:)
   real(RKIND)::slab(size(out,1),size(out,2),1)
   integer::id
   call check(nf90_inq_varid(ncid,name,id));call check(nf90_get_var(ncid,id,slab))
   out=slab(:,:,1)
 end subroutine
 subroutine write_pool(filename,previous)
   character(*),intent(in)::filename
   logical,intent(in)::previous
   open(10,file=filename,access='stream',form='unformatted',status='replace')
   do j=1,8
     if(previous) then
       write(10) lbc%fields(j)%previous
     else
       write(10) lbc%fields(j)%current
     end if
   end do
   if(previous) then
     write(10) lbc%old_scalars
   else
     write(10) lbc%scalars
   end if
   close(10)
 end subroutine
end program
