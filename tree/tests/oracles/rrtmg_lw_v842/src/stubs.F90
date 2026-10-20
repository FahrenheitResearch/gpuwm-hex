! Single-rank framework stubs; no scheme arithmetic is replaced.
module mpas_derived_types
 type dm_info
  integer :: my_proc_id=0
 end type
end module
module mpas_dmpar
 use mpas_derived_types
 integer,parameter :: IO_NODE=0
contains
 subroutine mpas_dmpar_bcast_int(dminfo,a)
 type(dm_info),intent(in)::dminfo
 integer::a
 end subroutine
 subroutine mpas_dmpar_bcast_reals(dminfo,n,a)
 type(dm_info),intent(in)::dminfo
 integer,intent(in)::n
 real::a(*)
 end subroutine
end module
module mpas_atmphys_constants
 use mpas_constants,only:cp,gravity
end module
module mpas_atmphys_utilities
contains
 subroutine physics_error_fatal(m)
 character(*),intent(in)::m
 print *,m
 error stop 1
 end subroutine
end module
module mpas_log
contains
 subroutine mpas_log_write(m)
 character(*),intent(in)::m
 print *,m
 end subroutine
end module
module mpas_io_units
contains
 subroutine mpas_new_unit(u,unformatted)
 integer,intent(out)::u
 logical,intent(in),optional::unformatted
 u=20
 end subroutine
 subroutine mpas_release_unit(u)
 integer,intent(in)::u
 end subroutine
end module
