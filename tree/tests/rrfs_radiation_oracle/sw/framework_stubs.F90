! Standalone, one-process framework only. All physics modules are pristine copies.
module mpas_derived_types
 integer,parameter::MPAS_LOG_ERR=1,IO_NODE=0
 type dm_info
 integer::my_proc_id=0
 end type
end module
module mpas_log
contains
 subroutine mpas_log_write(message,messageType,intArgs,realArgs)
 character(len=*),intent(in)::message
 integer,optional,intent(in)::messageType,intArgs(:)
 real,optional,intent(in)::realArgs(:)
 write(*,'(A)') message
 end subroutine
end module
module mpas_atmphys_utilities
contains
 subroutine physics_error_fatal(message)
 character(len=*),intent(in)::message
 write(*,'(A)') message
 error stop 1
 end subroutine
 subroutine physics_message(message)
 character(len=*),intent(in)::message
 write(*,'(A)') message
 end subroutine
end module
module mpas_dmpar
 use mpas_derived_types
contains
 subroutine mpas_dmpar_bcast_int(dminfo,a)
 type(dm_info),intent(in)::dminfo
 integer,intent(inout)::a
 end subroutine
 subroutine mpas_dmpar_bcast_real(dminfo,a)
 type(dm_info),intent(in)::dminfo
 real,intent(inout)::a
 end subroutine
 subroutine mpas_dmpar_bcast_reals(dminfo,n,a)
 type(dm_info),intent(in)::dminfo
 integer,intent(in)::n
 real,intent(inout)::a(..)
 end subroutine
end module
module mpas_io_units
contains
 subroutine mpas_new_unit(u,unformatted)
 integer,intent(out)::u
 logical,optional,intent(in)::unformatted
 u=31
 end subroutine
 subroutine mpas_release_unit(u)
 integer,intent(in)::u
 end subroutine
end module
