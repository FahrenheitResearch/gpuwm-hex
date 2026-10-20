! Framework-only stubs. Numerical modules are copied pristine from MPAS v8.4.2.
module mpas_atmphys_constants
implicit none
real,parameter :: cp=7.0*287.0/2.0, r_d=287.0, xlf=3.50e5, xlv=2.50e6
real,parameter :: rho_w=1000.0, stbolt=5.67051e-8, karman=0.4, gravity=9.80616
end module
module mpas_atmphys_utilities
contains
subroutine physics_error_fatal(message)
character(len=*),intent(in)::message
write(*,*) trim(message)
error stop 1
end subroutine
end module
module module_sf_urban
real::oasis=1.0
integer::iri_scheme=0
external urban
end module
module module_sf_bep
external bep
end module
module module_sf_bep_bem
external bep_bem
end module
module mpas_atmphys_date_time
contains
subroutine cal_mon_day(julian,julyr,jmonth,jday)
integer::julian,julyr,jmonth,jday
error stop 'MPAS Noah driver disables urban calendar path'
end subroutine
end module
subroutine urban()
error stop 'MPAS disables urban physics'
end subroutine
subroutine bep()
error stop 'MPAS disables BEP'
end subroutine
subroutine bep_bem()
error stop 'MPAS disables BEP-BEM'
end subroutine
subroutine wrf_message(message)
character(len=*)::message
write(*,*)message
end subroutine
subroutine wrf_error_fatal(message)
character(len=*)::message
error stop 'WRF branch outside MPAS Noah driver'
end subroutine
module mpas_dmpar
implicit none
integer,parameter::IO_NODE=0
type dm_info
integer::my_proc_id=0
end type
contains
subroutine mpas_dmpar_bcast_char(dminfo,a)
type(dm_info)::dminfo
character(len=*)::a
end subroutine
subroutine mpas_dmpar_bcast_int(dminfo,a)
type(dm_info)::dminfo
integer::a
end subroutine
subroutine mpas_dmpar_bcast_ints(dminfo,n,a)
type(dm_info)::dminfo
integer::n,a(n)
end subroutine
subroutine mpas_dmpar_bcast_real(dminfo,a)
type(dm_info)::dminfo
real::a
end subroutine
subroutine mpas_dmpar_bcast_reals(dminfo,n,a)
type(dm_info)::dminfo
integer::n
real::a(n)
end subroutine
end module
module mpas_io_units
contains
subroutine mpas_new_unit(n)
integer::n
n=27
end subroutine
subroutine mpas_release_unit(n)
integer::n
end subroutine
end module
subroutine physics_message(message)
character(len=*)::message
end subroutine
subroutine mpas_log_write(message)
character(len=*)::message
end subroutine
