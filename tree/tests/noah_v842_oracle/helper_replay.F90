subroutine helper_noah_fac2mit(x,ix,n) bind(C,name="helper_noah_fac2mit")
use iso_c_binding
use module_sf_noahlsm, only: FAC2MIT
implicit none

integer(c_int),value :: n
real(c_float) :: x(2,n)
integer(c_int) :: ix(1,n)
integer :: c

do c=1,n

call FAC2MIT(x(1,c),x(2,c))

enddo
end subroutine

subroutine helper_noah_frh2o(x,ix,n) bind(C,name="helper_noah_frh2o")
use iso_c_binding
use module_sf_noahlsm, only: FRH2O
implicit none

integer(c_int),value :: n
real(c_float) :: x(7,n)
integer(c_int) :: ix(1,n)
integer :: c

do c=1,n

call FRH2O(x(1,c),x(2,c),x(3,c),x(4,c), &
x(5,c),x(6,c),x(7,c))

enddo
end subroutine

subroutine helper_gl_snow_new(x,ix,n) bind(C,name="helper_gl_snow_new")
use iso_c_binding

implicit none
external :: SNOW_NEW
integer(c_int),value :: n
real(c_float) :: x(4,n)
integer(c_int) :: ix(1,n)
integer :: c

do c=1,n

call SNOW_NEW(x(1,c),x(2,c),x(3,c),x(4,c))

enddo
end subroutine

subroutine helper_noah_evapo(x,ix,n) bind(C,name="helper_noah_evapo")
use iso_c_binding
use module_sf_noahlsm, only: EVAPO
implicit none

integer(c_int),value :: n
real(c_float) :: x(43,n)
integer(c_int) :: ix(2,n)
integer :: c

do c=1,n

call EVAPO(x(1,c),x(2:5,c),ix(1,c),x(6,c), &
x(7,c),x(8,c),x(9:12,c),x(13:16,c), &
x(17,c),x(18,c),x(19,c),x(20,c), &
x(21,c),x(22,c),x(23,c),x(24,c), &
x(25,c),x(26,c),x(27,c),x(28,c), &
x(29,c),x(30:33,c),x(34,c),x(35,c), &
x(36,c),ix(2,c),x(37:40,c),x(41,c), &
x(42,c),x(43,c))

enddo
end subroutine

subroutine helper_noah_nopac(x,ix,n) bind(C,name="helper_noah_nopac")
use iso_c_binding
use module_sf_noahlsm, only: NOPAC
implicit none

integer(c_int),value :: n
real(c_float) :: x(81,n)
integer(c_int) :: ix(7,n)
integer :: c

do c=1,n

call NOPAC(x(1,c),x(2,c),x(3,c),x(4:7,c), &
x(8,c),x(9,c),x(10,c),x(11,c), &
x(12,c),x(13,c),ix(1,c),x(14,c), &
x(15,c),x(16,c),x(17,c),x(18,c), &
x(19,c),x(20,c),x(21,c),x(22,c), &
x(23,c),x(24,c),x(25,c),x(26:29,c), &
x(30,c),x(31,c),x(32,c),x(33,c), &
x(34,c),x(35,c),x(36:39,c),x(40,c), &
x(41,c),x(42,c),x(43,c),x(44:47,c), &
x(48,c),x(49,c),x(50,c),x(51,c), &
x(52,c),x(53,c),x(54,c),x(55,c), &
x(56,c),x(57:60,c),x(61,c),ix(2,c), &
x(62:65,c),x(66,c),x(67,c),x(68,c), &
x(69,c),x(70,c),x(71,c),x(72,c), &
x(73,c),ix(3,c),ix(4,c),x(74,c), &
x(75,c),x(76,c),ix(5,c),ix(6,c), &
x(77,c),x(78,c),x(79,c),ix(7,c), &
x(80,c),x(81,c))

enddo
end subroutine

subroutine helper_noah_snopac(x,ix,n) bind(C,name="helper_noah_snopac")
use iso_c_binding
use module_sf_noahlsm, only: SNOPAC
implicit none

integer(c_int),value :: n
real(c_float) :: x(93,n)
integer(c_int) :: ix(9,n)
integer :: c
logical :: SNOWNG_arg,UA_PHYS_arg
do c=1,n
SNOWNG_arg=ix(1,c)/=0
UA_PHYS_arg=ix(6,c)/=0
call SNOPAC(x(1,c),x(2,c),x(3,c),x(4,c), &
SNOWNG_arg,x(5:8,c),x(9,c),x(10,c), &
x(11,c),x(12,c),x(13,c),x(14,c), &
ix(2,c),x(15,c),x(16,c),x(17,c), &
x(18,c),x(19,c),x(20,c),x(21,c), &
x(22,c),x(23,c),x(24,c),x(25,c), &
x(26:29,c),x(30,c),x(31,c),x(32,c), &
x(33,c),x(34,c),x(35,c),x(36,c), &
x(37,c),x(38,c),x(39,c),x(40,c), &
x(41:44,c),x(45,c),x(46,c),x(47,c), &
x(48,c),x(49:52,c),x(53,c),x(54,c), &
x(55,c),x(56,c),x(57,c),x(58,c), &
x(59,c),x(60,c),x(61,c),x(62,c), &
x(63:66,c),x(67,c),ix(3,c),x(68,c), &
x(69:72,c),x(73,c),x(74,c),x(75,c), &
x(76,c),x(77,c),x(78,c),x(79,c), &
x(80,c),x(81,c),x(82,c),x(83,c), &
x(84,c),x(85,c),x(86,c),ix(4,c), &
ix(5,c),x(87,c),x(88,c),UA_PHYS_arg, &
x(89,c),x(90,c),x(91,c),ix(7,c), &
ix(8,c),x(92,c),ix(9,c),x(93,c))
ix(1,c)=merge(1,0,SNOWNG_arg)
ix(6,c)=merge(1,0,UA_PHYS_arg)
enddo
end subroutine

subroutine helper_noah_penman(x,ix,n) bind(C,name="helper_noah_penman")
use iso_c_binding
use module_sf_noahlsm, only: PENMAN
implicit none

integer(c_int),value :: n
real(c_float) :: x(29,n)
integer(c_int) :: ix(3,n)
integer :: c
logical :: SNOWNG_arg,FRZGRA_arg,UA_PHYS_arg
do c=1,n
SNOWNG_arg=ix(1,c)/=0
FRZGRA_arg=ix(2,c)/=0
UA_PHYS_arg=ix(3,c)/=0
call PENMAN(x(1,c),x(2,c),x(3,c),x(4,c), &
x(5,c),x(6,c),x(7,c),x(8,c), &
x(9,c),x(10,c),x(11,c),x(12,c), &
x(13,c),x(14,c),x(15,c),SNOWNG_arg, &
FRZGRA_arg,x(16,c),x(17,c),x(18,c), &
x(19,c),x(20,c),x(21,c),x(22,c), &
x(23,c),x(24,c),x(25,c),x(26,c), &
x(27,c),x(28,c),x(29,c),UA_PHYS_arg)
ix(1,c)=merge(1,0,SNOWNG_arg)
ix(2,c)=merge(1,0,FRZGRA_arg)
ix(3,c)=merge(1,0,UA_PHYS_arg)
enddo
end subroutine

subroutine helper_noah_tdfcnd(x,ix,n) bind(C,name="helper_noah_tdfcnd")
use iso_c_binding
use module_sf_noahlsm, only: TDFCND
implicit none

integer(c_int),value :: n
real(c_float) :: x(7,n)
integer(c_int) :: ix(2,n)
integer :: c

do c=1,n

call TDFCND(x(1,c),x(2,c),x(3,c),x(4,c), &
x(5,c),x(6,c),x(7,c),ix(1,c), &
ix(2,c))

enddo
end subroutine

subroutine helper_noah_alcalc(x,ix,n) bind(C,name="helper_noah_alcalc")
use iso_c_binding
use module_sf_noahlsm, only: ALCALC
implicit none

integer(c_int),value :: n
real(c_float) :: x(12,n)
integer(c_int) :: ix(1,n)
integer :: c
logical :: SNOWNG_arg
do c=1,n
SNOWNG_arg=ix(1,c)/=0
call ALCALC(x(1,c),x(2,c),x(3,c),x(4,c), &
x(5,c),x(6,c),x(7,c),x(8,c), &
x(9,c),x(10,c),SNOWNG_arg,x(11,c), &
x(12,c))
ix(1,c)=merge(1,0,SNOWNG_arg)
enddo
end subroutine

subroutine helper_gl_alcalc(x,ix,n) bind(C,name="helper_gl_alcalc")
use iso_c_binding

implicit none
external :: ALCALC
integer(c_int),value :: n
real(c_float) :: x(8,n)
integer(c_int) :: ix(1,n)
integer :: c
logical :: SNOWNG_arg
do c=1,n
SNOWNG_arg=ix(1,c)/=0
call ALCALC(x(1,c),x(2,c),x(3,c),x(4,c), &
x(5,c),x(6,c),x(7,c),SNOWNG_arg, &
x(8,c))
ix(1,c)=merge(1,0,SNOWNG_arg)
enddo
end subroutine

subroutine helper_noah_devap_hydro(x,ix,n) bind(C,name="helper_noah_devap_hydro")
use iso_c_binding
use module_sf_noahlsm, only: DEVAP_HYDRO
implicit none

integer(c_int),value :: n
real(c_float) :: x(16,n)
integer(c_int) :: ix(1,n)
integer :: c

do c=1,n

call DEVAP_HYDRO(x(1,c),x(2,c),x(3,c),x(4,c), &
x(5,c),x(6,c),x(7,c),x(8,c), &
x(9,c),x(10,c),x(11,c),x(12,c), &
x(13,c),x(14,c),x(15,c),x(16,c))

enddo
end subroutine

subroutine helper_noah_snow_new(x,ix,n) bind(C,name="helper_noah_snow_new")
use iso_c_binding
use module_sf_noahlsm, only: SNOW_NEW
implicit none

integer(c_int),value :: n
real(c_float) :: x(4,n)
integer(c_int) :: ix(1,n)
integer :: c

do c=1,n

call SNOW_NEW(x(1,c),x(2,c),x(3,c),x(4,c))

enddo
end subroutine
