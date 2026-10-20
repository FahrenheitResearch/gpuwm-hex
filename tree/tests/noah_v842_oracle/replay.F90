subroutine replay_noah(x,ix,n) bind(C,name="replay_noah")
use iso_c_binding
use module_sf_noahlsm

implicit none
integer(c_int),value::n
real(c_float)::x(110,n)
integer(c_int)::ix(14,n)
integer::c
logical::LOCAL_arg,UA_PHYS_arg,RDLAI2D_arg,USEMONALB_arg
do c=1,n
LOCAL_arg = ix(5,c) /= 0
UA_PHYS_arg = ix(9,c) /= 0
RDLAI2D_arg = ix(10,c) /= 0
USEMONALB_arg = ix(11,c) /= 0
call SFLX(&
ix(1,c), ix(2,c), x(1,c), ix(3,c), &
x(2,c), x(3,c), ix(4,c), x(4:7,c), &
LOCAL_arg, LUTYPE, SLTYPE, x(8,c), &
x(9,c), x(10,c), x(11,c), x(12,c), &
x(13,c), x(14,c), x(15,c), x(16,c), &
x(17,c), x(18,c), x(19,c), x(20,c), &
x(21,c), ix(6,c), ix(7,c), ix(8,c), &
x(22,c), x(23,c), x(24,c), x(25,c), &
x(26,c), x(27,c), x(28,c), x(29,c), &
x(30,c), x(31,c), x(32,c), x(33,c), &
x(34:37,c), x(38:41,c), x(42:45,c), x(46,c), &
x(47,c), x(48,c), x(49,c), x(50,c), &
x(51,c), x(52,c), x(53,c), x(54,c), &
x(55,c), x(56,c), x(57:60,c), x(61,c), &
x(62,c), x(63,c), x(64,c), x(65,c), &
x(66,c), x(67,c), x(68,c), x(69,c), &
x(70,c), x(71,c), x(72,c), x(73,c), &
x(74,c), UA_PHYS_arg, x(75,c), x(76,c), &
x(77,c), x(78,c), x(79,c), x(80,c), &
x(81,c), x(82,c), x(83,c), x(84,c), &
x(85,c), x(86,c), x(87,c), x(88,c), &
x(89,c), x(90,c), x(91:94,c), RDLAI2D_arg, &
USEMONALB_arg, x(95,c), x(96,c), x(97,c), &
x(98,c), x(99,c), x(100,c), ix(12,c), &
x(101,c), x(102,c), x(103,c), ix(13,c), &
x(104,c), x(105,c), x(106,c), x(107,c), &
x(108,c), ix(14,c), x(109,c), x(110,c))
ix(5,c) = merge(1,0,LOCAL_arg)
ix(9,c) = merge(1,0,UA_PHYS_arg)
ix(10,c) = merge(1,0,RDLAI2D_arg)
ix(11,c) = merge(1,0,USEMONALB_arg)
enddo
end subroutine

subroutine replay_glacial(x,ix,n) bind(C,name="replay_glacial")
use iso_c_binding
use module_sf_noahlsm
use module_sf_noahlsm_glacial_only, only: SFLX_GLACIAL
implicit none
integer(c_int),value::n
real(c_float)::x(49,n)
integer(c_int)::ix(4,n)
integer::c

do c=1,n

call SFLX_GLACIAL(&
ix(1,c), ix(2,c), ix(3,c), x(1,c), &
x(2,c), x(3,c), ix(4,c), x(4:7,c), &
x(8,c), x(9,c), x(10,c), x(11,c), &
x(12,c), x(13,c), x(14,c), x(15,c), &
x(16,c), x(17,c), x(18,c), x(19,c), &
x(20,c), x(21,c), x(22,c), x(23,c), &
x(24,c), x(25:28,c), x(29,c), x(30,c), &
x(31,c), x(32,c), x(33,c), x(34,c), &
x(35,c), x(36,c), x(37,c), x(38,c), &
x(39,c), x(40,c), x(41,c), x(42,c), &
x(43,c), x(44,c), x(45,c), x(46,c), &
x(47,c), x(48,c), x(49,c))

enddo
end subroutine

! MPAS v8.4.2 module_sf_noahlsm.F:4526-4760; module's unused MYJ surface path.
subroutine replay_sfcdif(x,ix,n) bind(C,name="replay_sfcdif")
use iso_c_binding
use module_sf_noahlsm,only:SFCDIF_off
integer(c_int),value::n
real(c_float)::x(8,n)
integer(c_int)::ix(*)
integer::c
do c=1,n
call SFCDIF_off(x(1,c),x(2,c),x(3,c),x(4,c),x(5,c),x(6,c),x(7,c),x(8,c))
enddo
end subroutine
