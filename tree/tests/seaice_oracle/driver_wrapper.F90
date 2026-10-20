! Exact arithmetic slices from MPAS mpas_atmphys_driver_seaice.F:160-458.
module seaice_driver_oracle
use iso_c_binding
implicit none
integer,parameter::RKIND=c_float
contains
 subroutine correct_tsk_over_seaice(ims,ime,jms,jme,its,ite,jts,jte,xice_thresh,xice,tsk,tsk_sea,tsk_ice)
!=================================================================================================================

!input arguments:
 integer,intent(in):: ims,ime,its,ite,jms,jme,jts,jte
 real(kind=RKIND),intent(in):: xice_thresh
 real(kind=RKIND),intent(in),dimension(ims:ime,jms:jme):: tsk,xice

!inout arguments:
 real(kind=RKIND),intent(inout),dimension(ims:ime,jms:jme):: tsk_sea,tsk_ice

!local variables:
 integer:: i,j

!-----------------------------------------------------------------------------------------------------------------

!initialize the local sea-surface temperature and local sea-ice temperature to the local surface
!temperature:
 do j = jts,jte
 do i = its,ite
    tsk_sea(i,j) = tsk(i,j)
    tsk_ice(i,j) = tsk(i,j)

    if(xice(i,j).ge.xice_thresh .and. xice(i,j).le.1._RKIND) then
       !over sea-ice grid cells, limit sea-surface temperatures to temperatures warmer than 271.4:
       tsk_sea(i,j) = max(tsk_sea(i,j),271.4_RKIND)

       !over sea-ice grid cells, avoids unphysically too cold sea-ice temperatures for grid cells
       !with small sea-ice fractions:
       if(xice(i,j).lt.0.2_RKIND .and. tsk_ice(i,j).lt.253.15_RKIND) tsk_ice(i,j) = 253.15_RKIND
       if(xice(i,j).lt.0.1_RKIND .and. tsk_ice(i,j).lt.263.15_RKIND) tsk_ice(i,j) = 263.15_RKIND
    endif
 enddo
 enddo

 end subroutine correct_tsk_over_seaice
subroutine seaice_from_oracle(n,a,b,ro,microp,convection) bind(C)
integer(c_int),value::n,microp,convection
real(c_float),intent(inout)::a(n,53),b(n,53)
real(c_float),intent(in)::ro(6)
real(c_float)::tsk_ice(n,1),xice_threshold,seaice_albedo_default,seaice_snowdepth_min,seaice_thickness_default
integer::i,j,its,ite,jts,jte,ims,ime,jms,jme
character(3)::microp_scheme,convection_scheme
its=1;ite=n;jts=1;jte=1;ims=1;ime=n;jms=1;jme=1
xice_threshold=ro(2);seaice_albedo_default=ro(3)
seaice_snowdepth_min=ro(6);seaice_thickness_default=ro(4)
microp_scheme='off';convection_scheme='off'
if(microp/=0) microp_scheme='on'
if(convection/=0) convection_scheme='on'
 do j = jts,jte
    do i = its,ite
       !--- in variables:
       a(i,11)      = b(i,11)
       a(i,6)       = b(i,6)
       a(i,7)       = b(i,7)
       a(i,10)    = b(i,10)
       a(i,12)        = b(i,12)
       a(i,13)       = b(i,13)
       a(i,8)    = b(i,39)/(1._RKIND-b(i,16))

       !--- inout variables:
       a(i,18)        = b(i,18)
       a(i,19)      = b(i,19)
       a(i,21)     = b(i,21)
       a(i,22)     = b(i,22)
       a(i,20)       = b(i,20)
       a(i,23)      = b(i,23)
       a(i,25)    = b(i,25)
       a(i,24)    = b(i,24)
       a(i,26) = b(i,26)
       a(i,17)     = seaice_albedo_default
       a(i,37)    = seaice_snowdepth_min
       a(i,38)  = seaice_thickness_default

       !--- output variables:
       a(i,29)       = b(i,29)
       a(i,30)        = b(i,30)
       a(i,31)       = b(i,31)
       a(i,32)       = b(i,32)
       a(i,33)    = b(i,33)
       a(i,34)      = b(i,34)
       a(i,35)      = b(i,35)

       !modify the surface albedo and surface emissivity, and surface temperatures over sea-ice points:
       if(b(i,11).ge.xice_threshold .and. b(i,11).le.1._RKIND) then
          a(i,16) = (b(i,16) - 0.08_RKIND*(1._RKIND-b(i,11))) / b(i,11)
          a(i,15)  = (b(i,15) - 0.98_RKIND*(1._RKIND-b(i,11))) / b(i,11)
       else
          a(i,15)  = b(i,15)
          a(i,16) = b(i,16)
       endif
    enddo

    !calculate sea-surface and sea-ice temperatures over sea-ice grid cells:
    call correct_tsk_over_seaice(ims,ime,jms,jme,its,ite,jts,jte,xice_threshold,a(:,11:11), &
                                 a(:,20:20),a(:,43:43),tsk_ice)
    do i = its,ite
       a(i,20) = tsk_ice(i,j)
    enddo
 enddo

 do j = jts,jte
    do i = its,ite
       a(i,5)     = 0._RKIND
       a(i,9) = 0._RKIND
    enddo
    if(microp_scheme .ne. 'off') then
       do i = its,ite
          a(i,5) = b(i,40)
          a(i,9) = a(i,9) + b(i,41)
       enddo
    endif
    if(convection_scheme .ne. 'off') then
       do i = its,ite
          a(i,9) = a(i,9) + b(i,42)
       enddo
    endif
 enddo

end subroutine
subroutine seaice_to_oracle(n,a,b,ro,microp,convection) bind(C)
integer(c_int),value::n,microp,convection
real(c_float),intent(inout)::a(n,53),b(n,53)
real(c_float),intent(in)::ro(6)
real(c_float)::tsk_ice(n,1),xice_threshold,seaice_albedo_default,seaice_snowdepth_min,seaice_thickness_default
integer::i,j,its,ite,jts,jte,ims,ime,jms,jme
character(3)::microp_scheme,convection_scheme
its=1;ite=n;jts=1;jte=1;ims=1;ime=n;jms=1;jme=1
xice_threshold=ro(2);seaice_albedo_default=ro(3)
seaice_snowdepth_min=ro(6);seaice_thickness_default=ro(4)
microp_scheme='off';convection_scheme='off'
if(microp/=0) microp_scheme='on'
if(convection/=0) convection_scheme='on'
 do j = jts,jte
    do i = its,ite
       if(a(i,11).ge.xice_threshold .and. a(i,11).le.1._RKIND) then
          b(i,44) = a(i,11)*b(i,44) + (1._RKIND-a(i,11))*a(i,45)

          a(i,13)  = a(i,11)*a(i,13)  + (1._RKIND-a(i,11))*a(i,46)
          a(i,35) = a(i,11)*a(i,35) + (1._RKIND-a(i,11))*a(i,47)
          a(i,23) = a(i,11)*a(i,23) + (1._RKIND-a(i,11))*a(i,48)
          a(i,29)  = a(i,11)*a(i,29)  + (1._RKIND-a(i,11))*a(i,49)
          a(i,30)   = a(i,11)*a(i,30)   + (1._RKIND-a(i,11))*a(i,50)
          a(i,31)  = a(i,11)*a(i,31)  + (1._RKIND-a(i,11))*a(i,51)
          a(i,7)  = a(i,11)*a(i,7)  + (1._RKIND-a(i,11))*a(i,52)
          a(i,34) = a(i,11)*a(i,34) + (1._RKIND-a(i,11))*a(i,53)
          a(i,20)  = a(i,11)*a(i,20)  + (1._RKIND-a(i,11))*a(i,43)
          a(i,16) = a(i,11)*a(i,16) + (1._RKIND-a(i,11))*0.08_RKIND
          a(i,15)  = a(i,11)*a(i,15)  + (1._RKIND-a(i,11))*0.98_RKIND
       endif
    enddo
 enddo

!--- update all variables:
 do j = jts,jte
    do i = its,ite
       !--- inout variables:
       b(i,18)        = a(i,18)
       b(i,19)      = a(i,19)
       b(i,21)     = a(i,21)
       b(i,22)     = a(i,22)
       b(i,20)  = a(i,20)
       b(i,25)    = a(i,25)
       b(i,24)    = a(i,24)
       b(i,26) = a(i,26)

       !--- output variables:
       b(i,32)       = a(i,32)
       b(i,33)    = a(i,33)

       b(i,13)  = a(i,13)
       b(i,35) = a(i,35)
       b(i,23) = a(i,23)
       b(i,34) = a(i,34)
       b(i,7)  = a(i,7)
       b(i,29)  = a(i,29)
       b(i,31)  = a(i,31)
       b(i,30)   = a(i,30)
       b(i,16) = a(i,16)
       b(i,15)  = a(i,15)
    enddo
 enddo

end subroutine
end module
