! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
!=================================================================================================================
 module mpas_atmphys_lsm_shared
 use mpas_kind_types


 implicit none
 private
 public:: correct_tsk_over_seaice,seaice_component_tsk


 contains


!=================================================================================================================
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

!=================================================================================================================
 subroutine seaice_component_tsk(ims,ime,jms,jme,its,ite,jts,jte,xice_thresh,xice,sst,tsk,tsk_sea,tsk_ice)
!=================================================================================================================

!RP3 (correct mode): over fractional sea-ice cells the stored skin temperature is the blend
!xice*tsk_ice + (1-xice)*tsk_sea. this recovers its components the way WRF does (module_surface_driver.F,
!subroutine get_local_ice_tsk, tice2tsk_if2cold=.false.): the open-water component is the sea-surface
!temperature limited to 271.4 K, the value the surface layer uses for the open-water fluxes, and the ice
!component is the algebraic inverse of the blend. blending the recovered ice temperature with the same
!open-water temperature returns the stored skin temperature, so an unchanged two-component state stays
!unchanged. correct_tsk_over_seaice instead sets both components to the blended value.

!input arguments:
 integer,intent(in):: ims,ime,its,ite,jms,jme,jts,jte
 real(kind=RKIND),intent(in):: xice_thresh
 real(kind=RKIND),intent(in),dimension(ims:ime,jms:jme):: sst,tsk,xice

!inout arguments:
 real(kind=RKIND),intent(inout),dimension(ims:ime,jms:jme):: tsk_sea,tsk_ice

!local variables:
 integer:: i,j

!-----------------------------------------------------------------------------------------------------------------

 do j = jts,jte
 do i = its,ite
    tsk_sea(i,j) = tsk(i,j)
    tsk_ice(i,j) = tsk(i,j)

    if(xice(i,j).ge.xice_thresh .and. xice(i,j).le.1._RKIND) then
       tsk_sea(i,j) = max(sst(i,j),271.4_RKIND)
       tsk_ice(i,j) = (tsk(i,j) - (1._RKIND-xice(i,j))*tsk_sea(i,j)) / xice(i,j)
       !lower limit of the recovered ice temperature, as in WRF get_local_ice_tsk:
       tsk_ice(i,j) = max(tsk_ice(i,j),221.4_RKIND)

       !over sea-ice grid cells, avoids unphysically too cold sea-ice temperatures for grid cells
       !with small sea-ice fractions (as in correct_tsk_over_seaice and WRF get_local_ice_tsk):
       if(xice(i,j).lt.0.2_RKIND .and. tsk(i,j).lt.253.15_RKIND) tsk_ice(i,j) = 253.15_RKIND
       if(xice(i,j).lt.0.1_RKIND .and. tsk(i,j).lt.263.15_RKIND) tsk_ice(i,j) = 263.15_RKIND
    endif
 enddo
 enddo

 end subroutine seaice_component_tsk

!=================================================================================================================
 end module mpas_atmphys_lsm_shared
!=================================================================================================================




