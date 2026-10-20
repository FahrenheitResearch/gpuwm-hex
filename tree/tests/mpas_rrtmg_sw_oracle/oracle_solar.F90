! Pinned MPAS driver_radiation_sw.F:1009-1050, unchanged radconst body.
program oracle_solar
 use mpas_kind_types, only: RKIND
 use mpas_atmphys_constants, only: solcon_0
 implicit none
 real(kind=RKIND) :: julian,degrad,dpd,declin,solcon
 integer :: day,quarter,u,v
 real(kind=RKIND) :: xtime_s,dt_radtsw,xtime_m,radt
 open(newunit=u,file='solar.bin',access='stream',form='unformatted',status='replace')
 open(newunit=v,file='driver_times.bin',access='stream',form='unformatted',status='replace')
 degrad=atan(1.0_RKIND)/45.0_RKIND
 dpd=360.0_RKIND/365.0_RKIND
 do day=0,366
 do quarter=0,3
  julian=real(day,RKIND)+real(quarter,RKIND)*0.25_RKIND
  call radconst(declin,solcon,julian,degrad,dpd)
  write(u)julian,degrad,dpd,declin,solcon
  xtime_s=real(day,RKIND)*734.125+real(quarter,RKIND)*.31
  dt_radtsw=123.75+real(quarter,RKIND)*97.125
  ! Exact MPAS driver_radiation_sw.F:874,899.
  xtime_m=xtime_s/60.
  radt=dt_radtsw/60.
  write(v)xtime_s,dt_radtsw,xtime_m,radt
 enddo
 enddo
 close(u)
 close(v)
contains
 subroutine radconst(declin,solcon,julian,degrad,dpd)
!=================================================================================================================

!input arguments:
!integer,intent(in):: julian
 real(kind=RKIND),intent(in):: julian
 real(kind=RKIND),intent(in):: degrad,dpd

!output arguments:
 real(kind=RKIND),intent(out):: declin,solcon

!local variables:
 real(kind=RKIND):: obecl,sinob,sxlong,arg,decdeg,djul,rjul,eccfac

!-----------------------------------------------------------------------------------------------------------------

   declin=0.
   solcon=0.

!obecl : obliquity = 23.5 degree.
        
   obecl=23.5*degrad
   sinob=sin(obecl)
        
!calculate longitude of the sun from vernal equinox:        

   if(julian.ge.80.)sxlong=dpd*(julian-80.)
   if(julian.lt.80.)sxlong=dpd*(julian+285.)
   sxlong=sxlong*degrad
   arg=sinob*sin(sxlong)
   declin=asin(arg)
   decdeg=declin/degrad

!solar constant eccentricity factor (paltridge and platt 1976)

   djul=julian*360./365.
   rjul=djul*degrad
   eccfac=1.000110+0.034221*cos(rjul)+0.001280*sin(rjul)+0.000719*  &
          cos(2*rjul)+0.000077*sin(2*rjul)
   solcon=solcon_0*eccfac

 end subroutine radconst
end program oracle_solar
