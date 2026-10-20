module driver_oracle
use iso_c_binding
use module_sf_sfcdiags
implicit none
integer,parameter::RKIND=4
contains
subroutine oracle_diagnostics(a,c,xice,q2mxy,t2mxy,n,lsm,threshold) bind(c)
integer(c_int),value:: n,lsm
real(c_float),intent(inout):: a(n,48)
real(c_float),intent(in):: c(13),xice(n),q2mxy(n),t2mxy(n)
real(c_float),value:: threshold
real:: rho,cp,P0,R_d,rcp,xice_threshold
integer:: i,nCellsSolve
character(16):: lsm_scheme
nCellsSolve=n
cp=c(1);R_d=c(4);rcp=c(3);P0=c(13);xice_threshold=threshold
lsm_scheme='default'
if(lsm==1) lsm_scheme='sf_noah'
if(lsm==2) lsm_scheme='sf_noahmp'
associate( &
psfc=>a(:,41), &
t2m=>a(:,18), &
th2m=>a(:,17), &
tsk=>a(:,42), &
chs2=>a(:,21), &
cqs2=>a(:,22), &
hfx=>a(:,5), &
qfx=>a(:,6), &
qsfc=>a(:,7), &
q2=>a(:,19), &
xland=>a(:,45))
 sf_select: select case(trim(lsm_scheme))
    case("sf_noah")
       do i = 1,nCellsSolve
          rho = psfc(i)/(R_d*tsk(i))
          if(cqs2(i) .lt. 1.e-5) then
             q2(i) = qsfc(i)
          else
             q2(i) = qsfc(i) - qfx(i)/(rho*cqs2(i))
          endif
          if(chs2(i) .lt. 1.e-5) then
             t2m(i) = tsk(i)
          else
             t2m(i) = tsk(i) - hfx(i)/(rho*cp*chs2(i))
          endif
          th2m(i) = t2m(i)*(P0/psfc(i))**rcp
       enddo

    case("sf_noahmp")
       do i = 1,nCellsSolve
          rho = psfc(i)/(R_d*tsk(i))
          if((xland(i)-1.5 .gt. 0._RKIND) .or. (xland(i)-1.5.le.0._RKIND .and. xice(i).ge.xice_threshold)) then
             if(cqs2(i) .lt. 1.e-5) then
                q2(i) = qsfc(i)
             else
                q2(i) = qsfc(i) - qfx(i)/(rho*cqs2(i))
             endif
             if(chs2(i) .lt. 1.e-5) then
                t2m(i) = tsk(i)
             else
                t2m(i) = tsk(i) - hfx(i)/(rho*cp*chs2(i))
             endif
          else
             q2(i)  = q2mxy(i)
             t2m(i) = t2mxy(i)
          endif
          th2m(i) = t2m(i)*(P0/psfc(i))**rcp
       enddo

    case default
 end select sf_select

end associate
end subroutine
subroutine oracle_wrf_diagnostics(a,c,n,ua) bind(c)
integer(c_int),value:: n,ua
real(c_float),intent(inout):: a(n,48)
real(c_float),intent(in):: c(13)
call SFCDIAGS( &
reshape(a(:,5),[n,1]), &
reshape(a(:,6),[n,1]), &
reshape(a(:,42),[n,1]), &
reshape(a(:,7),[n,1]), &
reshape(a(:,21),[n,1]), &
reshape(a(:,22),[n,1]), &
a(:,18:18), &
a(:,17:17), &
a(:,19:19), &
reshape(a(:,41),[n,1]), &
c(1),c(4),c(3),reshape(a(:,20),[n,1]), &
reshape(a(:,37),[n,1,1]),reshape(a(:,38),[n,1,1]),(ua/=0), &
1,n,1,1,1,1,1,n,1,1,1,1,1,n,1,1,1,1)
end subroutine
subroutine oracle_driver_to(a,sea,held,xice,n,threshold) bind(c)
integer(c_int),value:: n
real(c_float),intent(inout):: a(n,48)
real(c_float),intent(in):: sea(n,48),held(n),xice(n)
real(c_float),value:: threshold
integer:: i
 do i=1,n
 if(xice(i)>=threshold .and. xice(i)<=1.) then
          a(i,28)     = a(i,28)*xice(i)     + (1.-xice(i))*sea(i,28)
          a(i,23)   = a(i,23)*xice(i)   + (1.-xice(i))*sea(i,23)
          a(i,24)   = a(i,24)*xice(i)   + (1.-xice(i))*sea(i,24)
          a(i,29) = a(i,29)*xice(i) + (1.-xice(i))*sea(i,29)
          a(i,4)    = a(i,4)*xice(i)    + (1.-xice(i))*sea(i,4)
          a(i,11)   = a(i,11)*xice(i)   + (1.-xice(i))*sea(i,11)
          a(i,10)   = a(i,10)*xice(i)   + (1.-xice(i))*sea(i,10)
          a(i,26)   = a(i,26)*xice(i)   + (1.-xice(i))*sea(i,26)
          a(i,2)    = a(i,2)*xice(i)    + (1.-xice(i))*sea(i,2)
          a(i,27)   = a(i,27)*xice(i)   + (1.-xice(i))*sea(i,27)
          a(i,8)    = a(i,8)*xice(i)    + (1.-xice(i))*sea(i,8)
          a(i,19)     = a(i,19)*xice(i)   + (1.-xice(i))*sea(i,19)
          a(i,18)    = a(i,18)*xice(i)  + (1.-xice(i))*sea(i,18)
          a(i,17)   = a(i,17)*xice(i) + (1.-xice(i))*sea(i,17)
          a(i,15)    = a(i,15)*xice(i)  + (1.-xice(i))*sea(i,15)
          a(i,16)    = a(i,16)*xice(i)  + (1.-xice(i))*sea(i,16)
          a(i,33)     = a(i,33)*xice(i)   + (1.-xice(i))*sea(i,33)
          a(i,34)    = a(i,34)*xice(i)  + (1.-xice(i))*sea(i,34)
          a(i,31)     = a(i,31)*xice(i)   + (1.-xice(i))*sea(i,31)
          a(i,32)    = a(i,32)*xice(i)  + (1.-xice(i))*sea(i,32)
          a(i,3)   = a(i,3)*xice(i)   + (1.-xice(i))*sea(i,3)
                a(i,13) = a(i,13)*xice(i) + (1.-xice(i))*sea(i,13)
                a(i,12) = a(i,12)*xice(i) + (1.-xice(i))*sea(i,12)
if(xice(i)>=0.5) a(i,9)=held(i)
endif
enddo
end subroutine
subroutine oracle_driver_from(a,sea,meshDensity,sst,xice,n,len_disp,threshold,frac) bind(c)
integer(c_int),value:: n,frac
real(c_float),intent(inout):: a(n,48)
real(c_float),intent(out):: sea(n,48)
real(c_float),intent(in):: meshDensity(n),sst(n),xice(n)
real(c_float),value:: len_disp,threshold
integer:: i
 do i=1,n
a(i,47)=len_disp/meshDensity(i)**0.25
a(i,19)=0.
a(i,18)=0.
a(i,17)=0.
a(i,15)=0.
a(i,16)=0.
a(i,33)=0.
a(i,34)=0.
a(i,31)=0.
a(i,32)=0.
a(i,46)=0.
a(i,48)=0.
enddo
if(frac==0) return
sea=a
sea(:,7)=0.
do i=1,n
if(xice(i)>=threshold .and. xice(i)<=1.) then
sea(i,45)=2.
sea(i,44)=1.
sea(i,1)=0.0001
sea(i,42)=max(sst(i),271.4)
endif
enddo
end subroutine
end module
