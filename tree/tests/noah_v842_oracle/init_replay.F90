! MPAS v8.4.2 mpas_atmphys_lsm_noahinit.F:136-205, FNDSNOWH=.true.
subroutine replay_init(x,ix,n) bind(C,name="replay_init")
use iso_c_binding
use module_sf_noahlsm
use mpas_atmphys_constants, only: gravity
implicit none
integer(c_int),value::n
real(c_float)::x(15,n)
integer(c_int)::ix(4,n)
real::tslb(4),smois(4),sh2o(4),bx,fk,smcmax,psisat,free
integer::c,ivgtyp,isltyp,ns
integer,parameter::nSoilLevels=4
real,parameter::blim=5.5,hlice=3.335e5,t0=273.15
 do c=1,n
 if(ix(3,c)/=0)cycle
 ivgtyp=ix(1,c);isltyp=ix(2,c)
 if(isltyp<1)error stop 'out of range soil type'
 if(ix(4,c)==0) x(13,c)=maxalb(ivgtyp)*0.01
 tslb=x(1:4,c);smois=x(5:8,c);sh2o=x(9:12,c)
       bx = bb(isltyp)
       smcmax = maxsmc(isltyp)
       psisat = satpsi(isltyp)
       if((bx > 0.0).and.(smcmax > 0.0).and.(psisat > 0.0)) then
          do ns = 1, nSoilLevels
! ----------------------------------------------------------------------
!SH2O  <= SMOIS for T < 273.149K (-0.001C)
             if(tslb(ns) < 273.149) then
! ----------------------------------------------------------------------
! first guess following explicit solution for Flerchinger Eqn from Koren
! et al, JGR, 1999, Eqn 17 (KCOUNT=0 in FUNCTION FRH2O).
! ISLTPK is soil type
                bx = bb(isltyp)
                smcmax = maxsmc(isltyp)
                psisat = satpsi(isltyp)
                if(bx >  blim) bx = blim
                fk = (((hlice/(gravity*(-psisat))) * &
                     ((tslb(ns)-t0)/tslb(ns)) )**(-1/bx) )*smcmax
                if (fk < 0.02) fk = 0.02
                sh2o(ns) = min(fk,smois(ns))
! ----------------------------------------------------------------------
! now use iterative solution for liquid soil water content using
! FUNCTION FRH2O with the initial guess for SH2O from above explicit
! first guess.
                call frh2o(free,tslb(ns),smois(ns),sh2o(ns), &
                           smcmax,bx,psisat)
                sh2o(ns) = free
             else         ! of if (tslb(i,ns,j)
! ----------------------------------------------------------------------
! SH2O = SMOIS ( for T => 273.149K (-0.001C)
                sh2o(ns)=smois(ns)
! ----------------------------------------------------------------------
             endif        ! of if (tslb(i,ns,j)
          enddo
       else                  ! of if ((bx > 0.0)
          do ns = 1, nSoilLevels
             sh2o(ns)=smois(ns)
          enddo
       endif ! of if ((bx > 0.0)
x(9:12,c)=sh2o
enddo
end subroutine
