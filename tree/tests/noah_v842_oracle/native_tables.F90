! Numerical table loader extracted without changes from MPAS v8.4.2
! mpas_atmphys_lsm_noahinit.F:210-597; framework calls stubbed for one rank.
#define DM_BCAST_CHAR(A)     call mpas_dmpar_bcast_char(dminfo,A)
#define DM_BCAST_INTEGER(A)  call mpas_dmpar_bcast_int(dminfo,A)
#define DM_BCAST_INTEGERS(A) call mpas_dmpar_bcast_ints(dminfo,size(A),A)
#define DM_BCAST_REAL(A)     call mpas_dmpar_bcast_real(dminfo,A)
#define DM_BCAST_REALS(A)    call mpas_dmpar_bcast_reals(dminfo,size(A),A)
module native_noah_tables
use mpas_dmpar
use mpas_io_units
use mpas_atmphys_utilities
use module_sf_noahlsm
contains
 subroutine soil_veg_gen_parm(dminfo,mminlu,mminsl)
!=================================================================================================================

!input arguments:
 type(dm_info),intent(in):: dminfo
 character(len=*),intent(inout):: mminlu, mminsl

!local variables:
 character*128:: mess,message
 character*128:: astring

 integer,parameter:: open_ok  = 0
 integer,parameter:: loop_max = 10
 integer:: lumatch,iindex,lc,num_slope
 integer:: istat,loop_count,read_unit

!-----SPECIFY VEGETATION RELATED CHARACTERISTICS :
!ALBBCK: SFC albedo (in percentage)
!Z0    : Roughness length (m)
!SHDFAC: Green vegetation fraction (in percentage)

!Note: The ALBEDO, Z0, and SHDFAC values read from the following table
!ALBEDO,and Z0 are specified in LAND-USE TABLE. SHDFAC is the monthly green vegetation data.
!CMXTBL: MAX CNPY Capacity (m)
!NROTBL: Rooting depth (layer)
!RSMIN : Mimimum stomatal resistance (s m-1)
!RSMAX : Max. stomatal resistance (s m-1)
!RGL   : Parameters used in radiation stress function
!HS    : Parameter used in vapor pressure deficit function
!TOPT  : Optimum transpiration air temperature. (K)
!CMCMAX: Maximum canopy water capacity
!CFACTR: Parameter used in the canopy inteception calculation
!SNUP  : Threshold snow depth (in water equivalent m) that implies 100% snow cover.
!LAI   : Leaf area index (dimensionless)
!MAXALB: Upper bound on maximum albedo over deep snow
!
!-----READ IN VEGETATION PROPERTIES FROM VEGPARM.TBL

!-----------------------------------------------------------------------------------------------------------------
!call mpas_log_write('')
!call mpas_log_write('       enter subroutine soil_veg_gen_parm:')

!read in the vegetation properties from vegparm.tbl:

 if(dminfo % my_proc_id == IO_NODE) then
    !get a unit to open init file:
    call mpas_new_unit(read_unit)
    if ( read_unit < 0 ) &
       call physics_error_fatal('soil_veg_gen_parm: All file units are taken.  Change maxUnits in mpas_io_units.F')

    open(read_unit,file='VEGPARM.TBL',form='FORMATTED',status='OLD',iostat=istat)
    if(istat /= open_ok) &
       call physics_error_fatal('subroutine soil_veg_gen_arm: ' // &
                                'failure opening VEGPARM.TBL')

    lumatch=0

    loop_count = 0
    read(read_unit,fmt='(A)',end=2002) astring
    find_lutype : do while (lumatch == 0)
       read(read_unit,*,end=2002) lutype
       read(read_unit,*) lucats,iindex

       if(lutype.eq.trim(mminlu))then
          write(mess,*) '   landuse type = ' // trim ( lutype ) // ' found', &
                        lucats, ' categories'
          call physics_message(mess)
          lumatch=1
       else
          loop_count = loop_count + 1
          call physics_message('    skipping over lutype = ' // trim ( lutype ))

          find_vegetation_parameter_flag: do
             read(read_unit,fmt='(A)',end=2002) astring
             if(astring(1:21) .eq. 'Vegetation Parameters') then
                exit find_vegetation_parameter_flag
             elseif(loop_count .ge. loop_max) then
                call physics_error_fatal('too many loops in VEGPARM.TBL')
             endif
          enddo find_vegetation_parameter_flag
       endif
    enddo find_lutype

!prevent possible array overwrite, Bill Bovermann, IBM, May 6, 2008:
    if(size(shdtbl)       < lucats .or. &
       size(nrotbl)       < lucats .or. &
       size(rstbl)        < lucats .or. &
       size(rgltbl)       < lucats .or. &
       size(hstbl)        < lucats .or. &
       size(snuptbl)      < lucats .or. &
       size(maxalb)       < lucats .or. &
       size(laimintbl)    < lucats .or. &
       size(laimaxtbl)    < lucats .or. &
       size(z0mintbl)     < lucats .or. &
       size(z0maxtbl)     < lucats .or. &
       size(albedomintbl) < lucats .or. &
       size(albedomaxtbl) < lucats .or. &
       size(ztopvtbl)     < lucats .or. &
       size(zbotvtbl)     < lucats .or. &
       size(emissmintbl ) < lucats .or. &
       size(emissmaxtbl ) < lucats) then
       call physics_error_fatal('table sizes too small for value of lucats in module_sf_noahdrv.f')
    endif

    if(lutype.eq.mminlu)then
       do lc = 1, lucats
          read(read_unit,*) iindex,shdtbl(lc),nrotbl(lc),rstbl(lc),rgltbl(lc),hstbl(lc),snuptbl(lc),  &
                     maxalb(lc),laimintbl(lc),laimaxtbl(lc),emissmintbl(lc),emissmaxtbl(lc),   &
                     albedomintbl(lc),albedomaxtbl(lc),z0mintbl(lc),z0maxtbl(lc),ztopvtbl(lc), &
                     zbotvtbl(lc)
       enddo

       read (read_unit,*)
       read (read_unit,*)topt_data
       read (read_unit,*)
       read (read_unit,*)cmcmax_data
       read (read_unit,*)
       read (read_unit,*)cfactr_data
       read (read_unit,*)
       read (read_unit,*)rsmax_data
       read (read_unit,*)
       read (read_unit,*)bare
       read (read_unit,*)
       read (read_unit,*)natural
       read (read_unit,*)
       read (read_unit,*)
       read (read_unit,*)
       read (read_unit,*)lcz_1
       read (read_unit,*)
       read (read_unit,*)lcz_2
       read (read_unit,*)
       read (read_unit,*)lcz_3
       read (read_unit,*)
       read (read_unit,*)lcz_4
       read (read_unit,*)
       read (read_unit,*)lcz_5
       read (read_unit,*)
       read (read_unit,*)lcz_6
       read (read_unit,*)
       read (read_unit,*)lcz_7
       read (read_unit,*)
       read (read_unit,*)lcz_8
       read (read_unit,*)
       read (read_unit,*)lcz_9
       read (read_unit,*)
       read (read_unit,*)lcz_10
       read (read_unit,*)
       read (read_unit,*)lcz_11
    endif

    2002 continue
    close (read_unit)
    if(lumatch == 0) &
       call physics_error_fatal ('land use dataset '''//mminlu//''' not found in VEGPARM.TBL.')
      
 endif ! end dminfo

!distribute data to all processors: 
 DM_BCAST_CHAR(lutype)
 DM_BCAST_INTEGER(lucats)
 DM_BCAST_INTEGER(iindex)
 DM_BCAST_INTEGER(lumatch)
 DM_BCAST_REALS(shdtbl)
 DM_BCAST_INTEGERS(nrotbl)
 DM_BCAST_REALS(rstbl)
 DM_BCAST_REALS(rgltbl)
 DM_BCAST_REALS(hstbl)
 DM_BCAST_REALS(snuptbl)
 DM_BCAST_REALS(laimintbl)
 DM_BCAST_REALS(laimaxtbl)
 DM_BCAST_REALS(z0mintbl)
 DM_BCAST_REALS(z0maxtbl)
 DM_BCAST_REALS(emissmintbl)
 DM_BCAST_REALS(emissmaxtbl)
 DM_BCAST_REALS(albedomintbl)
 DM_BCAST_REALS(albedomaxtbl)
 DM_BCAST_REALS(ztopvtbl)
 DM_BCAST_REALS(zbotvtbl)
 DM_BCAST_REALS(maxalb)
 DM_BCAST_REAL(topt_data)
 DM_BCAST_REAL(cmcmax_data)
 DM_BCAST_REAL(cfactr_data)
 DM_BCAST_REAL(rsmax_data)
 DM_BCAST_INTEGER(bare)
 DM_BCAST_INTEGER(natural)
 DM_BCAST_INTEGER(lcz_1)
 DM_BCAST_INTEGER(lcz_2)
 DM_BCAST_INTEGER(lcz_3)
 DM_BCAST_INTEGER(lcz_4)
 DM_BCAST_INTEGER(lcz_5)
 DM_BCAST_INTEGER(lcz_6)
 DM_BCAST_INTEGER(lcz_7)
 DM_BCAST_INTEGER(lcz_8)
 DM_BCAST_INTEGER(lcz_9)
 DM_BCAST_INTEGER(lcz_10)
 DM_BCAST_INTEGER(lcz_11)

!call mpas_log_write(' LUTYPE  = '//trim(lutype))
!call mpas_log_write(' LUCATS  = $i',intArgs=(/lucats/))
!call mpas_log_write(' IINDEX  = $i',intArgs=(/iindex/))
!call mpas_log_write(' LUMATCH = $i',intArgs=(/lumatch/))

!call mpas_log_write(' TOPT_DATA   = $r',realArgs=(/topt_data/))
!call mpas_log_write(' CMCMAX_DATA = $r',realArgs=(/cmcmax_data/))
!call mpas_log_write(' CFACTR_DATA = $r',realArgs=(/cfactr_data/))
!call mpas_log_write(' RSMAX_DATA  = $r',realArgs=(/rsmax_data/))
!call mpas_log_write(' BARE        = $i',intArgs=(/bare/))
!call mpas_log_write(' NATURAL     = $i',intArgs=(/natural/))
!call mpas_log_write(' LCZ_1       = $i', intArgs=(/lcz_1/))
!call mpas_log_write(' LCZ_2       = $i', intArgs=(/lcz_2/))
!call mpas_log_write(' LCZ_3       = $i', intArgs=(/lcz_3/))
!call mpas_log_write(' LCZ_4       = $i', intArgs=(/lcz_4/))
!call mpas_log_write(' LCZ_5       = $i', intArgs=(/lcz_5/))
!call mpas_log_write(' LCZ_6       = $i', intArgs=(/lcz_6/))
!call mpas_log_write(' LCZ_7       = $i', intArgs=(/lcz_7/))
!call mpas_log_write(' LCZ_8       = $i', intArgs=(/lcz_8/))
!call mpas_log_write(' LCZ_9       = $i', intArgs=(/lcz_9/))
!call mpas_log_write(' LCZ_10      = $i', intArgs=(/lcz_10/))
!call mpas_log_write(' LCZ_11      = $i', intArgs=(/lcz_11/))
!call mpas_log_write('')
!do lc = 1, lucats
!   call mpas_log_write('$i $r $r $r $r $r $r $r $r $r $r $r $r $r $r $r $r $r', intArgs=(/lc/),     &
!                realArgs=(/shdtbl(lc),float(nrotbl(lc)),rstbl(lc),rgltbl(lc),hstbl(lc),snuptbl(lc), &
!                maxalb(lc),laimintbl(lc),laimaxtbl(lc),emissmintbl(lc),emissmaxtbl(lc),             &
!                albedomintbl(lc),albedomaxtbl(lc),z0mintbl(lc),z0maxtbl(lc),ztopvtbl(lc),           &
!                zbotvtbl(lc)/))
!enddo
 call mpas_log_write('    end read VEGPARM.TBL')

!read in soil properties from soilparm.tbl:

 if(dminfo % my_proc_id == IO_NODE) then
    open(read_unit,file='SOILPARM.TBL',form='FORMATTED',status='OLD',iostat=istat)
    if(istat /= open_ok) &
       call physics_error_fatal('module_sf_noahlsm.F: soil_veg_gen_parm: ' // &
                                'failure opening SOILPARM.TBL' )

    write(mess,*) '   input soil texture classification = ', trim (mminsl)
    call physics_message(mess)

    lumatch=0
    read(read_unit,*)
    read(read_unit,2000,end=2003) sltype
    2000 format(a4)
    read(read_unit,*)slcats,iindex
    if(sltype.eq.mminsl)then
       write(mess,*) '   soil texture classification = ', trim ( sltype ) , ' found', &
                  slcats,' categories'
       call physics_message ( mess )
       lumatch=1
    endif

!prevent possible array overwrite, Bill Bovermann, IBM, May 6, 2008:
    if(size(bb) < slcats .or. &
       size(drysmc) < slcats .or. &
       size(f11   ) < slcats .or. &
       size(maxsmc) < slcats .or. &
       size(refsmc) < slcats .or. &
       size(satpsi) < slcats .or. &
       size(satdk ) < slcats .or. &
       size(satdw ) < slcats .or. &
       size(wltsmc) < slcats .or. &
       size(qtz   ) < slcats) then
!      call wrf_error_fatal('table sizes too small for value of slcats in module_sf_noahdrv.f')
    endif
    if(sltype.eq.mminsl) then
       do lc = 1, slcats
          read(read_unit,*) iindex,bb(lc),drysmc(lc),f11(lc),maxsmc(lc),refsmc(lc),satpsi(lc), &
                      satdk(lc),satdw(lc),wltsmc(lc),qtz(lc)
       enddo
    endif

    2003 continue
    close(read_unit)
    if(lumatch.eq.0)then
       call physics_message( 'soil texture in input file does not ' )
       call physics_message( 'match soilparm table'                 )
       call physics_error_fatal( 'inconsistent or missing soilparm file' )
    endif

 endif

!distribute data to all processors: 
 DM_BCAST_INTEGER(lumatch)
 DM_BCAST_CHAR(sltype)
 DM_BCAST_CHAR(mminsl)
 DM_BCAST_INTEGER(slcats)
 DM_BCAST_INTEGER(iindex)
 DM_BCAST_REALS(bb)
 DM_BCAST_REALS(drysmc)
 DM_BCAST_REALS(f11)
 DM_BCAST_REALS(maxsmc)
 DM_BCAST_REALS(refsmc)
 DM_BCAST_REALS(satpsi)
 DM_BCAST_REALS(satdk)
 DM_BCAST_REALS(satdw)
 DM_BCAST_REALS(wltsmc)
 DM_BCAST_REALS(qtz)

!call mpas_log_write(' LUMATCH=$i',intArgs=(/lumatch/))
!call mpas_log_write(' SLTYPE ='//trim(sltype))
!call mpas_log_write(' MMINSL ='//mminsl)
!call mpas_log_write(' SLCATS =$i',intArgs=(/slcats/))
!call mpas_log_write(' IINDEX =$i',intArgs=(/iindex/))
!call mpas_log_write('')
!do lc = 1, slcats
!   call mpas_log_write('$i $r $r $r $r $r $r $r $r $r $r', intArgs=(/lc/), &
!                realArgs=(/bb(lc),drysmc(lc),f11(lc),maxsmc(lc),refsmc(lc),satpsi(lc), &
!                satdk(lc),satdw(lc),wltsmc(lc),qtz(lc)/))
!enddo
 call mpas_log_write('    end read SOILPARM.TBL')

!read in general parameters from genparm.tbl:

 if(dminfo % my_proc_id == IO_NODE) then
    open(read_unit,file='GENPARM.TBL',form='FORMATTED',status='OLD',iostat=istat)
    if(istat /= open_ok) &
       call physics_error_fatal('module_sf_noahlsm.F: soil_veg_gen_parm: ' // &
                                'failure opening GENPARM.TBL' )
    read(read_unit,*)
    read(read_unit,*)
    read(read_unit,*) num_slope

    slpcats=num_slope
!prevent possible array overwrite, Bill Bovermann, IBM, May 6, 2008:
    if(size(slope_data) < num_slope) &
       call physics_error_fatal('NUM_SLOPE too large for slope_data array' // &
                                'in module_sf_noahdrv')

    do lc = 1, slpcats
       read(read_unit,*)slope_data(lc)
    enddo
    read(read_unit,*)
    read(read_unit,*)sbeta_data
    read(read_unit,*)
    read(read_unit,*)fxexp_data
    read(read_unit,*)
    read(read_unit,*)csoil_data
    read(read_unit,*)
    read(read_unit,*)salp_data
    read(read_unit,*)
    read(read_unit,*)refdk_data
    read(read_unit,*)
    read(read_unit,*)refkdt_data
    read(read_unit,*)
    read(read_unit,*)frzk_data
    read(read_unit,*)
    read(read_unit,*)zbot_data
    read(read_unit,*)
    read(read_unit,*)czil_data
    read(read_unit,*)
    read(read_unit,*)smlow_data
    read(read_unit,*)
    read(read_unit,*)smhigh_data
    read(read_unit,*)
    read(read_unit,*)lvcoef_data
    close(read_unit)
    call mpas_release_unit(read_unit)
 endif

 DM_BCAST_INTEGER(num_slope)
 DM_BCAST_INTEGER(slpcats)
 DM_BCAST_REALS(slope_data)
 DM_BCAST_REAL(sbeta_data)
 DM_BCAST_REAL(fxexp_data)
 DM_BCAST_REAL(csoil_data)
 DM_BCAST_REAL(salp_data)
 DM_BCAST_REAL(refdk_data)
 DM_BCAST_REAL(refkdt_data)
 DM_BCAST_REAL(frzk_data)
 DM_BCAST_REAL(zbot_data)
 DM_BCAST_REAL(czil_data)
 DM_BCAST_REAL(smlow_data)
 DM_BCAST_REAL(smhigh_data)
 DM_BCAST_REAL(lvcoef_data)

!call mpas_log_write('')
!write(mess,*) 'input general parameters'
!call physics_message(mess)
!call mpas_log_write('NUM_SLOPE=$i',intArgs=(/num_slope/))
!do lc = 1, slpcats
!   call mpas_log_write('$i $r', intArgs=(/lc/),realArgs=(/slope_data(lc)/))
!enddo
!call mpas_log_write('       end subroutine soil_veg_gen_parm:')
 call mpas_log_write('    end read GENPARM.TBL')

 end subroutine soil_veg_gen_parm
subroutine read_tables() bind(C,name="read_tables")
type(dm_info)::info
character(len=256)::veg,soil
veg='MODIFIED_IGBP_MODIS_NOAH'
soil='STAS'
call soil_veg_gen_parm(info,veg,soil)
end subroutine
! Native table selection through the same unmodified MPAS decoder.
subroutine read_tables_for(vchars,schars,nv,ns) bind(C,name="read_tables_for")
use iso_c_binding
integer(c_int),value::nv,ns
character(kind=c_char)::vchars(nv),schars(ns)
type(dm_info)::info
character(len=256)::veg,soil
integer::k
veg=' ';soil=' '
do k=1,nv
veg(k:k)=vchars(k)
enddo
do k=1,ns
soil(k:k)=schars(k)
enddo
call soil_veg_gen_parm(info,veg,soil)
end subroutine
end module
