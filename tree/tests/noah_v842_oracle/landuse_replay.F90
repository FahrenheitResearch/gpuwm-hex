! MPAS v8.4.2 mpas_atmphys_landuse.F:257-315; unmodified numerical slice.
subroutine replay_landuse(x,ix,packed,n) bind(C,name="replay_landuse")
use iso_c_binding
implicit none
integer(c_int),value::n
real(c_float),target::x(14,n),packed(100,12,7)
integer(c_int),target::ix(9,n)
real,pointer::latCell(:),snoalb(:),snowc(:),xice(:),albbck(:),z0(:),znt(:),embck(:),sfc_albedo(:),sfc_emiss(:),thc(:),mavail(:),ust(:)
real,pointer::albd(:,:),slmo(:,:),sfem(:,:),sfz0(:,:),therin(:,:),scfx(:,:)
integer,pointer::ivgtyp(:)
integer::c,iCell,isn,is,julday,isice,iswater
integer,parameter::nCells=1
logical::config_sfc_albedo,config_frac_seaice
real::xice_threshold
albd=>packed(:,:,1);slmo=>packed(:,:,2);sfem=>packed(:,:,3)
sfz0=>packed(:,:,4);therin=>packed(:,:,5);scfx=>packed(:,:,6)
do c=1,n
if(ix(5,c)/=0)cycle
ivgtyp=>ix(1:1,c);julday=ix(2,c);iswater=ix(3,c);isice=ix(4,c)
config_sfc_albedo=ix(6,c)/=0;config_frac_seaice=ix(7,c)/=0
xice_threshold=x(14,c)
latCell=>x(1:1,c)
snoalb=>x(2:2,c)
snowc=>x(3:3,c)
xice=>x(4:4,c)
albbck=>x(5:5,c)
z0=>x(6:6,c)
znt=>x(7:7,c)
embck=>x(8:8,c)
sfc_albedo=>x(9:9,c)
sfc_emiss=>x(10:10,c)
thc=>x(11:11,c)
nullify(mavail,ust)
if(ix(8,c)/=0)mavail=>x(12:12,c)
if(ix(9,c)/=0)ust=>x(13:13,c)

!defines the surface properties over the entire domain:
 do iCell = 1, nCells

    !finds the season as function of julian day (summer=1, winter=2): summer in the Northern
    !Hemisphere is defined between March 15th and September 15th (winter otherwise).
    isn = 1
    if(julday.lt.105 .or. julday.ge.288) isn=2
    if(latCell(iCell) .lt. 0.) isn=3-isn

    is = ivgtyp(iCell)
    
    !set no data points to water:
    if(is.eq.0) is = iswater
    if(.not. config_sfc_albedo) albbck(iCell) = albd(is,isn)/100.
    sfc_albedo(iCell) = albbck(iCell)

    if(snowc(iCell) .gt. 0.5) then
       albbck(iCell) = albd(isice,isn) / 100.
       if(config_sfc_albedo) then
          sfc_albedo(iCell) = snoalb(iCell)
       else
          sfc_albedo(iCell) = albbck(iCell) / (1+scfx(is,isn))
       endif
    endif
    thc(iCell)    = therin(is,isn) / 100.
    z0(iCell)     = sfz0(is,isn) / 100.
    znt(iCell)    = z0(icell)
    embck(iCell)  = sfem(is,isn)
    sfc_emiss(iCell) = embck(iCell)

    if(associated(mavail)) mavail(iCell) = slmo(is,isn)
    if(associated(ust)) ust(iCell) = 0.0001
 
    !set sea-ice points to land with ice/snow surface properties:
    if(xice(iCell) .ge. xice_threshold) then
       albbck(iCell) = albd(isice,isn) / 100.
       embck(iCell)  = sfem(isice,isn)
       if(config_frac_seaice) then
          !0.08 is the albedo over open water.
          !0.98 is the emissivity over open water.
          sfc_albedo(iCell) = xice(iCell)*albbck(iCell) + (1-xice(iCell))*0.08
          sfc_emiss(iCell)  = xice(iCell)*embck(iCell)  + (1-xice(iCell))*0.98
       else
          sfc_albedo(iCell) = albbck(iCell)
          sfc_emiss(iCell)  = embck(iCell)
       endif
       thc(iCell) = therin(isice,isn) / 100.
       z0(icell)  = sfz0(isice,isn) / 100.
       znt(iCell) = z0(iCell)

       if(associated(mavail)) mavail(iCell) = slmo(isice,isn)
    endif

 enddo

!call mpas_log_write('--- end subroutine landuse_init_forMPAS')
enddo
end subroutine
! MPAS v8.4.2 mpas_atmphys_landuse.F:177-220; standalone native table decoder.
subroutine read_landuse(packed) bind(C,name="read_landuse")
use iso_c_binding
implicit none
real(c_float)::packed(100,12,7)
character(len=35)::lutype
integer::u,lucats,luseas,ic,is,istat
real::li
open(newunit=u,file='LANDUSE.TBL',status='old',action='read')
do
 read(u,'(a35)')lutype
 read(u,*)lucats,luseas
 if(lutype=='MODIFIED_IGBP_MODIS_NOAH')exit
 do is=1,luseas
  read(u,*)
  do ic=1,lucats
   read(u,*)
  enddo
 enddo
enddo
packed=0
 do is=1,luseas
 read(u,*)
 do ic=1,lucats
 read(u,*)li,packed(ic,is,1),packed(ic,is,2),packed(ic,is,3),packed(ic,is,4), &
  packed(ic,is,5),packed(ic,is,6),packed(ic,is,7)
 enddo
 enddo
close(u)
end subroutine

subroutine read_landuse_for(packed,chars,nchar) bind(C,name="read_landuse_for")
use iso_c_binding
implicit none
real(c_float)::packed(100,12,7)
integer(c_int),value::nchar
character(kind=c_char)::chars(nchar)
character(len=35)::lutype,wanted
integer::k
integer::u,lucats,luseas,ic,is,istat
real::li
wanted=' '
do k=1,nchar
wanted(k:k)=chars(k)
enddo
open(newunit=u,file='LANDUSE.TBL',status='old',action='read')
do
 read(u,'(a35)')lutype
 read(u,*)lucats,luseas
 if(lutype==wanted)exit
 do is=1,luseas
  read(u,*)
  do ic=1,lucats
   read(u,*)
  enddo
 enddo
enddo
packed=0
 do is=1,luseas
 read(u,*)
 do ic=1,lucats
 read(u,*)li,packed(ic,is,1),packed(ic,is,2),packed(ic,is,3),packed(ic,is,4), &
  packed(ic,is,5),packed(ic,is,6),packed(ic,is,7)
 enddo
 enddo
close(u)
end subroutine
