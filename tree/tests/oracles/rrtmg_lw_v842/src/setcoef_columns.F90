! MPAS: module_ra_rrtmg_lw.F:3567-3932, including band-16-only mode.
program setcoef_columns
use rrtmg_lw_setcoef,only:setcoef
use mpas_atmphys_rrtmg_lwinit,only:rrtmg_initlw_forMPAS
use mpas_derived_types,only:dm_info
use lw_binio
implicit none
integer,parameter::nl=8
real::pavel(nl),tavel(nl),tz(0:nl),tbound,semiss(16),coldry(nl),wkl(38,nl),wbroad(nl)
real::planklay(nl,16),planklev(0:nl,16),plankbnd(16)
type(dm_info)::info
integer::scenario,k,istart,laytrop
character(20)::prefix
character(1024)::outdir
real::colh2o(nl),colco2(nl),colo3(nl),coln2o(nl),colco(nl),colch4(nl),colo2(nl),colbrd(nl),fac00(nl),fac01(nl),fac10(nl),fac11(nl),rat_h2oco2(nl),rat_h2oco2_1(nl),rat_h2oo3(nl),rat_h2oo3_1(nl),rat_h2on2o(nl),rat_h2on2o_1(nl),rat_h2och4(nl),rat_h2och4_1(nl),rat_n2oco2(nl),rat_n2oco2_1(nl),rat_o3co2(nl),rat_o3co2_1(nl),selffac(nl),selffrac(nl),forfac(nl),forfrac(nl),minorfrac(nl),scaleminor(nl),scaleminorn2(nl)
integer::jp(nl),jt(nl),jt1(nl),indself(nl),indfor(nl),indminor(nl)
call get_command_argument(1,outdir)
call rrtmg_initlw_forMPAS(info)
call bio_open(trim(outdir)//'/native-setcoef.bin')
do scenario=0,13
istart=1;if(mod(scenario,2)==1)istart=16
pavel=[900.,700.,500.,300.,100.,30.,10.,.00001]
tavel=155.+real(scenario/2)*32.
if(scenario>=12)tavel=250.
tz=tavel(1)+1.;tbound=tavel(1)+12.;semiss=.975
coldry=1.e23;wbroad=8.e22;wkl=1.e18
wkl(1,:)=1.e22;wkl(2,:)=4.e19;wkl(3,:)=1.e17;wkl(6,:)=1.e18
! Zero CO2/O3/N2O/CH4 in both full-spectrum and band-16-only mode.
if(scenario>=12)then
 wkl(2:4,:)=0.;wkl(6,:)=0.
endif
planklay=0.;planklev=0.;plankbnd=0.;laytrop=0
colh2o=0.
colco2=0.
colo3=0.
coln2o=0.
colco=0.
colch4=0.
colo2=0.
colbrd=0.
fac00=0.
fac01=0.
fac10=0.
fac11=0.
rat_h2oco2=0.
rat_h2oco2_1=0.
rat_h2oo3=0.
rat_h2oo3_1=0.
rat_h2on2o=0.
rat_h2on2o_1=0.
rat_h2och4=0.
rat_h2och4_1=0.
rat_n2oco2=0.
rat_n2oco2_1=0.
rat_o3co2=0.
rat_o3co2_1=0.
selffac=0.
selffrac=0.
forfac=0.
forfrac=0.
minorfrac=0.
scaleminor=0.
scaleminorn2=0.
jp=0
jt=0
jt1=0
indself=0
indfor=0
indminor=0
call setcoef(nl,istart,pavel,tavel,tz,tbound,semiss,coldry,wkl,wbroad,laytrop,jp,jt,jt1,planklay,planklev,plankbnd, &
colh2o,colco2,colo3,coln2o,colco,colch4,colo2,colbrd,fac00,fac01,fac10,fac11,rat_h2oco2,rat_h2oco2_1,rat_h2oo3,rat_h2oo3_1,rat_h2on2o,rat_h2on2o_1,rat_h2och4,rat_h2och4_1,rat_n2oco2,rat_n2oco2_1,rat_o3co2,rat_o3co2_1,selffac,selffrac,indself,forfac,forfrac,indfor,minorfrac,scaleminor,scaleminorn2,indminor)
write(prefix,'(i3.3)')scenario
call wr_i0(trim(prefix)//'/istart',istart)
call wr_r1(trim(prefix)//'/pavel',pavel)
call wr_r1(trim(prefix)//'/tavel',tavel)
call wr_r1(trim(prefix)//'/tz',tz)
call wr_r1(trim(prefix)//'/semiss',semiss)
call wr_r1(trim(prefix)//'/coldry',coldry)
call wr_r1(trim(prefix)//'/wbroad',wbroad)
call wr_r1(trim(prefix)//'/colh2o',colh2o)
call wr_r1(trim(prefix)//'/colco2',colco2)
call wr_r1(trim(prefix)//'/colo3',colo3)
call wr_r1(trim(prefix)//'/coln2o',coln2o)
call wr_r1(trim(prefix)//'/colco',colco)
call wr_r1(trim(prefix)//'/colch4',colch4)
call wr_r1(trim(prefix)//'/colo2',colo2)
call wr_r1(trim(prefix)//'/colbrd',colbrd)
call wr_r1(trim(prefix)//'/fac00',fac00)
call wr_r1(trim(prefix)//'/fac01',fac01)
call wr_r1(trim(prefix)//'/fac10',fac10)
call wr_r1(trim(prefix)//'/fac11',fac11)
call wr_r1(trim(prefix)//'/rat_h2oco2',rat_h2oco2)
call wr_r1(trim(prefix)//'/rat_h2oco2_1',rat_h2oco2_1)
call wr_r1(trim(prefix)//'/rat_h2oo3',rat_h2oo3)
call wr_r1(trim(prefix)//'/rat_h2oo3_1',rat_h2oo3_1)
call wr_r1(trim(prefix)//'/rat_h2on2o',rat_h2on2o)
call wr_r1(trim(prefix)//'/rat_h2on2o_1',rat_h2on2o_1)
call wr_r1(trim(prefix)//'/rat_h2och4',rat_h2och4)
call wr_r1(trim(prefix)//'/rat_h2och4_1',rat_h2och4_1)
call wr_r1(trim(prefix)//'/rat_n2oco2',rat_n2oco2)
call wr_r1(trim(prefix)//'/rat_n2oco2_1',rat_n2oco2_1)
call wr_r1(trim(prefix)//'/rat_o3co2',rat_o3co2)
call wr_r1(trim(prefix)//'/rat_o3co2_1',rat_o3co2_1)
call wr_r1(trim(prefix)//'/selffac',selffac)
call wr_r1(trim(prefix)//'/selffrac',selffrac)
call wr_r1(trim(prefix)//'/forfac',forfac)
call wr_r1(trim(prefix)//'/forfrac',forfrac)
call wr_r1(trim(prefix)//'/minorfrac',minorfrac)
call wr_r1(trim(prefix)//'/scaleminor',scaleminor)
call wr_r1(trim(prefix)//'/scaleminorn2',scaleminorn2)
call wr_r0(trim(prefix)//'/tbound',tbound)
call wr_r2(trim(prefix)//'/wkl',wkl)
call wr_i0(trim(prefix)//'/laytrop',laytrop)
call wr_i1(trim(prefix)//'/jp',jp)
call wr_i1(trim(prefix)//'/jt',jt)
call wr_i1(trim(prefix)//'/jt1',jt1)
call wr_i1(trim(prefix)//'/indself',indself)
call wr_i1(trim(prefix)//'/indfor',indfor)
call wr_i1(trim(prefix)//'/indminor',indminor)
call wr_r2(trim(prefix)//'/planklay',planklay)
call wr_r2(trim(prefix)//'/planklev',planklev)
call wr_r1(trim(prefix)//'/plankbnd',plankbnd)
enddo
call bio_close()
end program setcoef_columns
