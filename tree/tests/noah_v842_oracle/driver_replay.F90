! MPAS v8.4.2 physics_wrf/module_sf_noahdrv.F:44-1794; pristine driver, urban=0.
subroutine replay_driver(x,ix,n) bind(C,name="replay_driver")
use iso_c_binding
use module_sf_noahdrv, only: lsm
implicit none
integer(c_int),value::n
real(c_float)::x(79,n)
integer(c_int)::ix(7,n)
integer::c
real(c_float)::DZ8W(1,2,1)
real(c_float)::QV3D(1,2,1)
real(c_float)::P8W3D(1,2,1)
real(c_float)::T3D(1,2,1)
real(c_float)::TSK(1,1)
real(c_float)::HFX(1,1)
real(c_float)::QFX(1,1)
real(c_float)::LH(1,1)
real(c_float)::GRDFLX(1,1)
real(c_float)::QGH(1,1)
real(c_float)::SWDOWN(1,1)
real(c_float)::GLW(1,1)
real(c_float)::SMSTAV(1,1)
real(c_float)::SMSTOT(1,1)
real(c_float)::SFCRUNOFF(1,1)
real(c_float)::UDRUNOFF(1,1)
integer(c_int)::IVGTYP(1,1)
integer(c_int)::ISLTYP(1,1)
integer(c_int)::ISURBAN
integer(c_int)::ISICE
real(c_float)::VEGFRA(1,1)
real(c_float)::ALBEDO(1,1)
real(c_float)::ALBBCK(1,1)
real(c_float)::ZNT(1,1)
real(c_float)::Z0(1,1)
real(c_float)::TMN(1,1)
real(c_float)::XLAND(1,1)
real(c_float)::XICE(1,1)
real(c_float)::EMISS(1,1)
real(c_float)::EMBCK(1,1)
real(c_float)::SNOWC(1,1)
real(c_float)::QSFC(1,1)
real(c_float)::RAINBL(1,1)
real(c_float)::DT
real(c_float)::DZS(4)
integer(c_int)::ITIMESTEP
real(c_float)::SMOIS(1,4,1)
real(c_float)::TSLB(1,4,1)
real(c_float)::SNOW(1,1)
real(c_float)::CANWAT(1,1)
real(c_float)::CHS(1,1)
real(c_float)::CHS2(1,1)
real(c_float)::CQS2(1,1)
real(c_float)::SR(1,1)
real(c_float)::CHKLOWQ(1,1)
real(c_float)::LAI(1,1)
real(c_float)::QZ0(1,1)
real(c_float)::SH2O(1,4,1)
real(c_float)::SNOWH(1,1)
real(c_float)::SNOALB(1,1)
real(c_float)::SHDMIN(1,1)
real(c_float)::SHDMAX(1,1)
real(c_float)::SNOTIME(1,1)
real(c_float)::ACSNOM(1,1)
real(c_float)::ACSNOW(1,1)
real(c_float)::SNOPCX(1,1)
real(c_float)::POTEVP(1,1)
real(c_float)::SMCREL(1,4,1)
real(c_float)::XICE_THRESHOLD
logical::USEMONALB
real(c_float)::RIB(1,1)
real(c_float)::NOAHRES(1,1)
integer(c_int)::OPT_THCND
real(c_float)::FLX4_2D(1,1)
real(c_float)::FVB_2D(1,1)
real(c_float)::FBUR_2D(1,1)
real(c_float)::FGSN_2D(1,1)
real(c_float)::GSW(1,1)
real(c_float)::SWDDIR(1,1)
real(c_float)::SWDDIF(1,1)
real(c_float)::CPM(1,1)
real(c_float)::UST_URB2D(1,1)
real(c_float)::FRC_URB2D(1,1)
integer(c_int)::UTYPE_URB2D(1,1)
do c=1,n
DZ8W = reshape(x(1:2,c),shape(DZ8W))
QV3D = reshape(x(3:4,c),shape(QV3D))
P8W3D = reshape(x(5:6,c),shape(P8W3D))
T3D = reshape(x(7:8,c),shape(T3D))
TSK = reshape(x(9:9,c),shape(TSK))
HFX = reshape(x(10:10,c),shape(HFX))
QFX = reshape(x(11:11,c),shape(QFX))
LH = reshape(x(12:12,c),shape(LH))
GRDFLX = reshape(x(13:13,c),shape(GRDFLX))
QGH = reshape(x(14:14,c),shape(QGH))
SWDOWN = reshape(x(15:15,c),shape(SWDOWN))
GLW = reshape(x(16:16,c),shape(GLW))
SMSTAV = reshape(x(17:17,c),shape(SMSTAV))
SMSTOT = reshape(x(18:18,c),shape(SMSTOT))
SFCRUNOFF = reshape(x(19:19,c),shape(SFCRUNOFF))
UDRUNOFF = reshape(x(20:20,c),shape(UDRUNOFF))
IVGTYP = reshape(ix(1:1,c),shape(IVGTYP))
ISLTYP = reshape(ix(2:2,c),shape(ISLTYP))
ISURBAN = ix(3,c)
ISICE = ix(4,c)
VEGFRA = reshape(x(21:21,c),shape(VEGFRA))
ALBEDO = reshape(x(22:22,c),shape(ALBEDO))
ALBBCK = reshape(x(23:23,c),shape(ALBBCK))
ZNT = reshape(x(24:24,c),shape(ZNT))
Z0 = reshape(x(25:25,c),shape(Z0))
TMN = reshape(x(26:26,c),shape(TMN))
XLAND = reshape(x(27:27,c),shape(XLAND))
XICE = reshape(x(28:28,c),shape(XICE))
EMISS = reshape(x(29:29,c),shape(EMISS))
EMBCK = reshape(x(30:30,c),shape(EMBCK))
SNOWC = reshape(x(31:31,c),shape(SNOWC))
QSFC = reshape(x(32:32,c),shape(QSFC))
RAINBL = reshape(x(33:33,c),shape(RAINBL))
DT = x(34,c)
DZS = reshape(x(35:38,c),shape(DZS))
ITIMESTEP = ix(5,c)
SMOIS = reshape(x(39:42,c),shape(SMOIS))
TSLB = reshape(x(43:46,c),shape(TSLB))
SNOW = reshape(x(47:47,c),shape(SNOW))
CANWAT = reshape(x(48:48,c),shape(CANWAT))
CHS = reshape(x(49:49,c),shape(CHS))
CHS2 = reshape(x(50:50,c),shape(CHS2))
CQS2 = reshape(x(51:51,c),shape(CQS2))
SR = reshape(x(52:52,c),shape(SR))
CHKLOWQ = reshape(x(53:53,c),shape(CHKLOWQ))
LAI = reshape(x(54:54,c),shape(LAI))
QZ0 = reshape(x(55:55,c),shape(QZ0))
SH2O = reshape(x(56:59,c),shape(SH2O))
SNOWH = reshape(x(60:60,c),shape(SNOWH))
SNOALB = reshape(x(61:61,c),shape(SNOALB))
SHDMIN = reshape(x(62:62,c),shape(SHDMIN))
SHDMAX = reshape(x(63:63,c),shape(SHDMAX))
SNOTIME = reshape(x(64:64,c),shape(SNOTIME))
ACSNOM = reshape(x(65:65,c),shape(ACSNOM))
ACSNOW = reshape(x(66:66,c),shape(ACSNOW))
SNOPCX = reshape(x(67:67,c),shape(SNOPCX))
POTEVP = reshape(x(68:68,c),shape(POTEVP))
SMCREL = reshape(x(69:72,c),shape(SMCREL))
XICE_THRESHOLD = x(73,c)
USEMONALB = ix(6,c) /= 0
RIB = reshape(x(74:74,c),shape(RIB))
NOAHRES = reshape(x(75:75,c),shape(NOAHRES))
OPT_THCND = ix(7,c)
FLX4_2D = reshape(x(76:76,c),shape(FLX4_2D))
FVB_2D = reshape(x(77:77,c),shape(FVB_2D))
FBUR_2D = reshape(x(78:78,c),shape(FBUR_2D))
FGSN_2D = reshape(x(79:79,c),shape(FGSN_2D))
GSW=0
SWDDIR=0
SWDDIF=0
CPM=0
UST_URB2D=0
FRC_URB2D=0
UTYPE_URB2D=0
call lsm( &
DZ8W=DZ8W, &
QV3D=QV3D, &
P8W3D=P8W3D, &
T3D=T3D, &
TSK=TSK, &
HFX=HFX, &
QFX=QFX, &
LH=LH, &
GRDFLX=GRDFLX, &
QGH=QGH, &
SWDOWN=SWDOWN, &
GLW=GLW, &
SMSTAV=SMSTAV, &
SMSTOT=SMSTOT, &
SFCRUNOFF=SFCRUNOFF, &
UDRUNOFF=UDRUNOFF, &
IVGTYP=IVGTYP, &
ISLTYP=ISLTYP, &
ISURBAN=ISURBAN, &
ISICE=ISICE, &
VEGFRA=VEGFRA, &
ALBEDO=ALBEDO, &
ALBBCK=ALBBCK, &
ZNT=ZNT, &
Z0=Z0, &
TMN=TMN, &
XLAND=XLAND, &
XICE=XICE, &
EMISS=EMISS, &
EMBCK=EMBCK, &
SNOWC=SNOWC, &
QSFC=QSFC, &
RAINBL=RAINBL, &
DT=DT, &
DZS=DZS, &
ITIMESTEP=ITIMESTEP, &
SMOIS=SMOIS, &
TSLB=TSLB, &
SNOW=SNOW, &
CANWAT=CANWAT, &
CHS=CHS, &
CHS2=CHS2, &
CQS2=CQS2, &
SR=SR, &
CHKLOWQ=CHKLOWQ, &
LAI=LAI, &
QZ0=QZ0, &
SH2O=SH2O, &
SNOWH=SNOWH, &
SNOALB=SNOALB, &
SHDMIN=SHDMIN, &
SHDMAX=SHDMAX, &
SNOTIME=SNOTIME, &
ACSNOM=ACSNOM, &
ACSNOW=ACSNOW, &
SNOPCX=SNOPCX, &
POTEVP=POTEVP, &
SMCREL=SMCREL, &
XICE_THRESHOLD=XICE_THRESHOLD, &
USEMONALB=USEMONALB, &
RIB=RIB, &
NOAHRES=NOAHRES, &
OPT_THCND=OPT_THCND, &
FLX4_2D=FLX4_2D, &
FVB_2D=FVB_2D, &
FBUR_2D=FBUR_2D, &
FGSN_2D=FGSN_2D, &
GSW=GSW, &
SWDDIR=SWDDIR, &
SWDDIF=SWDDIF, &
MMINLU='MODIFIED_IGBP_MODIS_NOAH', &
NUM_SOIL_LAYERS=4, &
CPM=CPM, &
ROVCP=0.0, &
MYJ=.FALSE., &
FRPCPN=.FALSE., &
RDLAI2D=.FALSE., &
UA_PHYS=.FALSE., &
IDS=1, &
IDE=1, &
JDS=1, &
JDE=1, &
KDS=1, &
KDE=2, &
IMS=1, &
IME=1, &
JMS=1, &
JME=1, &
KMS=1, &
KME=2, &
ITS=1, &
ITE=1, &
JTS=1, &
JTE=1, &
KTS=1, &
KTE=2, &
SF_URBAN_PHYSICS=0, &
UST_URB2D=UST_URB2D, &
NUM_ROOF_LAYERS=1, &
NUM_WALL_LAYERS=1, &
NUM_ROAD_LAYERS=1, &
JULIAN=1, &
JULYR=1, &
FRC_URB2D=FRC_URB2D, &
UTYPE_URB2D=UTYPE_URB2D, &
NUM_URBAN_NDM=1, &
URBAN_MAP_ZRD=1, &
URBAN_MAP_ZWD=1, &
URBAN_MAP_GD=1, &
URBAN_MAP_ZD=1, &
URBAN_MAP_ZDF=1, &
URBAN_MAP_BD=1, &
URBAN_MAP_WD=1, &
URBAN_MAP_GBD=1, &
URBAN_MAP_FBD=1, &
URBAN_MAP_ZGRD=1, &
NUM_URBAN_HI=1, &
FASDAS=0)
x(1:2,c) = reshape(DZ8W,[size(DZ8W)])
x(3:4,c) = reshape(QV3D,[size(QV3D)])
x(5:6,c) = reshape(P8W3D,[size(P8W3D)])
x(7:8,c) = reshape(T3D,[size(T3D)])
x(9:9,c) = reshape(TSK,[size(TSK)])
x(10:10,c) = reshape(HFX,[size(HFX)])
x(11:11,c) = reshape(QFX,[size(QFX)])
x(12:12,c) = reshape(LH,[size(LH)])
x(13:13,c) = reshape(GRDFLX,[size(GRDFLX)])
x(14:14,c) = reshape(QGH,[size(QGH)])
x(15:15,c) = reshape(SWDOWN,[size(SWDOWN)])
x(16:16,c) = reshape(GLW,[size(GLW)])
x(17:17,c) = reshape(SMSTAV,[size(SMSTAV)])
x(18:18,c) = reshape(SMSTOT,[size(SMSTOT)])
x(19:19,c) = reshape(SFCRUNOFF,[size(SFCRUNOFF)])
x(20:20,c) = reshape(UDRUNOFF,[size(UDRUNOFF)])
ix(1:1,c) = reshape(IVGTYP,[size(IVGTYP)])
ix(2:2,c) = reshape(ISLTYP,[size(ISLTYP)])
ix(3,c) = ISURBAN
ix(4,c) = ISICE
x(21:21,c) = reshape(VEGFRA,[size(VEGFRA)])
x(22:22,c) = reshape(ALBEDO,[size(ALBEDO)])
x(23:23,c) = reshape(ALBBCK,[size(ALBBCK)])
x(24:24,c) = reshape(ZNT,[size(ZNT)])
x(25:25,c) = reshape(Z0,[size(Z0)])
x(26:26,c) = reshape(TMN,[size(TMN)])
x(27:27,c) = reshape(XLAND,[size(XLAND)])
x(28:28,c) = reshape(XICE,[size(XICE)])
x(29:29,c) = reshape(EMISS,[size(EMISS)])
x(30:30,c) = reshape(EMBCK,[size(EMBCK)])
x(31:31,c) = reshape(SNOWC,[size(SNOWC)])
x(32:32,c) = reshape(QSFC,[size(QSFC)])
x(33:33,c) = reshape(RAINBL,[size(RAINBL)])
x(34,c) = DT
x(35:38,c) = reshape(DZS,[size(DZS)])
ix(5,c) = ITIMESTEP
x(39:42,c) = reshape(SMOIS,[size(SMOIS)])
x(43:46,c) = reshape(TSLB,[size(TSLB)])
x(47:47,c) = reshape(SNOW,[size(SNOW)])
x(48:48,c) = reshape(CANWAT,[size(CANWAT)])
x(49:49,c) = reshape(CHS,[size(CHS)])
x(50:50,c) = reshape(CHS2,[size(CHS2)])
x(51:51,c) = reshape(CQS2,[size(CQS2)])
x(52:52,c) = reshape(SR,[size(SR)])
x(53:53,c) = reshape(CHKLOWQ,[size(CHKLOWQ)])
x(54:54,c) = reshape(LAI,[size(LAI)])
x(55:55,c) = reshape(QZ0,[size(QZ0)])
x(56:59,c) = reshape(SH2O,[size(SH2O)])
x(60:60,c) = reshape(SNOWH,[size(SNOWH)])
x(61:61,c) = reshape(SNOALB,[size(SNOALB)])
x(62:62,c) = reshape(SHDMIN,[size(SHDMIN)])
x(63:63,c) = reshape(SHDMAX,[size(SHDMAX)])
x(64:64,c) = reshape(SNOTIME,[size(SNOTIME)])
x(65:65,c) = reshape(ACSNOM,[size(ACSNOM)])
x(66:66,c) = reshape(ACSNOW,[size(ACSNOW)])
x(67:67,c) = reshape(SNOPCX,[size(SNOPCX)])
x(68:68,c) = reshape(POTEVP,[size(POTEVP)])
x(69:72,c) = reshape(SMCREL,[size(SMCREL)])
x(73,c) = XICE_THRESHOLD
ix(6,c) = merge(1,0,USEMONALB)
x(74:74,c) = reshape(RIB,[size(RIB)])
x(75:75,c) = reshape(NOAHRES,[size(NOAHRES)])
ix(7,c) = OPT_THCND
x(76:76,c) = reshape(FLX4_2D,[size(FLX4_2D)])
x(77:77,c) = reshape(FVB_2D,[size(FVB_2D)])
x(78:78,c) = reshape(FBUR_2D,[size(FBUR_2D)])
x(79:79,c) = reshape(FGSN_2D,[size(FGSN_2D)])
enddo
end subroutine
