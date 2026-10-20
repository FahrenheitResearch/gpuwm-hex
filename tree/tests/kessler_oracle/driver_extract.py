"""Extract actual MPAS Kessler driver arithmetic into a standalone C bridge.

Only pool access/allocation is replaced. The [level,cell] Fortran declarations
are transposed for cells-fastest test storage; every arithmetic expression is
copied without modification. The one WRF j dimension is retained for locals.
"""
from pathlib import Path
import re

FIELDS = ("theta_m", "qv", "qc", "qr", "rho_zz", "zz", "exner", "exner_b",
          "rtheta_b", "rtheta_p", "pressure_p", "pressure_b", "rt_diabatic_tend", "dtheta_dt_mp", "relhum")
SURFACE = ("surface_pressure", "tend_sfc_pressure", "rainnc", "rainncv", "precipw")
DRIVER_FIELDS = (*FIELDS, "zgrid", *SURFACE, "i_rainnc")


def extract(directory, reference):
    root = Path(reference) / "src/core_atmosphere/physics"
    interface = (root / "mpas_atmphys_interface.F").read_text().splitlines()
    driver = (root / "mpas_atmphys_driver_microphysics.F").read_text().splitlines()
    thompson = (root / "physics_wrf/module_mp_thompson.F").read_text().splitlines()

    def window(lines, first, last):
        # MPAS state [k,i] -> test contiguous [i,k]; local WRF [i,k,j] stays.
        out = "\n".join(lines[first-1:last])
        return re.sub(r"\((k(?:\+1)?|[123]),i\)", r"(i,\1)", out)

    src = "subroutine driver_oracle( &\n" + ", &\n".join(DRIVER_FIELDS) + ", &\nv,dt_microp,dt_dyn,mp_top_level,config_bucket_rainnc,accumulate,diags,nz,nc) bind(C)\n"
    src += "use iso_c_binding\nuse module_mp_kessler\nimplicit none\n"
    src += "integer,parameter:: RKIND=c_float\ninteger(c_int),value:: nz,nc,mp_top_level,accumulate,diags\n"
    src += "real(c_float),value:: dt_microp,dt_dyn,config_bucket_rainnc\nreal(c_float):: v(13)\n"
    src += "\n".join(f"real(c_float):: {f}(nc,nz)" for f in FIELDS) + "\n"
    src += "real(c_float):: zgrid(nc,nz+1)\n" + "\n".join(f"real(c_float):: {f}(nc)" for f in SURFACE) + "\n"
    src += "integer(c_int):: i_rainnc(nc)\n"
    locals_ = ("th_p", "qv_p", "qc_p", "qr_p", "rho_p", "pi_p", "pres_p", "z_p", "dz_p", "w_p")
    src += "\n".join(f"real(c_float):: {f}(nc,nz,1)" for f in locals_) + "\n"
    src += "real(c_float):: rainnc_p(nc,1),rainncv_p(nc,1),w(nc,nz)\n"
    src += "real(c_float):: R_v,R_d,P0,rcv,gravity,rvord,rho_a,rho1,rho2,tem1,tem2\n"
    src += "integer:: i,j,k,its,ite,jts,jte,kts,kte\nlogical:: l_acrain\n"
    src += "real(c_float):: tempc,p1d(nz),t1d(nz),qv1d(nz),qvs1d(nz)\n"
    src += "its=1\nite=nc\njts=1\njte=1\nkts=1\nkte=nz\nw=0\n"
    src += "R_d=v(10)\nR_v=461.6_c_float\nrvord=v(9)\nP0=v(11)\nrcv=v(12)\ngravity=v(13)\nl_acrain=accumulate/=0\n"
    src += window(driver, 531, 540) + "\n" # precipitation reset
    src += window(interface, 657, 675) + "\n"
    src += "call kessler(th_p,qv_p,qc_p,qr_p,rho_p,pi_p,dt_microp,z_p,v(1),v(2),v(3),v(4), &\n"
    src += "v(5),v(6),v(7),v(8),dz_p,rainnc_p,rainncv_p,1,nc,1,1,1,nz,1,nc,1,1,1,nz,1,nc,1,1,1,nz)\n"
    src += "if(diags/=0) then\n" + window(driver,830,843) + "\nendif\n"
    src += window(driver, 605, 632) + "\n"
    src += window(interface, 903, 964) + "\ncontains\n" + window(thompson,4597,4627) + "\n" + window(thompson,4632,4658) + "\nend subroutine\n"
    Path(directory, "driver_oracle.f90").write_text(src)
