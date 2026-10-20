! MPAS: module_ra_rrtmg_lw.F:3068-3091, immutable module DATA.
program diffusivity
use rrtmg_lw_rtrnmc,only:a0,a1,a2,wtdiff,rec_6
use lw_binio
implicit none
character(1024)::outdir
call get_command_argument(1,outdir)
call bio_open(trim(outdir)//'/native-diffusivity.bin')
call wr_r1('rrtmg_lw_rtrnmc/a0',a0)
call wr_r1('rrtmg_lw_rtrnmc/a1',a1)
call wr_r1('rrtmg_lw_rtrnmc/a2',a2)
call wr_r0('rrtmg_lw_rtrnmc/wtdiff',wtdiff)
call wr_r0('rrtmg_lw_rtrnmc/rec_6',rec_6)
call bio_close()
end program diffusivity
