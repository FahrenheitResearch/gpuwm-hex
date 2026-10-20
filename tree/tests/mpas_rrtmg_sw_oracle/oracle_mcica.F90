! Genuine MPAS module_ra_rrtmg_sw.F:1403-2012, including MT and KISS.
program oracle_mcica
 use mcica_subcol_gen_sw,only:mcica_subcol_sw
 use mpas_atmphys_rrtmg_swinit,only:rrtmg_initsw_forMPAS
 use mpas_derived_types,only:dm_info
 implicit none
 type(dm_info)::dminfo
 integer,parameter::nc=2,ng=112
#ifdef SINGLE_LAYER_ORACLE
 integer,parameter::nl=1,first_rng=1
#else
 integer,parameter::nl=12,first_rng=0
#endif
 integer::u,g,c,k,overlap,rng,idcor,seed,juldat,actual_rng,b
 real::play(nc,nl),cf(nc,nl),ci(nc,nl),cl(nc,nl),cs(nc,nl),height(nc,nl),rei(nc,nl),rel(nc,nl),res(nc,nl),lat
 real::tau(14,nc,nl),ssa(14,nc,nl),asy(14,nc,nl),fsf(14,nc,nl)
 real::ocf(ng,nc,nl),oci(ng,nc,nl),ocl(ng,nc,nl),ocs(ng,nc,nl)
 real::otau(ng,nc,nl),ossa(ng,nc,nl),oasy(ng,nc,nl),ofsf(ng,nc,nl),orei(nc,nl),orel(nc,nl),ores(nc,nl)
 call rrtmg_initsw_forMPAS(dminfo)
 seed=7;juldat=279;lat=-41.25
 do k=1,nl
 do c=1,nc
  play(c,k)=1000.125/(1.+real(k-1)*.23)+real(c)*.07125
  cf(c,k)=real(mod(k+c,5))*.2
  ci(c,k)=real(mod(k,3))*1.3;cl(c,k)=real(mod(k,4))*2.7;cs(c,k)=real(mod(k,5))*3.8
  height(c,k)=250.+real(k*k)*235.7+real(c)*.34
  rei(c,k)=30.+real(k);rel(c,k)=10.+real(k);res(c,k)=50.+real(k)
  do b=1,14
   tau(b,c,k)=real(b)*.013+real(k)*.033
   ssa(b,c,k)=.7+real(b)*.01;asy(b,c,k)=.5+real(b)*.01;fsf(b,c,k)=.04+real(b)*.02
  enddo
 enddo
 enddo
 if(nl==1)then
  open(newunit=u,file='mcica-single.bin',access='stream',form='unformatted',status='replace')
 else
  open(newunit=u,file='mcica.bin',access='stream',form='unformatted',status='replace')
 endif
 write(u)nc,nl,ng,(3-first_rng)*10
 write(u)seed,juldat,lat
 write(u)play,cf,ci,cl,cs,height,rei,rel,res,tau,ssa,asy,fsf
 do rng=first_rng,2
 do overlap=1,5
 do idcor=0,1
  actual_rng=rng
  write(u)overlap,rng,idcor
  call mcica_subcol_sw(1,nc,nl,overlap,seed,actual_rng,play,cf,ci,cl,cs,rei,rel,res,tau,ssa,asy,fsf, &
    height,idcor,juldat,lat,ocf,oci,ocl,ocs,orei,orel,ores,otau,ossa,oasy,ofsf)
  write(u)actual_rng
  write(u)ocf,oci,ocl,ocs,otau,ossa,oasy,ofsf,orei,orel,ores
 enddo
 enddo
 enddo
 close(u)
end program oracle_mcica
