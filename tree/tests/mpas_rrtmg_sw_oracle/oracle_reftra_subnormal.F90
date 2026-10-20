! Pristine MPAS module_ra_rrtmg_sw.F:2541-2821 with subnormal inputs.
program oracle_reftra_subnormal
 use rrtmg_sw_reftra,only:reftra_sw
 use mpas_atmphys_rrtmg_swinit,only:rrtmg_initsw_forMPAS
 use mpas_derived_types,only:dm_info
 implicit none
 type(dm_info)::dminfo
 integer,parameter::nl=16,ncase=8
 integer,parameter::bits(8)=[1,2,255,65535,1048576,4194304,8388607,8388608]
 real,parameter::mu(ncase)=[.1,.3333333,.5,.7,1.,.57735026,.999999,.01]
 real::g(nl),w(nl),tau(nl),r(nl),rd(nl),t(nl),td(nl)
 logical::check(nl)
 integer::u,k,i
 call rrtmg_initsw_forMPAS(dminfo)
 check=.true.
 do k=1,8
  g(k)=.5;w(k)=transfer(bits(k),0.);tau(k)=1.
  g(k+8)=.5;w(k+8)=.5;tau(k+8)=transfer(bits(k),0.)
 enddo
 open(newunit=u,file='reftra-subnormal.bin',access='stream',form='unformatted',status='replace')
 write(u)nl,ncase
 write(u)g,tau,w
 write(u)merge(1,0,check)
 do i=1,ncase
  write(u)mu(i)
  call reftra_sw(nl,check,g,mu(i),tau,w,r,rd,t,td)
  write(u)r,rd,t,td
 enddo
 close(u)
end program oracle_reftra_subnormal
