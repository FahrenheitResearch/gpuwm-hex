! Genuine MPAS module_ra_rrtmg_sw.F:2541-2821, limiting optical properties.
program oracle_reftra
 use rrtmg_sw_reftra,only:reftra_sw
 use mpas_atmphys_rrtmg_swinit,only:rrtmg_initsw_forMPAS
 use mpas_derived_types,only:dm_info
 implicit none
 type(dm_info)::dminfo
 integer,parameter::nl=336
 real,parameter::gg(7)=[0.,.5,.75,.9,1.,-1.,.3333333]
 real,parameter::ww(6)=[0.,.5,.75,.999999,1.,1.e-10]
 real,parameter::tt(8)=[0.,1.e-12,1.e-5,.009,.06,1.,50.,1000.]
 real,parameter::mu(8)=[1.e-10,.1,.3333333,.5,.7,1.,.57735026,.999999]
 real::g(nl),w(nl),tau(nl),r(nl),rd(nl),t(nl),td(nl)
 logical::check(nl)
 integer::u,a,b,c,k,i
 call rrtmg_initsw_forMPAS(dminfo)
 k=0
 do a=1,7
 do b=1,6
 do c=1,8
  k=k+1;g(k)=gg(a);w(k)=ww(b);tau(k)=tt(c);check(k)=mod(k,13)/=0
 enddo
 enddo
 enddo
 open(newunit=u,file='reftra.bin',access='stream',form='unformatted',status='replace')
 write(u)nl,8
 write(u)g,tau,w
 write(u)merge(1,0,check)
 do i=1,8
  write(u)mu(i)
  call reftra_sw(nl,check,g,mu(i),tau,w,r,rd,t,td)
  write(u)r,rd,t,td
 enddo
 close(u)
end program oracle_reftra
