! Full compiled GFL module, instrumented by r1. No native algebra changed.
! RRFS-MPAS 1d6e370e GFL/module_cu_gfl_deep.F90:2639-2794.
program saved_state_probe
 use iso_fortran_env, only: int32,real32
 use mpas_oracle_dump
 use module_cu_gfl_deep, only: fct1d3
 implicit none
 integer,parameter::n=8,ktop=3
 integer::sample,event,k
 real(real32)::z(n),tracr(n),massflx(n),trflx(n),out(n),dt,g
 character(len=*),parameter::key='fct_saved_state_probe'
 dt=30.;g=9.80616
 do sample=1,3
  do k=1,n
   z(k)=1000.-100.*real(k-1,real32)
  enddo
  if(sample==2)z(2)=z(1)
  tracr=.001;massflx=.00001;trflx=.00000001;out=73.25
  call oracle_set_context([1_int32],sample,0)
  call oracle_begin(key,event)
  call oracle_scope_begin(key,event)
  call oracle_write(key,'pre','z',z,event,cell_axis=0,level_axis=1)
  call oracle_write(key,'pre','tracr',tracr,event,cell_axis=0,level_axis=1)
  call oracle_write(key,'pre','massflx',massflx,event,cell_axis=0,level_axis=1)
  call oracle_write(key,'pre','trflx',trflx,event,cell_axis=0,level_axis=1)
  call oracle_write(key,'pre','out',out,event,cell_axis=0,level_axis=1)
  call fct1d3(ktop,n,dt,z,tracr,massflx,trflx,out,g)
  call oracle_write(key,'post','out',out,event,cell_axis=0,level_axis=1)
  call oracle_scope_end()
 enddo
end program
