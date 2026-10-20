// MPAS module_cu_gf.mpas.F:154-198; JTF=0 skips forcing reads.
// MPAS module_cu_gf.mpas.F:54-453; test-only optional argument ABI.
__device__ void gf_empty_forcing_column(int nz,int mask,GFReal* lev,GFReal* surface,int* isurface,
 GFReal* outlev,GFReal* outsurface,int* outint,int* control,
 GFReal* constants,GFWorkspace &workspace){
 int one=1,two=2,nplus=nz+1;
 cu_grell_freitas(workspace,control,&surface[7],&surface[5],&surface[6],
 lev+0*nz,lev+1*nz,lev+3*nz,lev+2*nz,lev+4*nz,lev+5*nz,lev+6*nz,lev+7*nz,
 lev+8*nz,lev+9*nz,surface+3,surface+0,surface+1,surface+2,surface+4,
 (mask&8)?lev+11*nz:nullptr,(mask&4)?lev+10*nz:nullptr,lev+13*nz,lev+14*nz,lev+12*nz,isurface,
 constants+0,constants+1,constants+2,constants+3,
 control+1,control+2,control+3,
 outsurface+0,outsurface+1,outint+0,outint+1,outint+2,outsurface+2,outsurface+3,
 outsurface+4,outsurface+5,(mask&1)?outlev+4*nz:nullptr,(mask&2)?outlev+5*nz:nullptr,
 outlev+0*nz,outlev+1*nz,outlev+2*nz,outlev+3*nz,
 &one,&one,&one,&one,&one,&nplus,&one,&two,&one,&one,&one,&nplus,
 &one,&one,&one,&one,&one,&nz);
}
#ifdef GF_HOST_WITNESS
extern "C" void gf_host_empty_forcing(int nz,int mask,float* lev,float* surface,int* isurface,
 float* outlev,float* outsurface,int* outint,int* control,float* constants){
 unsigned long long size=4096ULL*(nz+1)+65536ULL;char* scratch=(char*)malloc(size);
 GFWorkspace ws{scratch,size,0,0,0};
 gf_empty_forcing_column(nz,mask,(GFReal*)lev,(GFReal*)surface,isurface,(GFReal*)outlev,
  (GFReal*)outsurface,outint,control,(GFReal*)constants,ws);free(scratch);
}
#else
extern "C" __global__ void gf_aux_empty_forcing(int nz,int mask,float* lev,float* surface,int* isurface,
 float* outlev,float* outsurface,int* outint,int* control,float* constants,char* scratch){
 GFWorkspace ws{scratch,4096ULL*(nz+1)+65536ULL,0,0,0};
 gf_empty_forcing_column(nz,mask,(GFReal*)lev,(GFReal*)surface,isurface,(GFReal*)outlev,
  (GFReal*)outsurface,outint,control,(GFReal*)constants,ws);
}
#endif
