// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:1360-1433
// Each thread owns one receiving (cell,level). Source neighbor iteration order is retained.
extern "C" __global__ void gfl_spread(int nc,int owned,int nz,int me,int me2,int sub3d,
 const int* mask,const int* ne,const int* coc,const int* n2,const int* coc2,
 const float* sigma,const float* far,const float* sub,float* tend){
 int ix=blockIdx.x*blockDim.x+threadIdx.x;
 if(ix>=owned*nz)return;
 int c=ix/nz,k=ix%nz;
 if(sub3d>0 && mask[c]>4)return;
 for(int f=0;f<4;f++){
  int dst=(f*nc+c)*nz+k;
  float val=tend[dst];
  if(sub3d<=0){tend[dst]=__fadd_rn(val,sub[dst]);continue;}
  val=__fadd_rn(val,__fmul_rn(__fsub_rn(1.0f,sigma[c]),sub[dst]));
  for(int i=0;i<ne[c];i++){
   int other=coc[c*me+i]-1;
   if(sigma[other]==0.0f)continue;
   float add=__fdiv_rn(__fmul_rn(__fsub_rn(sigma[other],far[other]),sub[(f*nc+other)*nz+k]),(float)ne[c]);
   val=__fadd_rn(val,add);
  }
  for(int j=0;j<n2[c];j++){
   int other=coc2[c*me2+j]-1;
   float add=__fdiv_rn(__fmul_rn(far[other],sub[(f*nc+other)*nz+k]),(float)n2[c]);
   val=__fadd_rn(val,add);
  }
  tend[dst]=val;
 }
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:1481-1584
// Two-ring construction: unique candidates appended in native edge order, owned cells only.
extern "C" __global__ void gfl_cells_on_cells(int owned,int me,int me2,const int* ne,
 const int* coc,int* n2,int* coc2){
 int c=blockIdx.x*blockDim.x+threadIdx.x;
 if(c>=owned)return;
 int group[50];
 for(int k=0;k<50;k++)group[k]=0;
 int count=1+ne[c],inner=count;
 group[0]=c+1;
 for(int i=0;i<ne[c];i++)group[i+1]=coc[c*me+i];
 for(int i=0;i<ne[c];i++){
  int other=coc[c*me+i]-1;
  for(int j=0;j<ne[other];j++){
   int candidate=coc[other*me+j];bool absent=true;
   for(int k=0;k<count;k++)if(group[k]==candidate)absent=false;
   if(absent){group[count]=candidate;count++;}
  }
 }
 int outer=count-inner;
 if(outer>me2){asm("trap;");return;} // Native MPAS_LOG_CRIT, mpas_atm_core.F:1549-1554
 for(int i=0;i<outer;i++)coc2[c*me2+i]=group[i+inner];
 n2[c]=outer;
}

// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:1025-1057
extern "C" __global__ void gfl_driver_dx(int n,float len_disp,const float* density,float* dx){
 int c=blockIdx.x*blockDim.x+threadIdx.x;
 if(c<n)dx[c]=(GFReal(len_disp)/gf_pow(GFReal(density[c]),GFReal(0.25f))).value;
}
// RRFS-MPAS 1d6e370e MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_convection.F:1064-1067
extern "C" __global__ void gfl_driver_radiation(int n,int nz,int nc,const float* lw,const float* sw,float* result){
 int ix=blockIdx.x*blockDim.x+threadIdx.x;
 if(ix<n*nz){int c=ix/nz,k=ix%nz;result[c*(nz+1)+k]=__fadd_rn(lw[k*nc+c],sw[k*nc+c]);}
}
