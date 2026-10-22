// SPDX-License-Identifier: Apache-2.0
// Backward Euler finite-volume diffusion, lower-boundary deposition, settling.
extern "C" __global__ void chemistry_column(
 int ns,int nz,int nc,int solved,float dt,const float *rho,const float *z,
 const float *kh,const float *vd,const float *vs,const float *source,
 float *q,double *upper,double *rhs,double *dry,double *settled,double *emitted,int *invalid) {
 int sc=blockIdx.x*blockDim.x+threadIdx.x;if(sc>=ns*nc)return;
 int s=sc/nc,c=sc%nc;if(c>=solved)return;
 int plane=nz*nc;
 for(int k=0;k<nz;++k){
  int i=k*nc+c,j=s*plane+i;
  double dz=double(z[(k+1)*nc+c])-z[i],r=rho[i],h=kh[i],v=q[j];
  if(!(dz>0&&r>0&&h>=0&&v>=0&&isfinite(dz)&&isfinite(r)&&isfinite(h)&&isfinite(v))){
   atomicExch(invalid,1);return;}
  double cb=0,ct=0;
  if(k>0){double d0=double(z[i])-z[(k-1)*nc+c];
   cb=.5*(r+rho[i-nc])*double(kh[i-nc])/(.5*(dz+d0));}
  if(k<nz-1){double d1=double(z[(k+2)*nc+c])-z[(k+1)*nc+c];
   ct=.5*(r+rho[i+nc])*h/(.5*(dz+d1));}
  double a=-dt*cb,b=r*dz+dt*(cb+ct+r*vs[s]+(k==0?r*vd[s]:0.));
  double u=k<nz-1?-dt*(ct+rho[i+nc]*vs[s]):0.;
  double d=r*dz*v+(k==0?dt*double(source[s*nc+c]):0.);
  if(k){b-=a*upper[j-nc];d-=a*rhs[j-nc];}
  upper[j]=u/b;rhs[j]=d/b;
 }
 double next=0;
 for(int k=nz-1;k>=0;--k){int j=s*plane+k*nc+c;
  double v=rhs[j]-upper[j]*next;
  if(!(v>=0&&isfinite(v))){atomicExch(invalid,1);return;}
  q[j]=float(v);next=v;
 }
 emitted[sc]+=dt*double(source[sc]);
 dry[sc]+=dt*double(rho[c])*vd[s]*next;
 settled[sc]+=dt*double(rho[c])*vs[s]*next;
}
extern "C" __global__ void chemistry_products(
 int ns,int nz,int nc,int solved,const float *rho,const float *z,const float *q,
 const float *ext,float *column,float *surface,float *aod){
 int sc=blockIdx.x*blockDim.x+threadIdx.x;if(sc>=ns*nc)return;
 int s=sc/nc,c=sc%nc;if(c>=solved){column[sc]=surface[sc]=aod[sc]=0;return;}
 double mass=0;
 for(int k=0;k<nz;++k){int i=k*nc+c;
  mass+=double(q[s*nz*nc+i])*rho[i]*(double(z[(k+1)*nc+c])-z[i]);}
 column[sc]=float(mass);surface[sc]=q[s*nz*nc+c]*rho[c]*1.e9f;aod[sc]=float(mass*ext[s]);
}
