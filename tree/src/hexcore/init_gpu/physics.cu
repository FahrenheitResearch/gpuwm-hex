// Port of rw-mpas/init/{vinterp,dynamics,soil}.rs. Host operation order retained.
#ifndef MAX_FG
#define MAX_FG 128
#endif
#ifndef MAX_Z
#define MAX_Z 128
#endif
__device__ float vi(float x,const float*z,const float*v,int n,int mode,int*error){
 if(x<z[0]){if(mode==0)return v[0];if(mode==2)return v[0]-(x-z[0])*0.0065f;
   float slope=(v[1]-v[0])/(z[1]-z[0]);return v[0]+slope*(x-z[0]);}
 if(x>=z[n-1]){if(mode==0)return v[n-1];if(mode==2){atomicExch(error,1);return v[0];}
   float slope=(v[n-1]-v[n-2])/(z[n-1]-z[n-2]);return v[n-1]+slope*(x-z[n-1]);}
 for(int k=0;k<n-1;k++)if(x>=z[k]&&x<z[k+1]){
   float wm=(z[k+1]-x)/(z[k+1]-z[k]),wp=(x-z[k])/(z[k+1]-z[k]);return wm*v[k]+wp*v[k+1];}
 return v[n-1];
}
__device__ void sort_index(float*z,int*ix,int n){
 for(int i=0;i<n;i++)ix[i]=i;
 for(int i=1;i<n;i++){float key=z[i];int id=ix[i],j=i-1;
   while(j>=0&&z[j]>key){z[j+1]=z[j];ix[j+1]=ix[j];j--;}z[j+1]=key;ix[j+1]=id;}
}
extern "C" __global__ void vertical_cells(const float*t,const float*rh,const float*sh,const float*p,const float*z,
 const float*ps,const float*soilz,const float*levels,const float*zgrid,const int*isobaric,
 int nc,int nb,int stride,int nf,int nz,int mode,
 float*to,float*rho,float*sho,float*po,float*pso,int*error){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc*nb)return;int b=c/nc,cell=c%nc;
 float coord[MAX_FG],values[MAX_FG];int ix[MAX_FG];
 for(int k=0;k<nf;k++)coord[k]=levels[k]==200100.0f?99999.0f:z[(long long)c*stride+k];
 sort_index(coord,ix,nf);const float*zg=zgrid+(long long)cell*(nz+1);
 for(int f=0;f<4;f++){
   for(int k=0;k<nf;k++){int src=ix[k];long long off=(long long)c*stride+src;
     if(f==0)values[k]=t[off];else if(f==1)values[k]=rh[off];else if(f==2)values[k]=fmaxf(sh[off],0.0f);
     else {float val=levels[src]==200100.0f?ps[c]:isobaric[b]?levels[src]:p[off];values[k]=gfk_log(val);}}
   for(int k=0;k<nz;k++){float target=0.5f*(zg[k]+zg[k+1]);
     float value=vi(target,coord,values,nf-1,f==0?mode:f==3?1:0,error);long long dest=(long long)c*nz+k;
     if(f==0)to[dest]=value;else if(f==1)rho[dest]=value;else if(f==2)sho[dest]=value;else po[dest]=gfk_exp(value);}
   if(f==3)pso[c]=gfk_exp(vi(zg[0],coord,values,nf,1,error));
 }
}
extern "C" __global__ void vertical_edges(const float*u,const float*v,const float*z,const float*levels,
 const float*angle,const int*coe,const float*zgrid,int nc,int ne,int nb,int stride,int nf,int nz,float*out,int*error){
 int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=ne*nb)return;int b=e/ne,edge=e%ne;
 int c1=min(max(coe[edge*2],1)-1,nc-1),c2=min(max(coe[edge*2+1],1)-1,nc-1);
 float coord[MAX_FG],values[MAX_FG],temp[MAX_FG];int ix[MAX_FG];
 float ca=glibc_cosf(angle[edge]),sa=glibc_sinf(angle[edge]);
 for(int k=0;k<nf;k++){
   coord[k]=levels[k]==200100.0f?99999.0f:0.5f*(z[((long long)b*nc+c1)*stride+k]+z[((long long)b*nc+c2)*stride+k]);
   temp[k]=ca*u[(long long)e*stride+k]+sa*v[(long long)e*stride+k];
 }
 sort_index(coord,ix,nf);for(int k=0;k<nf;k++)values[k]=temp[ix[k]];
 const float*z1=zgrid+(long long)c1*(nz+1),*z2=zgrid+(long long)c2*(nz+1);
 for(int k=0;k<nz;k++){float target=0.25f*(z1[k]+z1[k+1]+z2[k]+z2[k+1]);out[(long long)e*nz+k]=vi(target,coord,values,nf-1,0,error);}
}
__device__ float rslf(float p,float t){
 const float coeff[9]={0.611583699e03f,0.444606896e02f,0.143177157e01f,0.264224321e-1f,0.299291081e-3f,0.203154182e-5f,0.702620698e-8f,0.379534310e-11f,-0.321582393e-13f};
 float x=fmaxf(t-273.16f,-80.0f),esl=coeff[8];for(int k=7;k>=0;k--)esl=coeff[k]+x*esl;
 esl=fminf(esl,p*0.15f);return 0.622f*esl/(p-esl);
}
extern "C" __global__ void dynamics_cells(const float*t,const float*p,const float*rh,const float*sh,
 const float*zgrid,const float*zz,const float*fzm,const float*fzp,const float*dzu,float rdzw0,
 const int*use_sh,int nc,int nb,int nz,float late,
 float*qout,float*thetaout,float*rhoout,float*rhobaseout,float*thetabout,float*rhozout,
 float*precipout,float*surfaceout,float*relhum,int*iterations){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc*nb)return;int b=c/nc,cell=c%nc;
 const float gravity=9.80616f,rgas=287.0f,cp=7.0f*287.0f/2.0f,cv=cp-287.0f,rvord=461.6f/287.0f,p0=1.0e5f,t0b=250.0f;
 const float*zg=zgrid+(long long)cell*(nz+1),*metric=zz+(long long)cell*nz;
 float q[MAX_Z],theta[MAX_Z],rhoz[MAX_Z],pb[MAX_Z],pp[MAX_Z],rp[MAX_Z],rb[MAX_Z],pressure[MAX_Z];
 float precip=0.0f;
 for(int k=0;k<nz;k++){
   long long off=(long long)c*nz+k;pressure[k]=p[off];
   q[k]=use_sh[b]?sh[off]/(1.0f-sh[off]):0.01f*rslf(pressure[k],t[off])*rh[off];
   float exner=gfk_pow(__fdiv_rn(pressure[k],p0),rgas/cp);
   theta[k]=t[off]*gfk_pow(p0/pressure[k],rgas/cp);
   rhoz[k]=__fdiv_rn(pressure[k],rgas)/(exner*theta[k]*(1.0f+(rvord-1.0f)*q[k]));rhoz[k]/=1.0f+q[k];
 }
 for(int k=0;k<nz;k++)precip+=rhoz[k]*q[k]*(zg[k+1]-zg[k]);
 for(int k=0;k<nz;k++){
   float ztemp=0.5f*(zg[k+1]+zg[k]);pb[k]=p0*gfk_exp(__fdiv_rn(-gravity*ztemp,rgas*t0b));
   float exnerbase=gfk_pow(__fdiv_rn(pb[k],p0),rgas/cp);rb[k]=__fdiv_rn(pb[k],rgas*t0b);
   thetabout[(long long)c*nz+k]=t0b/exnerbase;
 }
 for(int k=0;k<nz;k++){rb[k]/=metric[k];rhoz[k]/=metric[k];pp[k]=pressure[k]-pb[k];rp[k]=rhoz[k]-rb[k];}
 rhoz[0]=gfk_pow(__fdiv_rn(pressure[0],p0),cv/cp)*(p0/rgas)/(theta[0]*(1.0f+late*q[0]))/metric[0];rp[0]=rhoz[0]-rb[0];
 iterations[(long long)c*nz]=0;
 for(int k=1;k<nz;k++){
   int it=0;float check=2.0f*0.0001f;
   while(it<30&&check>0.0001f){
     check=pp[k];pp[k]=pp[k-1]-(fzm[k]*rp[k]+fzp[k]*rp[k-1])*gravity*dzu[k]
       -(fzm[k]*rhoz[k]*q[k]+fzp[k]*rhoz[k-1]*q[k-1])*gravity*dzu[k];
     pressure[k]=pp[k]+pb[k];float exner=gfk_pow(__fdiv_rn(pressure[k],p0),rgas/cp);
     rhoz[k]=__fdiv_rn(pressure[k],rgas)/(exner*theta[k]*(1.0f+late*q[k]))/metric[k];rp[k]=rhoz[k]-rb[k];
     check=fabsf(check-pp[k]);it++;
   }
   iterations[(long long)c*nz+k]=it;
 }
 float surface=0.5f*gravity/rdzw0*(1.25f*rhoz[0]*(1.0f+q[0])-0.25f*rhoz[1]*(1.0f+q[1]))+pp[0]+pb[0];
 for(int k=0;k<nz;k++){
   long long off=(long long)c*nz+k;
   float theta_m=theta[k]*(1.0f+late*q[k]);
   thetaout[off]=theta_m/(1.0f+late*q[k]);rhoout[off]=rhoz[k]*metric[k];qout[off]=q[k];
   rhobaseout[off]=rb[k];rhozout[off]=rhoz[k];float temp=t[off],humidity=rh[off];
   if(temp<=273.15f){
     float eis=0.01f*gfk_exp(9.550426f-(5723.265f/temp)+(3.53068f*gfk_log(temp))-(0.00728332f*temp));
     float ews=6.112f*gfk_exp(17.67f*(temp-273.15f)/((temp-273.15f)+243.5f));
     float r1=eis;if(temp>253.15f){float f=__fdiv_rn(273.15f-temp,20.0f);r1=f*eis+(1.0f-f)*ews;}
     r1=fmaxf(r1,1.0e-12f);ews=fmaxf(ews,0.0f);humidity=fminf(fmaxf(ews/r1*humidity,0.0f),100.0f);
   }
   relhum[off]=humidity;
 }
 precipout[c]=precip;surfaceout[c]=surface;
}
__device__ float ruval(const float*u,const float*rhoz,const int*coe,int nc,int ne,int nz,int b,int e,int k){
 int c1=min(max(coe[e*2],1)-1,nc-1),c2=min(max(coe[e*2+1],1)-1,nc-1);
 return u[((long long)b*ne+e)*nz+k]*0.5f*(rhoz[((long long)b*nc+c1)*nz+k]+rhoz[((long long)b*nc+c2)*nz+k]);
}
extern "C" __global__ void w_cells(const float*u,const float*rhoz,const float*zz,const float*fzm,const float*fzp,
 const float*zb,const float*zb3,const int*coe,const int*neoc,const int*eoc,
 int nc,int ne,int nb,int nz,int maxedges,int order,float coef,float*out){
 long long ix=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(ix>=(long long)nb*nc*(nz+1))return;
 int k=ix%(nz+1),c=ix/(nz+1),cell=c%nc,b=c/nc;if(k==0||k==nz){out[ix]=0.0f;return;}
 float rw=0.0f;long long ck=(long long)cell*nz+k;
 for(int i=0;i<neoc[cell];i++){
   int e=eoc[(long long)cell*maxedges+i];if(e==0||e>ne)continue;e--;
   bool side=coe[e*2]==cell+1;float ru=ruval(u,rhoz,coe,nc,ne,nz,b,e,k);
   float flux=fzm[k]*ru+fzp[k]*ruval(u,rhoz,coe,nc,ne,nz,b,e,k-1);
   float metric=fzm[k]*zz[ck]+fzp[k]*zz[ck-1];long long met=((long long)e*2+(side?0:1))*(nz+1)+k;
   if(side)rw-=metric*zb[met]*flux;else rw+=metric*zb[met]*flux;
   if(order==3){float sign=ru>=0.0f?1.0f:-1.0f;if(side)rw+=sign*coef*metric*zb3[met]*flux;else rw-=sign*coef*metric*zb3[met]*flux;}
 }
 long long dest=(long long)c*nz+k;out[ix]=rw/(fzp[k]*rhoz[dest-1]+fzm[k]*rhoz[dest]);
}
// Soil/surface fields are computed per cell, for the whole member batch.
extern "C" __global__ void soil_cells(const int*land,const float*ter,const float*soiltemp,const float*albedo,
 const float*green,const float*snoalb0,const int*veg0,const int*soil0,int isice,
 const float*soilz,const float*skin0,const float*snow0,const float*ice0,const float*st0,const float*sm0,
 const float*depth,const float*t2,const float*rh2,const float*ps,
 int nc,int nb,int ns,int deep_corrected,int fractional,float tskthreshold,
 int m1,int m2,float w1,float w2,float span,
 float*skin,float*tmn,float*tslb,float*smois,float*sh2o,float*ice,float*seaice,float*snow,float*snowc,float*snowh,
 float*xland,float*alb,float*veg,float*snoalb,int*vegtype,int*soiltype,float*shdmin,float*shdmax,
 float*dzs,float*zs,float*q2,int*errors,int*counts){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc*nb)return;int cell=c%nc,b=c/nc;
 float temp=skin0[c],deep=temp,st[MAX_FG],sm[MAX_FG];int repairs=0,badtemp=0;
 float shift=land[cell]==1?0.0065f*(ter[cell]-soilz[c]):0.0f;
 if(land[cell]==1){deep=soiltemp[cell]-0.0065f*ter[cell];temp-=shift;}
 for(int k=0;k<ns;k++){
   st[k]=st0[(long long)c*ns+k];sm[k]=sm0[(long long)c*ns+k];if(land[cell]==1){st[k]-=shift;if(st[k]<=0.0f)badtemp++;
     if(sm[k]<=0.0f){sm[k]=0.001f;repairs++;}}
   if(st[k]<=0.0f)atomicAdd(errors,1);if(sm[k]<0.0f)atomicAdd(errors+1,1);
 }
 atomicAdd(counts,repairs);atomicAdd(counts+1,badtemp);
 float dz[4]={0.10f,0.30f,0.60f,1.00f},target[4];target[0]=0.5f*dz[0];
 for(int k=1;k<4;k++)target[k]=target[k-1]+0.5f*dz[k-1]+0.5f*dz[k];
 float tt[4]={},mm[4]={},hh[4]={};
 if(land[cell]!=1){for(int k=0;k<4;k++){tt[k]=temp;mm[k]=1.0f;hh[k]=1.0f;}}
 else {
   float have[MAX_FG+2],ti[MAX_FG+2],mi[MAX_FG+2],mid[MAX_FG];
   mid[0]=0.5f*depth[b*ns];for(int k=1;k<ns;k++)mid[k]=mid[k-1]+0.5f*depth[b*ns+k-1]+0.5f*depth[b*ns+k];
   have[0]=0.0f;ti[0]=temp;mi[0]=sm[min(1,ns-1)];
   for(int k=0;k<ns;k++){have[k+1]=__fdiv_rn(mid[k],100.0f);ti[k+1]=st[k];mi[k+1]=sm[k];}
   have[ns+1]=3.0f;ti[ns+1]=deep;mi[ns+1]=deep_corrected?mi[ns]:mi[ns-1];
   for(int l=0;l<4;l++){bool assigned=false;
     for(int k=0;k<ns+1;k++)if(target[l]>=have[k]&&target[l]<=have[k+1]){
       float den=have[k+1]-have[k];
       tt[l]=(ti[k]*(have[k+1]-target[l])+ti[k+1]*(target[l]-have[k]))/den;
       mm[l]=(mi[k]*(have[k+1]-target[l])+mi[k+1]*(target[l]-have[k]))/den;assigned=true;break;
     }
     if(!assigned)atomicAdd(errors+2,1);
   }
 }
 if(land[cell]==1&&tt[0]>170.0f&&tt[0]<400.0f&&mm[0]<0.005f){for(int k=0;k<4;k++)mm[k]=0.005f;atomicAdd(counts+2,1);}
 long long mo=(long long)cell*12;
 float a=(albedo[mo+m2]*w2+albedo[mo+m1]*w1)/span;a=__fdiv_rn(a,100.0f);if(land[cell]==0)a=0.08f;
 float vf=(green[mo+m2]*w2+green[mo+m1]*w1)/span;
 float gmin=green[mo],gmax=green[mo];for(int m=1;m<12;m++){if(green[mo+m]<gmin)gmin=green[mo+m];if(green[mo+m]>gmax)gmax=green[mo+m];}
 float sn=snow0[c],snc=sn>=10.0f?1.0f:0.0f,snh=__fdiv_rn(sn*5.0f,1000.0f);
 float xl=land[cell]==1?1.0f:2.0f,xi=fminf(fmaxf(ice0[c],0.0f),1.0f),sa=snoalb0[cell];
 int iv=veg0[cell],is=soil0[cell];float threshold=fractional?0.02f:0.5f;
 if(!fractional)xi=xi>=0.5f?1.0f:0.0f;
 if(xi>=threshold||(land[cell]==0&&temp<tskthreshold)){
   if(land[cell]==0)deep=271.4f;iv=isice;is=16;sa=0.75f;vf=0.0f;xl=1.0f;
   for(int k=0;k<4;k++){float mid=3.0f/4.0f/2.0f+k*(3.0f/4.0f);tt[k]=__fdiv_rn((3.0f-mid)*temp+mid*deep,3.0f);mm[k]=1.0f;hh[k]=0.0f;}
   atomicAdd(counts+3,1);
 }else{xi=0.0f;if(land[cell]==0){snc=0.0f;snh=0.0f;sn=0.0f;}}
 skin[c]=temp;tmn[c]=deep;ice[c]=xi;seaice[c]=xi>0.0f?1.0f:0.0f;snow[c]=sn;snowc[c]=snc;snowh[c]=snh;
 xland[c]=xl;alb[c]=a;veg[c]=vf;snoalb[c]=sa;vegtype[c]=iv;soiltype[c]=is;shdmin[c]=gmin;shdmax[c]=gmax;
 for(int k=0;k<4;k++){long long off=(long long)c*4+k;tslb[off]=tt[k];smois[off]=mm[k];sh2o[off]=hh[k];dzs[off]=dz[k];zs[off]=target[k];}
 float es=6.112f*gfk_exp((17.27f*(t2[c]-273.16f))/(t2[c]-35.86f));float rs=0.622f*es*100.0f/(ps[c]-es*100.0f);
 q2[c]=0.01f*rs*rh2[c];
}
