"""Device source for MPAS-A v8.4.2 diagnostics; see routine citations below."""
CUDA_SOURCE = r"""
// MPAS-A v8.4.2, pinned f34984b. RKIND is REAL(4).
// Elementary functions come from the engine's shared glibc device headers.
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))

// mpas_isobaric_diagnostics.F:1005-1091, including the collective kupper
// search and exact-top equality behavior. One cooperative block handles
// arbitrary ncol with strided work, preserving native kount semantics.
extern "C" __global__ void fixed_pressure(
    int nc,int ni,int no,const float* p,const float* f,const float* po,
    float* out,int* upper) {
    int t=threadIdx.x;
    __shared__ int start,count,all;
    for(int c=t;c<nc;c+=blockDim.x) upper[c]=0;
    __syncthreads();
    for(int k=0;k<no;k++) {
        if(t==0) {start=ni-1;count=0;all=0;}
        __syncthreads();
        for(int c=t;c<nc;c+=blockDim.x) atomicMin(&start,upper[c]);
        __syncthreads();
        for(int kk=start;kk<ni-1;kk++) {
            for(int c=t;c<nc;c+=blockDim.x) {
                float q=po[c*no+k];
                if(q>p[c*ni+kk] && q<=p[c*ni+kk+1]) {
                    upper[c]=kk;atomicAdd(&count,1);
                }
            }
            __syncthreads();
            if(count==nc) {if(t==0) all=1;break;}
        }
        __syncthreads();
        for(int c=t;c<nc;c+=blockDim.x) {
            float q=po[c*no+k]; int u=upper[c];
            if(!all && q<p[c*ni]) out[c*no+k]=D(M(f[c*ni],q),p[c*ni]);
            else if(!all && q>p[c*ni+ni-1]) out[c*no+k]=f[c*ni+ni-1];
            else {
                float dpu=S(q,p[c*ni+u]),dpl=S(p[c*ni+u+1],q);
                out[c*no+k]=D(A(M(f[c*ni+u],dpl),M(f[c*ni+u+1],dpu)),A(dpl,dpu));
            }
        }
        __syncthreads();
    }
}

// The same routine across the whole grid (fixed_pressure is one block, kept as
// the referee). Per output level k, with start[k] = min over columns of the
// carried kupper: count[k*ni+kk] counts bracket hits at kk; a one-thread scan
// finds the first kk whose running count equals ncol exactly (native kount);
// apply re-finds each column's last hit through that kk, interpolates and
// folds its kupper into start[k+1]. Same words as the one-block kernel.
extern "C" __global__ void fixed_pressure_count(
    int nc,int ni,int no,int k,const float* p,const float* po,const int* upper,
    const int* start,int* count) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    float q=po[c*no+k];
    for(int kk=start[k];kk<ni-1;kk++)
        if(q>p[c*ni+kk] && q<=p[c*ni+kk+1]) atomicAdd(&count[k*ni+kk],1);
}
extern "C" __global__ void fixed_pressure_scan(
    int nc,int ni,int k,const int* start,const int* count,int* stop) {
    if(blockIdx.x||threadIdx.x)return;
    int total=0,last=ni-2,all=0;
    for(int kk=start[k];kk<ni-1;kk++) {
        total+=count[k*ni+kk];
        if(total==nc) {last=kk;all=1;break;}
    }
    stop[2*k]=last;stop[2*k+1]=all;
}
extern "C" __global__ void fixed_pressure_apply(
    int nc,int ni,int no,int k,const float* p,const float* f,const float* po,
    float* out,int* upper,int* start,const int* stop) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    float q=po[c*no+k];int u=upper[c];
    int last=stop[2*k],all=stop[2*k+1];
    for(int kk=start[k];kk<=last;kk++)
        if(q>p[c*ni+kk] && q<=p[c*ni+kk+1]) u=kk;
    upper[c]=u;
    if(!all && q<p[c*ni]) out[c*no+k]=D(M(f[c*ni],q),p[c*ni]);
    else if(!all && q>p[c*ni+ni-1]) out[c*no+k]=f[c*ni+ni-1];
    else {
        float dpu=S(q,p[c*ni+u]),dpl=S(p[c*ni+u+1],q);
        out[c*no+k]=D(A(M(f[c*ni+u],dpl),M(f[c*ni+u+1],dpu)),A(dpl,dpu));
    }
    atomicMin(&start[k+1],u);
}

// mpas_isobaric_diagnostics.F:1245-1351.
extern "C" __global__ void layer_mean(
    int nc,int nl,float p1,float p2,const float* f,const float* p,float* out) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    if(p[0]>p[nl-1]) {out[c]=0.f;return;}
    float pt=p1<p2?p1:p2,pb=p1<p2?p2:p1;
    int kt=-1,kb=-1;
    float wtp=0,wtm=0,wbp=0,wbm=0;
    const float* pc=p+c*nl; const float* fc=f+c*nl;
    for(int k=0;k<nl-1;k++) {
        if(pc[k]<=pt && pc[k+1]>pt) {
            kt=k;wtp=D(S(pt,pc[k]),S(pc[k+1],pc[k]));
            wtm=D(S(pc[k+1],pt),S(pc[k+1],pc[k]));
        }
        if(pc[k]<=pb && pc[k+1]>pb) {
            kb=k;wbm=D(S(pb,pc[k]),S(pc[k+1],pc[k]));
            wbp=D(S(pc[k+1],pb),S(pc[k+1],pc[k]));
        }
    }
    if(kt==-1 || kb==-1) {out[c]=0;return;}
    float x=A(M(wtm,fc[kt]),M(wtp,fc[kt+1]));
    if(kt==kb) {
        x=A(A(x,M(wbm,fc[kb])),M(wbp,fc[kb+1]));
        out[c]=M(.5f,x);return;
    }
    float total=S(pc[kt+1],pt);
    x=M(M(total,.5f),A(fc[kt+1],x));
    for(int k=kt+1;k<kb;k++) {
        float w=S(pc[k+1],pc[k]); total=A(total,w);
        x=A(x,M(M(w,.5f),A(fc[k],fc[k+1])));
    }
    float w=S(pb,pc[kb]); total=A(total,w);
    float temp=A(M(wbm,fc[kb]),M(wbp,fc[kb+1]));
    x=A(x,M(M(w,.5f),A(fc[kb],temp)));
    out[c]=D(x,total);
}

// mpas_isobaric_diagnostics.F:1094-1227. Inputs are column-major C arrays
// (column,level); p is hPa. A failed column marks the entire output invalid.
extern "C" __global__ void slp_columns(
    int nc,int nl,const float* p,const float* temp,const float* h,
    const float* q,float* out,int* failed) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
    const float* pc=p+c*nl;const float* tc=temp+c*nl;
    const float* hc=h+c*(nl+1);const float* qc=q+c*nl;
    int level=-1;
    for(int k=0;k<nl;k++)if(pc[k]<S(pc[0],100.f)){level=k;break;}
    if(level==-1){atomicExch(failed,1);return;}
    int lo=max(level-1,0),hi=min(lo+1,nl-2);
    if(lo==hi){atomicExch(failed,1);return;}
    float plo=pc[lo],phi=pc[hi];
    float tlo=M(tc[lo],A(1.f,M(.608f,qc[lo])));
    float thi=M(tc[hi],A(1.f,M(.608f,qc[hi])));
    float zlo=M(.5f,A(hc[lo],hc[lo+1]));
    float zhi=M(.5f,A(hc[hi],hc[hi+1]));
    float pp=S(pc[0],100.f);
    float lp=gfk_log(D(pp,phi)),ll=gfk_log(D(plo,phi));
    float tp=S(thi,M(M(S(thi,tlo),lp),ll));
    float zp=S(zhi,M(M(S(zhi,zlo),lp),ll));
    float ts=M(tp,gfk_pow(D(pc[0],pp),D(M(.0065f,287.f),9.80616f)));
    float tm=A(tp,M(.0065f,zp));
    const float limit=290.66f;
    if(ts<=limit && tm>=limit) tm=limit;
    else {float delta=S(ts,limit);tm=S(limit,M(.005f,M(delta,delta)));}
    float zhalf=M(.5f,A(hc[0],hc[1]));
    out[c]=M(pc[0],gfk_exp_fma(D(M(19.61232f,zhalf),M(287.f,A(tm,ts)))));
}

extern "C" __global__ void slp_finish(int nc,float* out,const int* failed,float scale) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<nc)out[c]=*failed?0.f:M(out[c],scale);
}

// mpas_isobaric_diagnostics.F:583-594 and 643-658.
extern "C" __global__ void prepare_mass(
    int nc,int nl,int stride,int thermo,const float* pp,const float* pb,
    const float* theta,const float* exner,const float* q,
    float* p,float* temp,float* dew) {
    int x=blockIdx.x*blockDim.x+threadIdx.x;if(x>=nc*nl)return;
    int c=x%nc,k=x/nc,s=k*stride+c;
    p[x]=D(A(pp[s],pb[s]),100.f);
    if(thermo && c<nc-1) {
        // rvord is the native constant folded from 461.6_RKIND / 287_RKIND.
        temp[k*(nc-1)+c]=M(D(theta[s],A(1.f,M(0x1.9bbda4p+0f,q[s]))),exner[s]);
        float evp=D(M(p[x],q[s]),A(q[s],.622f));evp=fmaxf(evp,1.e-8f);
        float l=gfk_log(D(evp,6.112f));
        dew[k*(nc-1)+c]=A(D(M(243.5f,l),S(17.67f,l)),273.15f);
    }
}

// mpas_isobaric_diagnostics.F:596-628. Heights are (level,cell).
extern "C" __global__ void prepare_interfaces(
    int nc,int nl,int hs,const float* p,const float* h,float* pi) {
    int x=blockIdx.x*blockDim.x+threadIdx.x;if(x>=nc*(nl+1))return;
    int c=x%nc,k=x/nc;float w1,w2,pa,pb;
    if(k==nl || k==0) {
        int a=k==nl?nl-1:0,b=k==nl?nl-2:1;
        float z0=h[k*hs+c];
        float z1=M(.5f,A(h[k*hs+c],h[(k==nl?k-1:k+1)*hs+c]));
        float z2=M(.5f,A(h[(k==nl?k-1:k+1)*hs+c],h[(k==nl?k-2:k+2)*hs+c]));
        w1=D(S(z0,z2),S(z1,z2));w2=S(1.f,w1);
        pa=p[a*(nc+1)+c];pb=p[b*(nc+1)+c];
    } else {
        float den=S(h[(k+1)*hs+c],h[(k-1)*hs+c]);
        w1=D(S(h[k*hs+c],h[(k-1)*hs+c]),den);
        w2=D(S(h[(k+1)*hs+c],h[k*hs+c]),den);
        pa=p[k*(nc+1)+c];pb=p[(k-1)*(nc+1)+c];
    }
    pi[x]=gfk_exp_fma(A(M(w1,gfk_log(pa)),M(w2,gfk_log(pb))));
}

// mpas_isobaric_diagnostics.F:630-641. Native 1-based connectivity includes
// its nCells+1 sentinel; do not discard that pressure contribution.
extern "C" __global__ void prepare_vertices(
    int nc,int nv,int nl,int degree,const int* cov,const float* kite,
    const float* area,const float* p,float* pv) {
    int x=blockIdx.x*blockDim.x+threadIdx.x;if(x>=nv*nl)return;
    int v=x%nv,k=x/nv;float sum=0.f;
    for(int d=0;d<degree;d++)sum=A(sum,M(kite[d*nv+v],p[k*(nc+1)+cov[d*nv+v]-1]));
    pv[x]=D(sum,area[v]);
}

// mpas_atm_core.F:916-965. Driver calls after advancing the clock and
// refreshing state level 1, before diagnostics manager compute and stream write.
extern "C" __global__ void output_diagnostics(
    int nc,int nl,const float* theta_m,const float* rho_zz,const float* zz,
    const float* q,const float* pb,const float* pp,float* theta,float* rho,float* p) {
    int x=blockIdx.x*blockDim.x+threadIdx.x;if(x>=nc*nl)return;
    int c=x%nc,k=x/nc,s=k*(nc+1)+c;
    theta[s]=D(theta_m[s],A(1.f,M(0x1.9bbda4p+0f,q[s])));
    rho[s]=M(rho_zz[s],zz[s]);p[s]=A(pb[s],pp[s]);
}

"""
