"""MPAS-A v8.4.2 Ertel PV, dynamic tropopause and tendency budget.

Device arrays retain Fortran layouts and one-based connectivity. Halo operations
are explicit driver callbacks. The strict native top curl's stable stray read
is reproduced, including its NaN payload. Replay captures may override scratch
inputs; they are not a forecast identity certificate.
"""
from __future__ import annotations

from .cloud_v842 import _array, _qv_array, _compile_cuda

PV_FIELDS = ("ertel_pv", "u_pv", "v_pv", "theta_pv", "vort_pv", "iLev_DT")
SCHEMES = ("lw", "sw", "bl", "cu", "mp", "mix")
BUDGET_FIELDS = tuple("depv_dt_" + s for s in SCHEMES) + (
    "dtheta_dt_mp", "depv_dt_diab", "depv_dt_fric", "depv_dt_diab_pv", "depv_dt_fric_pv")


CUDA_SOURCE = r'''
#define ADD(a,b) __fadd_rn(a,b)
#define SUB(a,b) __fsub_rn(a,b)
#define MUL(a,b) __fmul_rn(a,b)
#define DIV(a,b) __fdiv_rn(a,b)
struct V { float x[3]; };
// mpas_pv_diagnostics.F:155-192
__device__ float dot(V a,V b) {
    float s=0.0f;
    for(int i=0;i<3;++i) s=ADD(s,MUL(a.x[i],b.x[i]));
    return s;
}
// mpas_pv_diagnostics.F:337-374
__device__ V norm(V a) {
    float s=__fsqrt_rn(dot(a,a));
    for(int i=0;i<3;++i) a.x[i]=DIV(a.x[i],s);
    return a;
}
struct Mesh {
    int nc,nz,me;
    const int *nec,*coc,*eoc,*coe,*voc,*cov;
    const float *kite,*area,*dv,*normal,*tangent,*vertical,*z;
};
#define MESH_ARGS int nc,int nz,int me,const int* nec,const int* coc, \
    const int* eoc,const int* coe,const int* voc,const int* cov, \
    const float* kite,const float* area,const float* dv,const float* normal, \
    const float* tangent,const float* vertical,const float* z
#define MAKE_MESH Mesh m{nc,nz,me,nec,coc,eoc,coe,voc,cov,kite,area,dv,normal,tangent,vertical,z}
// mpas_pv_diagnostics.F:277-294
__device__ void basis(Mesh m,int c,V* xyz) {
    for(int j=0;j<2;++j) {
        for(int i=0;i<3;++i) xyz[j].x[i]=m.tangent[i+3*j+6*c];
        xyz[j]=norm(xyz[j]);
    }
    for(int i=0;i<3;++i) xyz[2].x[i]=m.vertical[i+3*c];
}
// mpas_pv_diagnostics.F:227-251
__device__ V earth(V a,V* xyz,float omega) {
    V e{{0.0f,0.0f,MUL(2.0f,omega)}};
    for(int i=0;i<3;++i) a.x[i]=ADD(a.x[i],dot(e,xyz[i]));
    return a;
}
// mpas_pv_diagnostics.F:194-225; unused native helper, including gradxu mutation.
extern "C" __global__ void form_ertel(int count,float omega,float* curl,
    const float* gradient,const float* density,const float* coordinates,float* out) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=count) return;
    V a,h,xyz[3];
    for(int i=0;i<3;++i) {
        a.x[i]=curl[i+3*j];h.x[i]=gradient[i+3*j];
        for(int k=0;k<3;++k) xyz[k].x[i]=coordinates[i+3*k+9*j];
    }
    a=earth(a,xyz,omega);
    for(int i=0;i<3;++i) curl[i+3*j]=a.x[i];
    out[j]=MUL(DIV(dot(a,h),density[j]),1.0e6f);
}
// mpas_pv_diagnostics.F:174-192; first match, native absent value -1.
__device__ int element_index(int value,const int* array,int size) {
    for(int i=0;i<size;++i) if(array[i]==value) return i+1;
    return -1;
}
// mpas_pv_diagnostics.F:253-275
__device__ float vvort(Mesh m,int c,int k,const float* v) {
    float s=0.0f;
    for(int i=0;i<m.nec[c];++i) {
        int vertex=m.voc[i+m.me*c]-1;
        int slot=element_index(c+1,m.cov+3*vertex,3)-1;
        s=ADD(s,DIV(MUL(m.kite[slot+3*vertex],v[k+m.nz*vertex]),m.area[c]));
    }
    return s;
}
// mpas_pv_diagnostics.F:312-335
__device__ float height(Mesh m,int c,int k) {
    return MUL(0.5f,ADD(m.z[k+(m.nz+1)*c],m.z[k+1+(m.nz+1)*c]));
}
__device__ float height_edge(Mesh m,int c,int n,int k) {
    float ht=MUL(0.5f,ADD(m.z[k+1+(m.nz+1)*c],m.z[k+1+(m.nz+1)*n]));
    float hb=MUL(0.5f,ADD(m.z[k+(m.nz+1)*c],m.z[k+(m.nz+1)*n]));
    return SUB(ht,hb);
}
// mpas_pv_diagnostics.F:355-403,936-984,1086-1114
__device__ float horiz(Mesh m,int c,int k,const float* field,bool w,V deriv) {
    float avg=0.0f;
    for(int i=0;i<m.nec[c];++i) avg=ADD(avg,100.0f);
    float vol=MUL(m.area[c],DIV(avg,float(m.nec[c]))), sum=0.0f;
    float v0=w?MUL(0.5f,ADD(field[k+1+(m.nz+1)*c],field[k+(m.nz+1)*c])):field[k+m.nz*c];
    for(int i=0;i<m.nec[c];++i) {
        int n=m.coc[i+m.me*c]-1, e=m.eoc[i+m.me*c]-1;
        float vn=w?MUL(0.5f,ADD(field[k+1+(m.nz+1)*n],field[k+(m.nz+1)*n])):field[k+m.nz*n];
        float ve=MUL(0.5f,ADD(vn,v0));
        V normal;
        for(int j=0;j<3;++j) normal.x[j]=m.normal[j+3*e];
        normal=norm(normal);
        // mpas_pv_diagnostics.F:296-310 (first normalization precedes sign)
        float sign=m.coe[2*e]==c+1?1.0f:-1.0f;
        for(int j=0;j<3;++j) normal.x[j]=MUL(normal.x[j],sign);
        normal=norm(normal); // native FV routine normalizes a second time
        float af=MUL(m.dv[e],100.0f);
        af=MUL(af,dot(normal,deriv));
        sum=ADD(sum,MUL(ve,af));
    }
    return DIV(sum,vol);
}
// mpas_pv_diagnostics.F:406-430
__device__ float one(float p,float m,float dz) {return DIV(SUB(p,m),dz);}
__device__ float center(float v,float p,float m,float z,float zp,float zm) {
    return MUL(0.5f,ADD(one(p,v,SUB(zp,z)),one(v,m,SUB(z,zm))));
}
// mpas_pv_diagnostics.F:1059-1146
__device__ V gradient(Mesh m,int c,int k,const float* f) {
    V xyz[3], g; basis(m,c,xyz);
    g.x[0]=horiz(m,c,k,f,false,xyz[0]);
    g.x[1]=horiz(m,c,k,f,false,xyz[1]);
    float z0=height(m,c,k), v=f[k+m.nz*c];
    if(k==0) g.x[2]=one(f[k+1+m.nz*c],v,SUB(height(m,c,k+1),z0));
    else if(k==m.nz-1) g.x[2]=one(v,f[k-1+m.nz*c],SUB(z0,height(m,c,k-1)));
    else g.x[2]=center(v,f[k+1+m.nz*c],f[k-1+m.nz*c],z0,height(m,c,k+1),height(m,c,k-1));
    return g;
}
__device__ V velocity(int c,int k,int nz,const float* x,const float* y,const float* z) {
    int j=k+nz*c; return V{{x[j],y[j],z[j]}};
}
// mpas_pv_diagnostics.F:936-1057
__device__ V curl(Mesh m,int c,int k,const float* x,const float* y,const float* z,
                 const float* w,const float* vort,const float* top_velp,int addEarth,float omega) {
    V xyz[3]; basis(m,c,xyz);
    float wx=horiz(m,c,k,w,true,xyz[0]), wy=horiz(m,c,k,w,true,xyz[1]);
    V v=velocity(c,k,m.nz,x,y,z);
    float du,dv,z0=height(m,c,k);
    if(k==0) {
        V p=velocity(c,k+1,m.nz,x,y,z);
        du=one(dot(p,xyz[0]),dot(v,xyz[0]),SUB(height(m,c,k+1),z0));
        dv=one(dot(p,xyz[1]),dot(v,xyz[1]),SUB(height(m,c,k+1),z0));
    } else if(k==m.nz-1) {
        V vm=velocity(c,k-1,m.nz,x,y,z), p;
        // Native line 1032 reads unassigned velCellp. The default launcher
        // reproduces the measured strict-binary scratch; captures aid replay.
        for(int j=0;j<3;++j) p.x[j]=top_velp[j+3*c];
        du=one(dot(v,xyz[0]),dot(vm,xyz[0]),SUB(z0,height(m,c,k-1)));
        dv=one(dot(v,xyz[1]),dot(p,xyz[1]),SUB(z0,height(m,c,k-1)));
    } else {
        V p=velocity(c,k+1,m.nz,x,y,z), vm=velocity(c,k-1,m.nz,x,y,z);
        du=center(dot(v,xyz[0]),dot(p,xyz[0]),dot(vm,xyz[0]),z0,height(m,c,k+1),height(m,c,k-1));
        dv=center(dot(v,xyz[1]),dot(p,xyz[1]),dot(vm,xyz[1]),z0,height(m,c,k+1),height(m,c,k-1));
    }
    V g{{SUB(wy,dv),SUB(du,wx),vvort(m,c,k,vort)}};
    return addEarth?earth(g,xyz,omega):g;
}
// mpas_pv_diagnostics.F:1277-1291,1605-1613
extern "C" __global__ void prepare(int count,const float* tm,const float* q,
    const float* rz,const float* zz,float rvord,float* theta,float* rho) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<count) {theta[j]=DIV(tm[j],ADD(1.0f,MUL(rvord,q[j])));rho[j]=MUL(rz[j],zz[j]);}
}
extern "C" __global__ void mix_theta(int count,const float* tend,const float* q,float rvord,float* mix) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<count) mix[j]=DIV(tend[j],ADD(1.0f,MUL(rvord,q[j])));
}
// mpas_pv_diagnostics.F:1174-1251,1429-1505,1540-1566
extern "C" __global__ void pv_dot(MESH_ARGS,int ns,int count,int mode,float omega,
    const float* x,const float* y,const float* zr,const float* w,const float* vort,
    const float* top,const float* rho,const float* scalar,float* out) {
    MAKE_MESH;
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=ns*count) return;
    int c=j/count,k=j%count;
    V g=curl(m,c,k,x,y,zr,w,vort,top,mode==0,omega), h=gradient(m,c,k,scalar);
    if(mode==0) for(int i=0;i<3;++i) g.x[i]=DIV(g.x[i],rho[k+nz*c]);
    else for(int i=0;i<3;++i) h.x[i]=DIV(h.x[i],rho[k+nz*c]);
    float value=MUL(dot(g,h),1.0e6f);
    // mpas_pv_diagnostics.F:1007-1033; native policy item 3. GNU propagates
    // the observed first-operand NaN payload; CUDA arithmetic canonicalizes it.
    if(k==nz-1 && isnan(top[3*c]))
        value=__int_as_float(__float_as_int(top[3*c])|0x00400000);
    out[k+nz*c]=value;
}
// mpas_pv_diagnostics.F:1007-1033; strict artifact scratch observed by ptrace.
// calc_grad_cell's array offset overwrites the first word between curl calls.
extern "C" __global__ void native_top(int nc,int nz,const float* z,float* top) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc) return;
    top[3*c]=__int_as_float(-(nz+1));top[3*c+1]=0.0f;
    top[3*c+2]=z[nz-1+nz*c];
}
// mpas_pv_diagnostics.F:253-275,753-819
extern "C" __global__ void vertex_average(MESH_ARGS,int ns,const float* vort,float* out) {
    MAKE_MESH; int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<ns*nz) out[j]=vvort(m,j/nz,j%nz,vort);
}
// mpas_pv_diagnostics.F:861-934
extern "C" __global__ void interp(int ns,int nz,const float* lat,const int* lev,
    const float* pv,const float* field,float target,float pvPrefix,float fieldPrefix,float* out) {
    int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=ns) return;
    int l=lev[c];
    if(l>nz) {out[c]=field[nz-1+nz*c];return;}
    if(l<1) {out[c]=field[nz*c];return;}
    // mpas_pv_diagnostics.F:899: l==1 aliases the preceding column's last
    // word. Default troposphere closure produces only 0 or >=2.
    float hi=pv[l-1+nz*c],lo=(l==1&&c==0)?pvPrefix:pv[l-2+nz*c],d=SUB(hi,lo);
    if(isnan(hi)||isnan(lo)) {
        float q=isnan(hi)?hi:lo;
        out[c]=__int_as_float(__float_as_int(q)|0x00400000);return;
    }
    float sign=copysignf(1.0f,lat[c]);
    float f=fabsf(d)<1.0e-6f?0.5f:DIV(SUB(MUL(target,sign),lo),d);
    float fh=field[l-1+nz*c],fl=(l==1&&c==0)?fieldPrefix:field[l-2+nz*c];
    out[c]=ADD(fl,MUL(SUB(fh,fl),f));
}
// mpas_pv_diagnostics.F:589-751 (serial domain closure preserves update order)
extern "C" __global__ void tropo_seed(int nc,int nz,const float* lat,const float* pv,
    float target,int* cand,int* mask,int* changed) {
    if(blockIdx.x||threadIdx.x) return;
    int n=0;
    for(int j=0;j<(nc+1)*nz;++j) {cand[j]=0;mask[j]=0;}
    for(int c=0;c<nc;++c) for(int k=0;k<nz;++k) {
        int j=k+nz*c;
        if(SUB(MUL(pv[j],copysignf(1.0f,lat[c])),target)<0.0f) {
            cand[j]=1; if(k<3) {mask[j]=1;++n;}
        }
    }
    *changed=n;
}
extern "C" __global__ void tropo_close(int nc,int nz,int me,const int* nec,const int* coc,
    const int* cand,int* mask,int* changed) {
    if(blockIdx.x||threadIdx.x) return;
    int n=*changed,total=0;
    while(n>0) {
        n=0;
        for(int c=0;c<nc;++c) for(int k=0;k<nz;++k) {
            int j=k+nz*c;
            if(!cand[j]||mask[j]) continue;
            if(k>0&&mask[j-1]) {mask[j]=1;++n;continue;}
            for(int i=0;i<nec[c];++i) if(mask[k+nz*(coc[i+me*c]-1)]) {
                mask[j]=1;++n;break;
            }
            if(k<nz-1&&mask[j+1]) {mask[j]=1;++n;continue;}
        }
        total+=n;
    }
    *changed=total;
}
extern "C" __global__ void tropo_levels(int nc,int nz,const int* mask,int* lev) {
    int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=nc) return;
    lev[c]=0;
    for(int k=nz-1;k>=0;--k) if(mask[k+nz*c]) {lev[c]=k+2;break;}
}
// mpas_pv_diagnostics.F:432-587; unused alternative, including native sticky nChanged.
extern "C" __global__ void strato(int nc,int nz,int me,const int* nec,const int* coc,
    const float* lat,const float* pv,float target,float stratoPV,int* cand,int* mask,int* lev) {
    if(blockIdx.x||threadIdx.x) return;
    for(int j=0;j<(nc+1)*nz;++j) {cand[j]=0;mask[j]=0;}
    for(int c=0;c<nc;++c) for(int k=0;k<nz;++k) {
        int j=k+nz*c;
        cand[j]=SUB(MUL(pv[j],copysignf(1.0f,lat[c])),target)>=0.0f;
    }
    int n=0;
    for(int c=0;c<nc;++c) for(int k=nz-6;k<nz;++k) if(cand[k+nz*c]) {
        mask[k+nz*c]=1;cand[k+nz*c]=0;++n;
    }
    while(n>0) {
        n=0;
        for(int c=0;c<nc;++c) for(int k=nz-1;k>=0;--k) {
            int j=k+nz*c; if(!cand[j]) continue;
            if(k<nz-1&&mask[j+1]) {mask[j]=1;cand[j]=0;++n;continue;}
            for(int i=0;i<nec[c];++i) if(mask[k+nz*(coc[i+me*c]-1)]) {
                mask[j]=1;cand[j]=0;++n;continue; // native CYCLE is neighbor loop
            }
            if(k>0&&mask[j-1]) {mask[j]=1;cand[j]=0;++n;continue;}
        }
    }
    for(int c=0;c<nc;++c) if(mask[nz*c]) {
        for(int k=1;k<nz;++k) {
            if(!mask[k+nz*c]) break;
            if(SUB(MUL(pv[k+nz*c],copysignf(1.0f,lat[c])),stratoPV)<0.0f)
                for(int i=0;i<=k;++i) mask[i+nz*c]=0;
        }
    }
    n=0;
    for(int c=0;c<nc;++c) {
        int k;for(k=0;k<nz;++k) if(mask[k+nz*c]) {n=1;break;}
        if(n) {
            if(k==0) {
                if(SUB(MUL(pv[nz*c],copysignf(1.0f,lat[c])),target)>0.0f) lev[c]=0;
            } else lev[c]=k+1;
        } else lev[c]=nz+2;
    }
}
// mpas_pv_diagnostics.F:1507-1507; native left-associated six-field sum.
extern "C" __global__ void diab_sum(int count,const float* lw,const float* sw,const float* bl,
    const float* cu,const float* mp,const float* mix,float* out) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<count) {
        float v=ADD(ADD(ADD(ADD(ADD(lw[j],sw[j]),bl[j]),cu[j]),mp[j]),mix[j]);
        // The stable native scratch NaN has the same payload in every term.
        const float terms[6]={lw[j],sw[j],bl[j],cu[j],mp[j],mix[j]};
        for(int i=0;i<6;++i) if(isnan(terms[i])) {
            v=__int_as_float(__float_as_int(terms[i])|0x00400000);break;
        }
        out[j]=v;
    }
}
// mpas_pv_diagnostics.F:1512-1516,1527-1538; mutations are intentional.
extern "C" __global__ void uncouple_u(int count,const float* euler,const float* rho,float* phys) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<count) phys[j]=ADD(phys[j],DIV(euler[j],rho[j]));
}
extern "C" __global__ void uncouple_w(int nc,int nz,const float* rho,float* w) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=nc*(nz+1)) return;
    int c=j/(nz+1),k=j%(nz+1);
    float r=k==0?rho[nz*c]:(k==nz?rho[nz-1+nz*c]:MUL(0.5f,ADD(rho[k-1+nz*c],rho[k+nz*c])));
    w[j]=DIV(w[j],r);
}
// mpas_pv_diagnostics.F:1148-1172; no atomics/reassociated reductions.
__device__ void vertical_curl_level(int ne,int nv,int nz,int k,const int* voe,const float* dc,
    const float* area,const float* u,float* vort,const int* order) {
    for(int v=0;v<nv+1;++v) vort[k+nz*v]=0.0f;
    for(int i=0;i<ne;++i) {
        int e=order?order[i]:i;
        int a=voe[2*e]-1,b=voe[2*e+1]-1;
        float value=MUL(dc[e],u[k+nz*e]);
        vort[k+nz*a]=SUB(vort[k+nz*a],value);
        vort[k+nz*b]=ADD(vort[k+nz*b],value);
    }
    for(int v=0;v<nv;++v) vort[k+nz*v]=DIV(vort[k+nz*v],area[v]);
}
extern "C" __global__ void vertical_curl(int ne,int nv,int nz,const int* voe,const float* dc,
    const float* area,const float* u,float* vort) {
    int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=nz) return;
    vertical_curl_level(ne,nv,nz,k,voe,dc,area,u,vort,nullptr);
}
extern "C" __global__ void vertical_curl_ordered(int ne,int nv,int nz,const int* voe,const float* dc,
    const float* area,const float* u,float* vort,const int* order) {
    int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=nz) return;
    vertical_curl_level(ne,nv,nz,k,voe,dc,area,u,vort,order);
}
// mpas_block_decomp.F:633-685; single global block's first-occurrence edge list.
// mpas_block_decomp.F:565-620; every edge is owned in the global single block.
extern "C" __global__ void global_block_edge_order(int nc,int ne,int me,const int* nec,
    const int* eoc,int* seen,int* order) {
    if(blockIdx.x||threadIdx.x) return;
    for(int e=0;e<ne;++e) seen[e]=0;
    int count=0;
    for(int c=0;c<nc;++c) for(int i=0;i<nec[c];++i) {
        int e=eoc[i+me*c]-1;
        if(e>=0&&e<ne&&!seen[e]) {seen[e]=1;order[count++]=e;}
    }
    // Preserve storage for any unreferenced boundary edges in a supplied mesh.
    for(int e=0;e<ne;++e) if(!seen[e]) order[count++]=e;
}
// mpas_vector_reconstruction.F:270-302; only Cartesian results used by budget.
extern "C" __global__ void reconstruct(int ns,int nz,int me,const int* nec,const int* eoc,
    const float* coeff,const float* u,float* x,float* y,float* z) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=ns*nz) return;
    int c=j/nz,k=j%nz;float a=0.0f,b=0.0f,d=0.0f;
    for(int i=0;i<nec[c];++i) {
        float v=u[k+nz*(eoc[i+me*c]-1)];int q=3*i+3*me*c;
        a=ADD(a,MUL(coeff[q],v));b=ADD(b,MUL(coeff[q+1],v));d=ADD(d,MUL(coeff[q+2],v));
    }
    x[j]=a;y[j]=b;z[j]=d;
 }
// mpas_pv_diagnostics.F:174-192,312-335: direct unused-helper replay entry.
extern "C" __global__ void helper_replay(int nc,int nz,int size,const int* values,
    const int* arrays,const float* z,int* indices,float* centers,float* edges) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc) return;
    indices[c]=element_index(values[c],arrays+size*c,size);
    Mesh m{};m.nz=nz;m.z=z;
    for(int k=0;k<nz;++k) {
        centers[k+nz*c]=height(m,c,k);
        edges[k+nz*c]=height_edge(m,c,(c+1)%nc,k);
    }
}
'''


class PVDiagnostics:
    """Pool setup and stream demand: mpas_pv_diagnostics.F:46-152."""

    def __init__(self, mesh=None, state=None, diag=None, tend=None, tend_physics=None, *,
                 exchange=None, max_int=None, distributed=False, omega=7.29212e-5,
                 rvord=None, edge_order=None):
        self.mesh, self.state, self.diag = mesh, state, diag
        self.edge_order = edge_order
        self.tend, self.tend_physics = tend, tend_physics
        self.utils = None
        self.exchange, self.max_int, self.distributed = exchange, max_int, distributed
        import numpy as np
        self.omega = np.float32(omega)
        # MPAS constants are folded in RKIND, not divided in Python double.
        self.rvord = np.float32(rvord) if rvord is not None else np.float32(461.6)/np.float32(287.0)
        self._module = None
        if distributed and (exchange is None or max_int is None):
            raise ValueError("distributed PV requires halo exchange and integer max reduction")

    def setup(self, pools, clock, utils):
        """Bind native pools and clock: mpas_pv_diagnostics.F:46-75."""
        self.mesh, self.state, self.diag = (pools[name] for name in ("mesh", "state", "diag"))
        self.tend, self.tend_physics = pools.get("tend"), pools.get("tend_physics")
        self.clock, self.utils = clock, utils

    def _kernel(self, name, count, args, serial=False):
        import cupy as cp
        if self._module is None:
            self._module = _compile_cuda(CUDA_SOURCE)
        if count:
            self._module.get_function(name)((1,) if serial else ((count+127)//128,),
                (1,) if serial else (128,), args)

    def _real(self, pool, name, ndim=2):
        return _array(pool[name], "float32", ndim)

    def form_ertel_pv(self, curl, gradient, density, coordinates, out):
        """Unused native scalar helper and mutation: mpas_pv_diagnostics.F:194-225."""
        import numpy as np
        count=density.size
        if curl.shape != (3,count) or gradient.shape != curl.shape or coordinates.shape != (3,3,count) or out.shape != (count,):
            raise ValueError("Ertel helper vector or coordinate extents disagree")
        self._kernel("form_ertel",count,(np.int32(count),self.omega,
            _array(curl,"float32",2),_array(gradient,"float32",2),
            _array(density,"float32",1),_array(coordinates,"float32",3),_array(out,"float32",1)))

    def set_global_edge_order(self):
        """Restore single-block order after global I/O: mpas_block_decomp.F:633-685.

        Local MPAS driver arrays already have this order. The global-pool
        adapter must retain the permutation when it reads stream-ordered data.
        No field values or mesh connectivity are changed.
        """
        import cupy as cp
        nc, _, me = self._sizes()
        ne = cp.int32(self.mesh["nEdges"])
        self.edge_order = cp.empty(int(ne), dtype=cp.int32)
        seen = cp.empty(int(ne), dtype=cp.int32)
        self._kernel("global_block_edge_order", 1, (nc, ne, me,
            self._int("nEdgesOnCell", 1), self._int("edgesOnCell", 2),
            seen, self.edge_order), True)
        return self.edge_order

    def vertical_curl(self, physical_force, out):
        """Preserve local edge reductions: mpas_pv_diagnostics.F:1148-1172."""
        import cupy as cp
        _, nz, _ = self._sizes()
        ne, nv = (cp.int32(self.mesh[name]) for name in ("nEdges", "nVertices"))
        arguments = (ne, nv, nz, self._int("verticesOnEdge", 2),
            self._real(self.mesh, "dcEdge", 1), self._real(self.mesh, "areaTriangle", 1),
            _array(physical_force, "float32", 2), _array(out, "float32", 2))
        kernel = "vertical_curl"
        if self.edge_order is not None:
            kernel = "vertical_curl_ordered"
            arguments += (_array(self.edge_order, "int32", 1),)
        self._kernel(kernel, int(nz), arguments)

    def _int(self, name, ndim):
        return _array(self.mesh[name], "int32", ndim)

    def _sizes(self):
        import numpy as np
        return tuple(np.int32(self.mesh[name]) for name in ("nCells", "nVertLevels", "maxEdges"))

    def _mesh_args(self):
        return (*self._sizes(), *(self._int(name, rank) for name, rank in (
            ("nEdgesOnCell", 1), ("cellsOnCell", 2), ("edgesOnCell", 2),
            ("cellsOnEdge", 2), ("verticesOnCell", 2), ("cellsOnVertex", 2))),
            *(self._real(self.mesh, name, rank) for name, rank in (
            ("kiteAreasOnVertex", 2), ("areaCell", 1), ("dvEdge", 1),
            ("edgeNormalVectors", 2), ("cellTangentPlane", 3),
            ("localVerticalUnitVectors", 2), ("zgrid", 2))))

    def _halo(self, pool, names):
        if self.exchange is not None:
            for name in names:
                self.exchange(name, pool.get(name))

    def _qv(self):
        return _qv_array(self.state)

    def _top(self, captured, velocity=None):
        """Reproduce stable strict-native scratch: mpas_pv_diagnostics.F:1007-1033.

        receipts/native-pv-scratch-trace.json observes -(nVertLevels+1) as the
        first float word, zero as the second, and the previous curl's top Z as
        the third. Native repeat/rank/host output hashes agree. No words are
        excluded and no diagnostic is disabled because of this native defect.
        """
        if captured is not None:
            return _array(captured, "float32", 2)
        import cupy as cp
        nc,nz,_=self._sizes()
        top=cp.empty((3,int(nc)+1),dtype=cp.float32,order="F")
        z=self._real(self.diag,"uReconstructZ") if velocity is None else velocity
        self._kernel("native_top",int(nc),(nc,nz,z,top))
        return top

    def calc_epv(self, *, captured_top_velp=None, level_count=None):
        """Ertel PV: mpas_pv_diagnostics.F:1174-1251.

        level_count permits replay of the defined levels below the native defect.
        """
        import cupy as cp
        nc, nz, _ = self._sizes()
        count = int(nz) if level_count is None else int(level_count)
        if not 0 <= count <= nz or nz < 2:
            raise ValueError("PV vertical derivatives require at least two levels and a valid replay count")
        top = self._top(captured_top_velp) if count == nz else cp.empty((3, int(nc)+1), dtype=cp.float32, order="F")
        self._dot(self.diag["theta"], self.diag["ertel_pv"], top, count=count)

    def _dot(self, scalar, out, top, *, count=None, velocity=None, mode=0):
        import numpy as np
        _, nz, _ = self._sizes()
        count = int(nz) if count is None else count
        ns = int(self.mesh["nCellsSolve"])
        velocity = velocity if velocity is not None else (
            *(self._real(self.diag, "uReconstruct"+s) for s in ("X", "Y", "Z")),
            self._real(self.state, "w"), self._real(self.diag, "vorticity"))
        self._kernel("pv_dot", ns*count, (*self._mesh_args(), np.int32(ns),
            np.int32(count), np.int32(mode), self.omega, *velocity, top,
            self._real(self.diag, "rho"), _array(scalar, "float32", 2),
            _array(out, "float32", 2)))

    def flood_fill_tropo(self, target=2.0):
        """Distributed native fixed point: mpas_pv_diagnostics.F:589-751."""
        import cupy as cp
        nc, nz, me = self._sizes()
        cand = cp.empty((int(nz), int(nc)+1), dtype=cp.int32, order="F")
        mask = _array(self.diag["inTropo"], "int32", 2)
        changed = cp.empty(1, dtype=cp.int32)
        self._kernel("tropo_seed", 1, (nc, nz, self._real(self.mesh, "latCell", 1),
            self._real(self.diag, "ertel_pv"), cp.float32(target), cand, mask, changed), True)
        while True:
            self._kernel("tropo_close", 1, (nc,nz,me,self._int("nEdgesOnCell",1),
                self._int("cellsOnCell",2),cand,mask,changed), True)
            n = int(changed.get()[0])
            global_n = self.max_int(n) if self.distributed else n
            if global_n <= 0:
                break
            if self.exchange is not None:
                self.exchange("inTropo", mask)
            changed.fill(global_n)
        self._kernel("tropo_levels", int(nc), (nc,nz,mask,
            _array(self.diag["iLev_DT"], "int32", 1)))

    def flood_fill_strato(self, target=2.0, strato_pv=10.0):
        """Unused native alternative: mpas_pv_diagnostics.F:432-587."""
        import cupy as cp
        nc,nz,me = self._sizes()
        if nz < 6:
            raise ValueError("MPAS stratosphere seeds nVertLevels-5: out of bounds below six levels")
        cand = cp.empty((int(nz),int(nc)+1), dtype=cp.int32, order="F")
        mask = cp.empty_like(cand)
        self._kernel("strato", 1, (nc,nz,me,self._int("nEdgesOnCell",1),
            self._int("cellsOnCell",2), self._real(self.mesh,"latCell",1),
            self._real(self.diag,"ertel_pv"),cp.float32(target),cp.float32(strato_pv),
            cand,mask,_array(self.diag["iLev_DT"],"int32",1)), True)
        return mask

    def interp_pv(self, field, out, target=2.0, *, captured_prefix=(0.0,0.0)):
        """Native endpoint fallback and linear interpolation: mpas_pv_diagnostics.F:861-934."""
        import cupy as cp
        nc,nz,_ = self._sizes()
        ns = cp.int32(self.mesh["nCellsSolve"])
        lev = _array(self.diag["iLev_DT"], "int32", 1)
        self._kernel("interp",int(ns),(ns,nz,self._real(self.mesh,"latCell",1),lev,
            self._real(self.diag,"ertel_pv"),_array(field,"float32",2),
            cp.float32(target),*(cp.float32(v) for v in captured_prefix),_array(out,"float32",1)))

    def interp_pv_diagnostics(self):
        """Four tropopause fields: mpas_pv_diagnostics.F:753-819."""
        import cupy as cp
        for src,dst in (("uReconstructZonal","u_pv"),
                        ("uReconstructMeridional","v_pv"),("theta","theta_pv")):
            self.interp_pv(self.diag[src],self.diag[dst])
        nc,nz,_ = self._sizes()
        ns=cp.int32(self.mesh["nCellsSolve"])
        v=cp.empty((int(nz),int(nc)+1),dtype=cp.float32,order="F")
        self._kernel("vertex_average",int(ns*nz),(*self._mesh_args(),ns,
            self._real(self.diag,"vorticity"),v))
        self.interp_pv(v,self.diag["vort_pv"])

    def atm_compute_pv_diagnostics(self, *, captured_top_velp=None):
        """Moist conversion, halos, PV and tropopause: mpas_pv_diagnostics.F:1253-1324."""
        import cupy as cp
        top=self._top(captured_top_velp)
        nc,nz,_=self._sizes()
        self._kernel("prepare",int(nc*nz),(cp.int32(nc*nz),
            self._real(self.state,"theta_m"),self._qv(),self._real(self.state,"rho_zz"),
            self._real(self.mesh,"zz"),self.rvord,self._real(self.diag,"theta"),self._real(self.diag,"rho")))
        self._halo(self.diag,("theta","uReconstructX","uReconstructY","uReconstructZ"))
        self._halo(self.state,("w",))
        self.calc_epv(captured_top_velp=top)
        self._halo(self.diag,("ertel_pv",))
        self.flood_fill_tropo()
        self.interp_pv_diagnostics()

    def calc_pv_budget(self, *, captured_top_velp=None, captured_friction_top_velp=None,
                       captured_reconstructed_halos=None):
        """Six diabatic terms and friction; native side effects: mpas_pv_diagnostics.F:1326-1573."""
        import cupy as cp
        top=self._top(captured_top_velp)
        friction_top=self._top(captured_friction_top_velp)
        nc,nz,me=self._sizes()
        ns=cp.int32(self.mesh["nCellsSolve"])
        # mpas_pv_diagnostics.F:1510-1552: reconstruction writes owned cells.
        # calc_gradxu_cell reads Cartesian velocities only in the owned column;
        # only w and scalar horizontal differences need exchanged neighbors.
        # Unwritten reconstructed halo storage is therefore not consumed.
        for scheme,name in zip(SCHEMES,("rthratenlw","rthratensw","rthblten", "rthcuten","dtheta_dt_mp","dtheta_dt_mix")):
            pool=self.tend_physics if scheme in ("lw","sw","bl","cu") else self.diag
            out=self._real(self.diag,"depv_dt_"+scheme)
            value=pool.get(name)
            if value is None:
                out[:,:int(ns)].fill(0.0)
            else:
                self._dot(value,out,top)
        # Native full-array expression includes halos and sentinel storage.
        out=self._real(self.diag,"depv_dt_diab")
        self._kernel("diab_sum",out.size,(cp.int32(out.size),
            *(self._real(self.diag,"depv_dt_"+s) for s in SCHEMES),out))
        ne,nv=cp.int32(self.mesh["nEdges"]),cp.int32(self.mesh["nVertices"])
        phys=self._real(self.diag,"tend_u_phys")
        self._kernel("uncouple_u",int(ne*nz),(cp.int32(ne*nz),self._real(self.tend,"u_euler"),
            self._real(self.diag,"rho_edge"),phys))
        vort=cp.empty((int(nz),int(nv)+1),dtype=cp.float32,order="F")
        self.vertical_curl(phys, vort)
        xyz=tuple(cp.empty((int(nz),int(nc)+1),dtype=cp.float32,order="F") for _ in range(3))
        if captured_reconstructed_halos is not None:
            for dst,src in zip(xyz,captured_reconstructed_halos):
                dst[...] = _array(src,"float32",2)
        self._kernel("reconstruct",int(ns*nz),(ns,nz,me,self._int("nEdgesOnCell",1),
            self._int("edgesOnCell",2),self._real(self.mesh,"coeffs_reconstruct",3),phys,*xyz))
        w=self._real(self.tend,"w_euler")
        self._kernel("uncouple_w",int(nc*(nz+1)),(nc,nz,self._real(self.diag,"rho"),w))
        self._dot(self.diag["theta"],self.diag["depv_dt_fric"],friction_top,
            velocity=(*xyz,w,vort),mode=1)

    def atm_compute_pv_budget_diagnostics(self, **captures):
        """Budget conversion and exchange: mpas_pv_diagnostics.F:1575-1648."""
        import cupy as cp
        self._top(captures.get("captured_top_velp"))
        self._top(captures.get("captured_friction_top_velp"))
        nc,nz,_=self._sizes()
        self._kernel("mix_theta",int(nc*nz),(cp.int32(nc*nz),self._real(self.tend,"theta_euler"),
            self._qv(),self.rvord,self._real(self.diag,"dtheta_dt_mix")))
        self._halo(self.tend_physics,("rthratenlw","rthratensw","rthcuten","rthblten"))
        self._halo(self.diag,("dtheta_dt_mp","dtheta_dt_mix","tend_u_phys"))
        self._halo(self.tend,("u_euler","w_euler"))
        self.calc_pv_budget(**captures)
        for s in ("diab","fric"):
            self.interp_pv(self.diag["depv_dt_"+s],self.diag["depv_dt_"+s+"_pv"])

    def compute(self, requested=None, **captures):
        """Stream-demand dispatch: mpas_pv_diagnostics.F:82-152."""
        if requested is None:
            if self.utils is None:
                raise ValueError("PV diagnostics require stream queries or explicit requested fields")
            requested = [name for name in PV_FIELDS+BUDGET_FIELDS
                         if self.utils.field_will_be_written(name)]
        needed=set(requested)
        if needed.intersection(PV_FIELDS+BUDGET_FIELDS):
            self.atm_compute_pv_diagnostics(captured_top_velp=captures.get("captured_top_velp"))
        if needed.intersection(BUDGET_FIELDS):
            self.atm_compute_pv_budget_diagnostics(**captures)
