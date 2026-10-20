// Arithmetic order follows rw-mpas/src/init/hinterp.rs. Compile without FMA or FTZ.
#define MSG (-1.0e30f)
__device__ float at(const float* a,int nx,int i,int j) {
  i=(i-1+nx)%nx; return a[(long long)(j-1)*nx+i];
}
__device__ bool bad(const float* a,const float* mask,int nx,int i,int j,float mv) {
  return at(a,nx,i,j)==MSG || (mv>=0.0f && at(mask,nx,i,j)==mv);
}
__device__ float oned(float x,float a,float b,float c,float d,int ftz) {
  float out=0.0f;
  if(x==0.0f) out=b; else if(x==1.0f) out=c;
  float bc=__fmul_rn(b,c),ad=__fmul_rn(a,d);
  if(ftz) {if(fabsf(bc)<0x1p-126f)bc=0.0f;if(fabsf(ad)<0x1p-126f)ad=0.0f;}
  if((__float_as_uint(bc)&0x7fffffffU)!=0) {
    if((__float_as_uint(ad)&0x7fffffffU)==0) {
      if(a==0.0f&&d==0.0f) out=b*(1.0f-x)+c*x;
      else if(a!=0.0f) out=b+x*(0.5f*(c-a)+x*(0.5f*(c+a)-b));
      else if(d!=0.0f) out=c+(1.0f-x)*(0.5f*(b-d)+(1.0f-x)*(0.5f*(b+d)-c));
    } else out=(1.0f-x)*(b+x*(0.5f*(c-a)+x*(0.5f*(c+a)-b)))
                  +x*(c+(1.0f-x)*(0.5f*(b-d)+(1.0f-x)*(0.5f*(b+d)-c)));
  }
  return out;
}
__device__ float sixteen(float xx,float yy,const float*a,const float*m,int nx,int ny,float mv,int ftz) {
  if((int)xx < -2 || (int)xx > nx+3 || (int)yy < 1 || (int)yy>ny) return MSG;
  int i=(int)(xx+0.00001f),j=(int)(yy+0.00001f);
  float x=xx-i,y=yy-j;
  if(fabsf(x)>0.0001f || fabsf(y)>0.0001f) {
    float st[4][4]; bool fail=false;
    for(int k=0;k<4;k++)for(int l=0;l<4;l++){
      int kk=min(max(i+k-1,-2),nx+3),ll=min(max(j+l-1,1),ny);
      st[k][l]=at(a,nx,kk,ll);
      if(mv>=0.0f && at(m,nx,kk,ll)==mv)fail=true;
      if(st[k][l]==0.0f)st[k][l]=1.0e-20f;
      if(st[k][l]==MSG)fail=true;
    }
    if(fail)return MSG;
    float a1=oned(x,st[0][0],st[1][0],st[2][0],st[3][0],ftz);
    float b1=oned(x,st[0][1],st[1][1],st[2][1],st[3][1],ftz);
    float c1=oned(x,st[0][2],st[1][2],st[2][2],st[3][2],ftz);
    float d1=oned(x,st[0][3],st[1][3],st[2][3],st[3][3],ftz);
    float out=oned(y,a1,b1,c1,d1,ftz);
    return out==1.0e-20f?0.0f:out;
  }
  if(i>=-2&&i<=nx+3&&j>=1&&j<=ny&&!bad(a,m,nx,i,j,mv))return at(a,nx,i,j);
  return MSG;
}
__device__ float four(float xx,float yy,const float*a,const float*m,int nx,int ny,float mv) {
  int lx=(int)floorf(xx),hx=(int)ceilf(xx),ly=(int)floorf(yy),hy=(int)ceilf(yy);
  if(lx < -2 || hx>nx+3 || ly<1 || hy>ny) return MSG;
  if(bad(a,m,nx,lx,ly,mv)||bad(a,m,nx,hx,ly,mv)||bad(a,m,nx,lx,hy,mv)||bad(a,m,nx,hx,hy,mv))return MSG;
  if(lx==hx){if(ly==hy)return at(a,nx,lx,ly);return at(a,nx,lx,ly)*(hy-yy)+at(a,nx,lx,hy)*(yy-ly);}
  if(ly==hy)return at(a,nx,lx,ly)*(hx-xx)+at(a,nx,hx,ly)*(xx-lx);
  return (yy-ly)*(at(a,nx,lx,hy)*(hx-xx)+at(a,nx,hx,hy)*(xx-lx))
       +(hy-yy)*(at(a,nx,lx,ly)*(hx-xx)+at(a,nx,hx,ly)*(xx-lx));
}
__device__ float dist(float xx,float yy,int px,int py) {
  float dx=xx-px,dy=yy-py;return fmaxf(1.0f-sqrtf(dx*dx+dy*dy),0.0f);
}
__device__ float weighted(float xx,float yy,const float*a,const float*m,int nx,int ny,float mv) {
  int lx=(int)floorf(xx),hx=(int)ceilf(xx),ly=(int)floorf(yy),hy=(int)ceilf(yy);
  float w00=dist(xx,yy,lx,ly),w01=dist(xx,yy,lx,hy),w10=dist(xx,yy,hx,ly),w11=dist(xx,yy,hx,hy);
  if(lx < -2 || hx>nx+3 || ly<1 || hy>ny) {
    if(xx>-2.5f&&lx < -2)lx=hx=-2;else if(xx<nx+3.5f&&hx>nx+3)lx=hx=nx+3;
    if(yy>0.5f&&ly<1)ly=hy=1;else if(yy<ny+0.5f&&hy>ny)ly=hy=ny;
    if(lx < -2 || hx>nx+3 || ly<1 || hy>ny)return MSG;
  }
  if(bad(a,m,nx,lx,ly,mv))w00=0.0f;if(bad(a,m,nx,lx,hy,mv))w01=0.0f;
  if(bad(a,m,nx,hx,ly,mv))w10=0.0f;if(bad(a,m,nx,hx,hy,mv))w11=0.0f;
  if(w00==0.0f&&w01==0.0f&&w10==0.0f&&w11==0.0f)return MSG;
  return (w00*at(a,nx,lx,ly)+w01*at(a,nx,lx,hy)+w10*at(a,nx,hx,ly)+w11*at(a,nx,hx,hy))/(w00+w01+w10+w11);
}
__device__ int nint(float v){return v>=0.0f?(int)floorf(v+0.5f):(int)ceilf(v-0.5f);}
__device__ void locate(float lat,float lon,const float*p,int nx,int ny,float&x,float&y) {
  // Mesh input is f32 radians, as in the Rust capsule reader.
  lat=lat*__uint_as_float(0x42652ee0);lon=lon*__uint_as_float(0x42652ee0);
  x=(lon-p[1])/p[3]+1.0f;y=(lat-p[0])/p[2]+1.0f;
  if(x<0.5f){x=((lon+360.0f)-p[1])/p[3]+1.0f;y=(lat-p[0])/p[2]+1.0f;}
  else if(x>=nx+0.5f){x=((lon-360.0f)-p[1])/p[3]+1.0f;y=(lat-p[0])/p[2]+1.0f;}
  if(y<0.5f)y=1.0f;else if(y>=ny+0.5f)y=(float)ny;
}
extern "C" __global__ void horizontal(const float*values,const long long*offsets,const int*meta,const float*proj,
 const float*lat,const float*lon,const int*land,int np,int ns,float*out,unsigned char*search) {
  long long ix=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(ix>=(long long)np*ns)return;
  int s=ix/np,c=ix%np;const int*q=meta+s*5;const float*p=proj+s*6;
  int nx=q[0],ny=q[1],methods=q[2],flag=q[3],ftz=q[4];
  if(flag>=0&&land[c]==flag){out[ix]=p[5];return;}
  const float*a=values+offsets[s*2],*m=values+offsets[s*2+1];float x,y;
  locate(lat[c],lon[c],p,nx,ny,x,y);float result=MSG;
  while(methods && result==MSG){int method=methods&15;methods>>=4;
    if(method==1)result=sixteen(x,y,a,m,nx,ny,p[4],ftz);
    else if(method==2)result=four(x,y,a,m,nx,ny,p[4]);
    else if(method==3)result=weighted(x,y,a,m,nx,ny,p[4]);
    else if(method==4){search[ix]=1;break;}
    else if(method==5){int i=nint(x),j=nint(y);if(i>=-2&&i<=nx+3&&j>=1&&j<=ny&&!bad(a,m,nx,i,j,p[4]))result=at(a,nx,i,j);}
  }
  out[ix]=result;
}
#ifndef SEARCH_CAP
#define SEARCH_CAP 256
#endif
__device__ bool enqueue(int id,int*queue,int&tail){
  for(int k=0;k<tail;k++)if(queue[k]==id)return true;
  if(tail==SEARCH_CAP)return false;queue[tail++]=id;return true;
}
extern "C" __global__ void search_horizontal(const float*values,const long long*offsets,const int*meta,const float*proj,
 const float*lat,const float*lon,int np,const long long*indices,int n,float*out,unsigned char*status){
 int tid=blockIdx.x*blockDim.x+threadIdx.x;if(tid>=n)return;long long ix=indices[tid];
 int s=ix/np,c=ix%np;const int*q=meta+s*5;const float*p=proj+s*6;
 int nx=q[0],ny=q[1],width=nx+6;const float*a=values+offsets[s*2],*m=values+offsets[s*2+1];
 float x,y;locate(lat[c],lon[c],p,nx,ny,x,y);int sx=nint(x),sy=nint(y);
 status[tid]=0;if(sx < -2||sx>nx+3||sy<1||sy>ny){out[ix]=MSG;return;}
 int queue[SEARCH_CAP],head=0,tail=1;queue[0]=(sy-1)*width+sx+2;
 int fi=0,fj=0;bool found=false,ok=true;
 while(head<tail){int id=queue[head++],i=id%width-2,j=id/width+1;bool hit=!bad(a,m,nx,i,j,p[4]);
   if(i>-2)ok=ok&&enqueue(id-1,queue,tail);if(i<nx+3)ok=ok&&enqueue(id+1,queue,tail);
   if(j>1)ok=ok&&enqueue(id-width,queue,tail);if(j<ny)ok=ok&&enqueue(id+width,queue,tail);
   if(!ok){status[tid]=1;return;}if(hit){fi=i;fj=j;found=true;break;}
 }
 if(!found){out[ix]=MSG;return;}
 float dx=fi-x,dy=fj-y,distance=dx*dx+dy*dy,result=at(a,nx,fi,fj);
 while(head<tail){int id=queue[head++],i=id%width-2,j=id/width+1;
   float dx=i-x,dy=j-y,d=dx*dx+dy*dy;
   if(!bad(a,m,nx,i,j,p[4])&&d<distance){distance=d;result=at(a,nx,i,j);}
 }
 out[ix]=result;
}
