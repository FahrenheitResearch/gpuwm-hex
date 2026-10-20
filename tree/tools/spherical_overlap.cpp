// CPU accelerator for hexcore.moving_remap, compiled on the named CPU box.
// Great-circle clipping and triangle excess match the independent Python path.
#include <cmath>
#include <algorithm>
#include <cstdint>
struct V { double x,y,z; };
static V add(V a,V b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
static V sub(V a,V b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static V mul(V a,double s){return {a.x*s,a.y*s,a.z*s};}
static double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static V cross(V a,V b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
static V norm(V a){return mul(a,1/std::sqrt(dot(a,a)));}
extern "C" int spherical_overlaps(const double* source, int width,
 const int64_t* counts, const int64_t* ids, int nids,
 const double* target, int nt, const double* center, double radius, double* out){
 if(width>30 || nt>30 || nt<3) return 1;
 V c={center[0],center[1],center[2]};
 for(int z=0;z<nids;z++){
  int64_t id=ids[z]; int n=counts[id]; if(n>30||n<3)return 2;
  V p[64],q[64];
  for(int i=0;i<n;i++) {auto a=source+(id*width+i)*3;p[i]={a[0],a[1],a[2]};}
  for(int i=0;i<nt && n;i++){
   auto aa=target+i*3;auto bb=target+((i+1)%nt)*3;
   V a={aa[0],aa[1],aa[2]},b={bb[0],bb[1],bb[2]};
   V normal=norm(cross(a,sub(b,a)));if(dot(normal,c)<0)normal=mul(normal,-1);
   int m=0;
   for(int k=0;k<n;k++){
    V a=p[(k+n-1)%n],b=p[k];double da=dot(a,normal),db=dot(b,normal);
    if((da>=0)!=(db>=0))q[m++]=norm(mul(add(mul(a,db),mul(b,-da)),1/(db-da)));
    if(db>=0)q[m++]=b;
    if(m>=62)return 3;
   }
   n=m;for(int k=0;k<n;k++)p[k]=q[k];
  }
  double area=0;
  for(int k=1;k<n-1;k++)area+=2*std::atan2(dot(p[0],cross(sub(p[k],p[0]),sub(p[k+1],p[0]))),1+dot(p[0],p[k])+dot(p[0],p[k+1])+dot(p[k],p[k+1]));
  out[z]=std::abs(area)*radius*radius;
 }
 return 0;
}
