"""Emit the same arithmetic using host glibc, to isolate libm from translation.

This is an oracle witness, never a forecast backend or CUDA fallback.
"""
from pathlib import Path
from hexcore.mpasphys.rrfs import ruc_lsm_8c9f1a8 as ruc
ROOT=Path(__file__).resolve().parents[4]
head='''#include <cmath>
#include <cstdlib>
#define RUCDEV
#define __constant__
float __fadd_rn(float a,float b){return a+b;}
float __fsub_rn(float a,float b){return a-b;}
float __fmul_rn(float a,float b){return a*b;}
float __fdiv_rn(float a,float b){return a/b;}
float __fsqrt_rn(float a){return sqrtf(a);}
float gfk_pow(float a,float b){return powf(a,b);}
float gfk_exp_fma(float a){return expf(a);}
float gfk_log(float a){return logf(a);}
float ruc_log10f_rn(float a){return log10f(a);}
float sfire_tanhf(float a){return tanhf(a);}
float glibc_cosf(float a){return cosf(a);}
void atomicOr(int *p,int x){*p|=x;}
'''
s=ruc.CUDA_SOURCE.replace('extern "C" __global__ void ruc_', 'extern "C" void ruc_').replace('if(blockIdx.x!=0 || threadIdx.x!=0)return;', '').replace('extern "C" __global__ void oracle_','extern "C" void translated_').replace('int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=nc)return;', 'int c=0;')
(ROOT/'scratch/native/translated.cpp').write_text(head+s)
