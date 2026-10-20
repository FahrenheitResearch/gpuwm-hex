"""Ordered physical-height mass integrals and rate interpolation on one CUDA device.

The default on CUDA hosts; an explicit CPU reference remains in moving_vertical.
CPU-only remapping with hidden devices never imports CuPy.
Each target/interface thread visits CSR donors in their stored order, matching
NumPy bincount's additions. No atomics, reassociation, or fused multiply-adds.
The geometry belongs to one immutable HeightConservativeMap; its device copy
is retained for that map's lifetime. Dynamic products are uploaded per call.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    return cp.RawKernel(r'''
    extern "C" __global__ void cumulative_integral(
        const long long *ptr, const long long *cols, const double *area,
        const unsigned char *layer, const double *sample,
        const double *source_z, const double *centres,
        const double *means, const double *slopes, const double *cumulative,
        const double *ratio, const double *below,
        double *out, const int ns, const int nt, const int levels,
        const long long nnz, const int aligned, const int extend) {
        const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= (long long)(levels+1)*nt) return;
        const int l = i / nt, row = i % nt;
        if(aligned && l == 0) {out[i]=0.;return;}
        double sum = 0.;
        for(long long j=ptr[row]; j<ptr[row+1]; ++j) {
            const long long c=cols[j];
            double carried;
            if(l==levels) carried=cumulative[(long long)levels*ns+c];
            else {
                const long long q=(long long)l*nnz+j;
                const long long k=layer[q], source=k*ns+c;
                const double z=sample[q], lower=source_z[source], centre=centres[source];
                const double dz=z-centre, dl=lower-centre;
                carried=(cumulative[source]+means[source]*(z-lower))
                    +(.5*slopes[source])*(dz*dz-dl*dl);
                if(extend && z<source_z[c]) carried=-ratio[c]*below[q];
            }
            sum += area[j]*carried;
        }
        out[i]=sum;
    }
    ''', 'cumulative_integral', options=('--fmad=false',))


def cumulative_integral(weights, means, slopes, cumulative, values, density):
    import cupy as cp
    # The class's source/target nominal level counts are equal in the moving
    # forecast. General remapping with unequal counts retains the CPU path.
    ns, nt = len(weights.source_area), len(weights.target_area)
    levels = len(weights.target_dzw)
    cache = getattr(weights, '_cuda_cumulative_geometry', None)
    device = cp.cuda.runtime.getDevice()
    if cache is None or cache['device'] != device:
        matrix = weights.overlap.tocsr()
        # COO traversal used by the host must be identical to CSR order.
        if not (np.array_equal(matrix.indices, weights.cols) and
                np.array_equal(np.repeat(np.arange(nt), np.diff(matrix.indptr)), weights.rows)):
            raise ValueError('CUDA cumulative remap requires ordered CSR donors')
        arrays = (matrix.indptr.astype(np.int64), weights.cols.astype(np.int64),
                  weights.areas, weights.sample_layer, weights.sample_height,
                  weights.source_zgrid, weights.source_centres)
        cache = dict(device=device, arrays=tuple(cp.asarray(np.ascontiguousarray(x)) for x in arrays))
        weights._cuda_cumulative_geometry = cache
    extend = density is not None and weights.below_ground is not None
    if extend:
        # The below-ground extension is prepared from the current atmosphere;
        # it can change between remaps even when geometry is reused.
        ratio = cp.asarray(values[0]/density[0])
        below = cp.asarray(weights.below_ground)
    else:
        ratio = below = cp.empty(0, dtype=cp.float64)
    products = tuple(cp.asarray(np.ascontiguousarray(x)) for x in (means, slopes, cumulative))
    out = cp.empty((levels+1, nt), dtype=cp.float64)
    _kernel()(((out.size+127)//128,), (128,), (*cache['arrays'], *products, ratio, below, out,
        np.int32(ns), np.int32(nt), np.int32(levels), np.int64(len(weights.areas)),
        np.int32(weights.endpoint_alignment), np.int32(extend)))
    return cp.asnumpy(out)  # Required host result boundary; no extra synchronization.


@lru_cache(maxsize=1)
def _layer_kernel():
    import cupy as cp
    return cp.RawKernel(r'''
    extern "C" __global__ void layer_integral(
        const long long *ptr, const long long *cols, const double *area,
        const double *sample, const double *centres, const double *target_area,
        const double *values, double *out, const int ns, const int nt,
        const int source_levels, const int target_levels, const int middle,
        const int searches, const long long nnz) {
        const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= (long long)target_levels*middle*nt) return;
        const int row=i%nt, channel=(i/nt)%middle, level=i/((long long)nt*middle);
        double sum=0.;
        for(long long donor=ptr[row]; donor<ptr[row+1]; ++donor) {
            const long long cell=cols[donor];
            const double height=.5*(sample[(long long)level*nnz+donor]
                                 +sample[(long long)(level+1)*nnz+donor]);
            int lo=0, hi=source_levels-1;
            for(int search=0;search<searches;++search) {
                const int mid=(lo+hi+1)/2;
                if(centres[(long long)mid*ns+cell]<=height) lo=mid;
                else hi=mid-1;
            }
            const int last=source_levels>1 ? source_levels-2 : 0;
            const int k=lo<last ? lo : last;
            const int upper=k+1<source_levels ? k+1 : source_levels-1;
            const double lower=centres[(long long)k*ns+cell];
            const double span=centres[(long long)upper*ns+cell]-lower;
            double fraction=(height-lower)/(span>0. ? span : 1.);
            fraction=fraction<0. ? 0. : (fraction>1. ? 1. : fraction);
            const double a=values[((long long)k*middle+channel)*ns+cell];
            const double b=values[((long long)upper*middle+channel)*ns+cell];
            const double carried=a+fraction*(b-a);
            sum+=area[donor]*carried;
        }
        out[i]=sum/target_area[row];
    }
    ''', 'layer_integral', options=('--fmad=false',))


def layer_integral(weights, values):
    """Interpolate current rates in height, then sum donors in CPU order."""
    import cupy as cp
    ns, nt = len(weights.source_area), len(weights.target_area)
    source_levels, target_levels = len(weights.source_dzw), len(weights.target_dzw)
    middle = int(np.prod(values.shape[1:-1], dtype=np.int64))
    device = cp.cuda.runtime.getDevice()
    cache = getattr(weights, '_cuda_rate_geometry', None)
    if cache is None or cache['device'] != device:
        matrix = weights.overlap.tocsr()
        if not (np.array_equal(matrix.indices, weights.cols) and
                np.array_equal(np.repeat(np.arange(nt), np.diff(matrix.indptr)), weights.rows)):
            raise ValueError('CUDA rate remap requires ordered CSR donors')
        arrays = (matrix.indptr.astype(np.int64), weights.cols.astype(np.int64),
                  weights.areas, weights.sample_height, weights.source_centres, weights.target_area)
        cache = dict(device=device, arrays=tuple(cp.asarray(np.ascontiguousarray(x)) for x in arrays))
        weights._cuda_rate_geometry = cache
    source = cp.asarray(np.ascontiguousarray(values))
    out = cp.empty((target_levels, middle, nt), dtype=cp.float64)
    _layer_kernel()(((out.size+127)//128,), (128,), (*cache['arrays'], source, out,
        np.int32(ns), np.int32(nt), np.int32(source_levels), np.int32(target_levels),
        np.int32(middle), np.int32(source_levels.bit_length()), np.int64(len(weights.areas))))
    return cp.asnumpy(out).reshape((target_levels, *values.shape[1:-1], nt))
