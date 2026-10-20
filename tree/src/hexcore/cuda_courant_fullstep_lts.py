"""Read-only per-column FP64 donor reductions, separate from native kernels."""
import numpy as np

CUDA_COLUMN_TRANSPORT = r'''
extern "C" __global__ void fullstep_column_transport(
    const int nk, const int nc, const int ne, const int width,
    const int *counts, const int *edges, const float *signs,
    const float *dv, const float *area, const float *rho,
    const float *uh, const float *ww, const float *rdzw,
    double *out, int *levels)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    double best = -1, hm = 0, vm = 0;
    int level = 0;
    bool invalid = !isfinite(area[c]) || area[c] <= 0 || counts[c] < 1 || counts[c] > width;
    for (int k=0; k<nk; ++k) {
        double h = 0;
        for (int s=0; s<counts[c] && s<width; ++s) {
            int e = edges[c*width+s];
            float sign = signs[c*width+s];
            if (e < 0 || e >= ne || (sign != 1 && sign != -1)) {invalid=true; continue;}
            double f = uh[k*ne+e], length = dv[e];
            invalid |= !isfinite(f) || !isfinite(length) || length <= 0;
            h += fmax(sign*f, 0.) * length / (double)area[c];
        }
        double density = rho[k*nc+c], rz = rdzw[k];
        double top = ww[(k+1)*nc+c], bottom = ww[k*nc+c];
        invalid |= !isfinite(density) || density <= 0 || !isfinite(rz) || rz <= 0 || !isfinite(top) || !isfinite(bottom);
        h /= density;
        double v = (fmax(top, 0.) + fmax(-bottom, 0.)) * rz / density;
        if (h+v > best) { best=h+v; level=k; }
        hm=fmax(hm,h); vm=fmax(vm,v);
    }
    out[c] = invalid || !isfinite(best) ? nan("") : best;
    out[nc+c]=hm; out[2*nc+c]=vm;
    levels[c]=level;
}
'''


class CudaColumnTransportMonitor:
    """Fixed-size scratch; never retain a source/flux array after observe."""
    def __init__(self, cp, mesh, n_levels):
        from .cuda_transport import _geometry
        counts, edges, _, _, signs, dv, area, width = _geometry(mesh)
        self.cp = cp
        self.geometry = counts, edges, signs, dv, area
        self.nc, self.ne, self.nk, self.width = int(area.size), int(dv.size), int(n_levels), width
        self.out = cp.empty((3, self.nc), dtype=cp.float64)
        self.level = cp.empty(self.nc, dtype=cp.int32)
        self.kernel = cp.RawKernel(CUDA_COLUMN_TRANSPORT, 'fullstep_column_transport', options=('--std=c++11',))

    def observe(self, rho_old, uh_avg, ww_avg, rdzw):
        cp = self.cp
        expected = [(self.nk,self.nc), (self.nk,self.ne), (self.nk+1,self.nc), (self.nk,)]
        values = [cp.asarray(x) for x in (rho_old, uh_avg, ww_avg, rdzw)]
        for x, shape in zip(values, expected):
            if x.shape != shape or x.dtype != cp.float32 or not x.flags.c_contiguous:
                raise ValueError(f'donor monitor requires contiguous FP32 {shape}, got {x.shape}/{x.dtype}')
        self.kernel(((self.nc+127)//128,), (128,),
            (np.int32(self.nk), np.int32(self.nc), np.int32(self.ne), np.int32(self.width),
             *self.geometry, *values, self.out, self.level))
        bad = cp.flatnonzero(~cp.isfinite(self.out[0]))
        if bad.size:
            raise ValueError(f'cell={int(bad[0].item())}: a finite local bound or conservative flux cannot be formed')
        return self.out, self.level
