"""Post-acoustic interior damping during byte-only halo DMA.

The selected kernel is seeded from the shipped damping kernel. Only an
integer edge-selection predicate is added. Interior edges are owned edges
whose two endpoint cells are owned, so they neither read nor write a halo.
The pack precedes this update; boundary damping follows the receive fence.
"""
from __future__ import annotations

import numpy as np

from .cuda_fp32 import CUDA_FTZ_HELPERS
from .cuda_horizontal import _CUDA_SOURCE as _HORIZONTAL_SOURCE


def selected_source():
    kernel = _HORIZONTAL_SOURCE.split('extern "C" __global__ void divergence_damping_f32(', 1)[1]
    kernel = 'extern "C" __global__ void divergence_damping_selected_f32(' + kernel
    kernel = kernel.replace("const float* theta_m,", "const unsigned char* interior_edge,\n    const int select_interior,\n    const float* theta_m,", 1)
    kernel = kernel.replace("if (edge >= nedges) return;",
                            "if (edge >= nedges || int(interior_edge[edge]) != select_interior) return;", 1)
    return CUDA_FTZ_HELPERS + '\n__device__ __forceinline__ int lidx(const int k, const int i, const int n) { return k * n + i; }\n' + kernel


_CUDA_SOURCE = selected_source()


def interior_edges(cells_on_edge, *, n_owned_cells, n_owned_edges):
    """Classify integer connectivity; native missing endpoints stay boundary."""
    cells = np.asarray(cells_on_edge)
    if cells.ndim != 2 or cells.shape[1] != 2 or cells.dtype.kind not in "iu":
        raise ValueError("interior damping needs integer (nEdges,2) connectivity; prevents selecting a halo-dependent edge")
    if not 0 <= n_owned_edges <= len(cells) or n_owned_cells < 0:
        raise ValueError("owned extents are outside the local mesh; prevents interior writes into the halo")
    result = np.zeros(len(cells), dtype=np.uint8)
    endpoints = cells[:n_owned_edges]
    result[:n_owned_edges] = np.all((endpoints >= 0) & (endpoints < n_owned_cells), axis=1)
    return result


def launch_selected(horizontal, total, arguments, selection):
    mask, selected = selection
    cp = horizontal.cp
    if (not isinstance(mask, cp.ndarray) or mask.shape != (horizontal.nedges,)
            or mask.dtype != cp.uint8 or int(mask.device.id) != int(cp.cuda.Device().id)
            or selected not in (0, 1)):
        raise ValueError("selected damping needs a resident byte mask and selector 0 or 1; prevents overlapping halo writes")
    name = "divergence_damping_selected_f32"
    kernel = horizontal._kernels.get(name)
    if kernel is None:
        from .cuda_shape import resolve
        kernel = resolve(horizontal.kernel_cache, name, _CUDA_SOURCE,
                         "hexcore.partition_overlap_v841",
                         (("nlev", int(arguments[-4])), ("ncells", int(arguments[-3])),
                          ("nedges", int(arguments[-2]))))
        horizontal._kernels[name] = kernel
    if total:
        kernel(((total + 255) // 256,), (256,), (mask, np.int32(selected), *arguments))
