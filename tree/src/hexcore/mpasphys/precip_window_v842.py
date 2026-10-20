"""RP6 surface precipitation window (correct mode with config_pbl_interval).

Native correct mode (upstream patch, mpas-rad-pbl-lsm-seaice): physics_driver adds every step's rainncv, raincv,
snowncv, graupelncv and sr*rainncv to rainncv_bl, raincv_bl, snowncv_bl, graupelncv_bl and frozen_bl
(mpas_atmphys_lsm_shared.F precip_window_add), and lsm_from_MPAS, driver_lsm_noahmp and seaice_from_MPAS read
those totals and sr_bl in place of the one-step fields; after the sea-ice call the window is emptied
(precip_window_reset). WRF does the same with RAINBL (module_surface_driver.F).

The port's native-cpu land-surface and sea-ice slots are pristine MPAS v8.4.2, so the driver hands them the
window instead: open copies the window totals into the native one-step fields just before the land-surface call
and close copies the step's own words back just after the sea-ice call. The pristine slots then read exactly the
words native correct mode reads, and every later reader (surface diagnostics, PBL, GWDO, convection, output) sees
the step's own precipitation. The arithmetic is the native routine's, in single precision with no contraction,
flush-to-zero or approximate division, so the window words equal the native strict build's.

The window lives on the CUDA device; the pristine Registry restart stream cannot hold it (native correct mode adds
it to the restart stream under the bl_precip_window_in package), so a native-file restart of a correct-mode run
with config_pbl_interval is refused at the door (runtime_v842) rather than dropping the window's precipitation.
"""
from __future__ import annotations
from hexcore import numerics as _hex_numerics

from typing import Any, Callable, Mapping

import numpy as np

#: The native one-step fields the window replaces, and the window total each one receives.
WINDOW_FIELDS = (("rainncv", "rainncv"), ("raincv", "raincv"), ("snowncv", "snowncv"),
                 ("graupelncv", "graupelncv"), ("sr", "sr"))
#: Window arrays: the four totals, frozen grid-scale water (sr*rainncv) and the frozen fraction sr_bl.
WINDOW_ARRAYS = ("rainncv", "raincv", "snowncv", "graupelncv", "frozen", "sr")
#: Microphysics schemes whose Registry rows carry snowncv and graupelncv (Registry.xml packages
#: mp_thompson_in;mp_thompson_aers_in;mp_wsm6_in).
SNOW_GRAUPEL_SCHEMES = ("mp_thompson", "mp_thompson_aerosols", "mp_wsm6")
SNAPSHOT_PREFIX = "precipitation_window."

#: precip_window_add and precip_window_reset of mpas_atmphys_lsm_shared.F, word for word. sr*rainncv is rounded
#: before it is added (no fused multiply-add, as the strict recipe's -ffp-contract=off).
SOURCE = r'''
extern "C" __global__ void precip_window_add(const int n, const int have_nc, const int have_c, const int have_sg,
    const float* __restrict__ rainncv, const float* __restrict__ raincv, const float* __restrict__ snowncv,
    const float* __restrict__ graupelncv, const float* __restrict__ sr,
    float* rainncv_bl, float* raincv_bl, float* snowncv_bl, float* graupelncv_bl, float* frozen_bl,
    float* sr_bl, int* nsteps_bl)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    nsteps_bl[i] = nsteps_bl[i] + 1;
    if (have_nc) {
        rainncv_bl[i] = rainncv_bl[i] + rainncv[i];
        const float frozen = sr[i] * rainncv[i];
        frozen_bl[i] = frozen_bl[i] + frozen;
    }
    if (have_c) raincv_bl[i] = raincv_bl[i] + raincv[i];
    if (have_sg) {
        snowncv_bl[i] = snowncv_bl[i] + snowncv[i];
        graupelncv_bl[i] = graupelncv_bl[i] + graupelncv[i];
    }
    float ratio = sr[i];
    if (nsteps_bl[i] > 1 && rainncv_bl[i] > 0.0f) {
        const float q = frozen_bl[i] / rainncv_bl[i];
        ratio = q < 1.0f ? q : 1.0f;
    }
    sr_bl[i] = ratio;
}

extern "C" __global__ void precip_window_reset(const int n, float* rainncv_bl, float* raincv_bl,
    float* snowncv_bl, float* graupelncv_bl, float* frozen_bl, int* nsteps_bl)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    rainncv_bl[i] = 0.0f; raincv_bl[i] = 0.0f; snowncv_bl[i] = 0.0f; graupelncv_bl[i] = 0.0f;
    frozen_bl[i] = 0.0f; nsteps_bl[i] = 0;
}
'''
OPTIONS = ("--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
THREADS = 128


def compile_strict(source: str) -> Any:
    """Direct NVRTC with the strict options, as compile_v842.MpasPhysicsKernelCache does.

    Breakage this prevents: a cupy.RawModule build of the same source flushed subnormal precipitation words
    (about 6e-39 mm) to zero on the 5090, so a one-step window was no longer the step's own words and the port
    left native from the first drizzle onwards (x1.2562, correct mode, interval = dt: 7,320,211 words in 2 h).
    """
    from cupy.cuda import compiler, function
    image, _ = compiler.compile_using_nvrtc(source, options=_hex_numerics.options(OPTIONS), cache_in_memory=True)
    module = function.Module()
    module.load(image)
    return module


class SurfacePrecipitationWindow:
    """Window totals for the solve cells; the native routines precip_window_add/precip_window_reset."""

    def __init__(self, native: Any, cells: int, selectors: Mapping[str, str], cp: Any):
        if cells < 1:
            raise ValueError("the precipitation window needs solve cells")
        self.native, self.cp, self.cells = native, cp, int(cells)
        self.device_pools = None
        # mpas_atmphys_driver.F precip_window_accumulate: the species a scheme pair actually produces.
        self.have_nc = selectors["microphysics"] != "off"
        self.have_c = selectors["convection"] != "off"
        self.have_sg = selectors["microphysics"] in SNOW_GRAUPEL_SCHEMES
        self.window = {name: cp.zeros(self.cells, dtype=cp.float32) for name in WINDOW_ARRAYS}
        self.steps = cp.zeros(self.cells, dtype=cp.int32)
        self.saved: dict[str, np.ndarray] | None = None
        module = compile_strict(SOURCE)
        self._add = module.get_function("precip_window_add")
        self._reset = module.get_function("precip_window_reset")
        self._grid = ((self.cells + THREADS - 1) // THREADS,)

    def present(self) -> tuple[str, ...]:
        """One-step fields the pristine slots read for this scheme pair (sr only with microphysics)."""
        names = []
        if self.have_nc:
            names += ["rainncv", "sr"]
        if self.have_c:
            names.append("raincv")
        if self.have_sg:
            names += ["snowncv", "graupelncv"]
        return tuple(names)

    def _solve_words(self, name: str) -> np.ndarray:
        if self.device_pools is None:
            host = self.native.get_field("diag_physics", name)
        else:
            pools = self.device_pools
            previous, pools.level = pools.level, 1
            try:
                host = pools.field("diag_physics", name)
            finally:
                pools.level = previous
        if host.ndim != 1 or host.shape[0] < self.cells or host.dtype != np.float32:
            raise ValueError(f"native diag_physics.{name} is not a single-precision cell field")
        return host

    def accumulate(self, call: Any = None) -> None:
        """precip_window_add for this step; mpas_atmphys_driver.F physics_driver (l_precip_bl)."""
        cp = self.cp
        sr = cp.asarray(self._solve_words("sr")[:self.cells])
        fields = {name: (cp.asarray(self._solve_words(name)[:self.cells]) if name in self.present() else sr)
                  for name in ("rainncv", "raincv", "snowncv", "graupelncv")}
        w = self.window
        self._add(self._grid, (THREADS,),
                  (np.int32(self.cells), np.int32(self.have_nc), np.int32(self.have_c), np.int32(self.have_sg),
                   fields["rainncv"], fields["raincv"], fields["snowncv"], fields["graupelncv"], sr,
                   w["rainncv"], w["raincv"], w["snowncv"], w["graupelncv"], w["frozen"], w["sr"], self.steps))

    def open(self, call: Any = None) -> None:
        """Hand the window to the land-surface and sea-ice calls (native reads the *_bl fields there)."""
        if self.saved is not None:
            raise RuntimeError("surface precipitation window is already open")
        saved = {}
        for name in self.present():
            host = self._solve_words(name)
            saved[name] = host[:self.cells].copy()
            host[:self.cells] = (self.window[name].get() if self.device_pools is None
                                 else self.window[name])
            self._put(name, host)
        self.saved = saved

    def close(self, call: Any = None) -> None:
        """Put the step's own words back and empty the window (precip_window_reset)."""
        if self.saved is None:
            raise RuntimeError("surface precipitation window is not open")
        for name, words in self.saved.items():
            host = self._solve_words(name)
            host[:self.cells] = words
            self._put(name, host)
        self.saved = None
        w = self.window
        self._reset(self._grid, (THREADS,), (np.int32(self.cells), w["rainncv"], w["raincv"], w["snowncv"],
                                             w["graupelncv"], w["frozen"], self.steps))

    def _put(self, name: str, host: np.ndarray) -> None:
        if self.device_pools is not None:
            self.device_pools.write_field("diag_physics", name, host, level=1)
            return
        _, _, levels = self.native._fields["diag_physics", name]
        self.native.put_field("diag_physics", name, host, level=1, nlevels=levels)

    def operations(self) -> Mapping[str, Callable[[Any], None]]:
        """Driver glue names (driver_v842.SURFACE_PRECIPITATION_WINDOW)."""
        return {"accumulate_surface_precipitation": self.accumulate,
                "open_surface_precipitation_window": self.open,
                "close_surface_precipitation_window": self.close}

    def snapshot(self) -> dict[str, np.ndarray]:
        """Window words for a driver checkpoint (between steps the window is never open)."""
        if self.saved is not None:
            raise RuntimeError("surface precipitation window is open")
        state = {SNAPSHOT_PREFIX + name: value.get() for name, value in self.window.items()}
        state[SNAPSHOT_PREFIX + "nsteps"] = self.steps.get()
        return state

    def keys(self) -> set[str]:
        return {SNAPSHOT_PREFIX + name for name in (*WINDOW_ARRAYS, "nsteps")}

    def restore(self, state: Mapping[str, np.ndarray]) -> None:
        if set(state) != self.keys():
            raise ValueError("surface precipitation window snapshot differs")
        for name in WINDOW_ARRAYS:
            value = np.asarray(state[SNAPSHOT_PREFIX + name])
            if value.shape != (self.cells,) or value.dtype != np.float32:
                raise ValueError(f"surface precipitation window {name} layout differs")
        steps = np.asarray(state[SNAPSHOT_PREFIX + "nsteps"])
        if steps.shape != (self.cells,) or steps.dtype != np.int32:
            raise ValueError("surface precipitation window step count layout differs")
        for name in WINDOW_ARRAYS:
            self.window[name][...] = self.cp.asarray(state[SNAPSHOT_PREFIX + name])
        self.steps[...] = self.cp.asarray(steps)
        self.saved = None
