"""N aerosol mass tracers on hex scalar FCT with conservative GPU sinks.

Species metadata defines kg/kg dry-air tracers outside moist thermodynamics.
The implicit finite-volume column operator uses the active PBL exch_h. Its
M-matrix preserves positivity without clipping or discarded mass.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = "gpuwm-hex.chemistry/v1"
# Module binding, not an instance attribute: the AOT pack inventory
# (cuda_backend/aot_build.py) evaluates each raw_kernel source argument in
# module scope, and `self.source` made the whole inventory refuse.
_SOURCE = Path(__file__).with_suffix(".cu").read_text()
_CONFIGS = {}
PRODUCT_SUFFIXES = ("_column_mass", "_surface_concentration", "_aod550",
                    "_dry_deposition", "_settled_mass")
RESERVED_FIELDS = frozenset("qv qc qr qi qs qg rho theta pressure exner w normal_u u_zonal v_meridional zgrid surface_pressure ter refl10cm t2 q2 u10 v10 tsk pblh hfx lh qfx swdown glw rainc rainnc snownc graupelnc tslb smois landmask ivgtyp xice snow snowh tke exch_h".split())


@dataclass(frozen=True)
class ChemistrySpecies:
    name: str
    dry_deposition_m_s: float
    settling_m_s: float
    mass_extinction_m2_kg: float

    def __post_init__(self):
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ValueError("chemistry species names must be lowercase identifiers")
        for key in ("dry_deposition_m_s", "settling_m_s", "mass_extinction_m2_kg"):
            value = getattr(self, key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 3.4028234663852886e38):
                raise ValueError(f"{self.name}.{key} must be finite, nonnegative and representable in FP32")


def load_config(path):
    path = Path(path).expanduser().resolve()
    raw = json.loads(path.read_text())
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"chemistry config must use {SCHEMA}")
    species = tuple(ChemistrySpecies(**row) for row in raw["species"])
    if not species or len({s.name for s in species}) != len(species):
        raise ValueError("chemistry config requires unique species rows")
    names = [s.name for s in species]
    products = ["aod550"] + [name+suffix for name in names for suffix in PRODUCT_SUFFIXES]
    if set(names) & RESERVED_FIELDS or len(set(names+products)) != len(names+products):
        raise ValueError("chemistry species overlap atmospheric scalars or product names")
    return path, raw, species



def product_slugs(forecast_dir):
    """Expand the completed run's species metadata into normal product slugs."""
    if forecast_dir is None:
        return ()
    receipt = Path(forecast_dir) / "chemistry-receipt.json"
    if not receipt.is_file():
        return ()
    raw = json.loads(receipt.read_text())
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"{receipt} must use {SCHEMA}")
    species = tuple(ChemistrySpecies(**row) for row in raw["species"])
    if not species or len({s.name for s in species}) != len(species):
        raise ValueError(f"{receipt} requires unique species rows")
    # The Rust WRF importer carries additional 2-D planes under wrf_<name>.
    # Use its store vocabulary so the real catalog can prove availability.
    return tuple("var:wrf_" + s.name + suffix for s in species
                 for suffix in PRODUCT_SUFFIXES) + ("var:wrf_aod550",)


def configure(path):
    """Register configuration as data; hash identity survives subprocesses."""
    from .species_row import (AEROSOL_MASS, SpeciesDeclaration, WSM6_SPECIES_ROW,
                              register_extended_row)
    from .physics_backend_admission import (FROZEN_WSM6_COLUMN_ROW,
                                            register_synthesized_row)
    path, raw, species = load_config(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    name = "chemistry_" + digest[:16]
    names = tuple(s.name for s in species)
    overlap = set(names) & set(WSM6_SPECIES_ROW.names())
    if overlap:
        raise ValueError(f"chemistry names overlap atmospheric scalars: {sorted(overlap)}")
    row = WSM6_SPECIES_ROW.with_extras(tuple(
        SpeciesDeclaration(s.name, AEROSOL_MASS, True, s.name.upper(),
                           "Aerosol mass mixing ratio per dry-air mass")
        for s in species))
    register_extended_row(row, scheme_alias=name)
    backend = replace(FROZEN_WSM6_COLUMN_ROW, name=name,
                      adapter_module=__name__, appended_scalar_names=names,
                      anchor_configuration_class=name, anchored=False,
                      unanchored_remedy="Whole forecast and observed plume qualification")
    _CONFIGS[name] = (path, raw, species, row, digest)
    return register_synthesized_row(backend)


def scheme_alias_for(backend):
    return backend


def seam_options_for(*, source_table=None):
    if source_table is not None:
        raise ValueError("use chemistry config source inputs, not a point-source table")
    return {}


from .thompson_column import pin_problem, pin_column_batch


class ChemistryTransport:
    """GPU data path. Source decoding/remapping belongs to the input provider."""
    def __init__(self, config, *, cache, area, nlev, ncells, solved=None):
        import cupy as cp
        import numpy as np
        self.path, self.raw, self.species, self.row, self.digest = config
        self.cache, self._np = cache, np
        self.area = cp.asarray(area, dtype=cp.float64)
        self.nlev, self.ncells = int(nlev), int(ncells)
        self.solved = self.ncells if solved is None else int(solved)
        self.offset = len(self.row.species) - len(self.species)
        self.surface_flux = {s.name: cp.zeros(self.ncells, cp.float32) for s in self.species}
        self.initial, self.boundary, self.driver = {}, None, None
        shape = (len(self.species), self.nlev, self.ncells)
        self.upper, self.rhs = (cp.empty(shape, cp.float64) for _ in range(2))
        self.dry = cp.zeros((len(self.species), self.ncells), cp.float64)
        self.settled = cp.zeros_like(self.dry)
        self.emitted = cp.zeros_like(self.dry)
        self.pending_emitted = cp.empty_like(self.dry)
        self.initial_mass_kg = None
        self.pending_dry, self.pending_settled = cp.empty_like(self.dry), cp.empty_like(self.dry)
        self.vd = cp.asarray([s.dry_deposition_m_s for s in self.species], dtype=cp.float32)
        self.vs = cp.asarray([s.settling_m_s for s in self.species], dtype=cp.float32)
        self.ext = cp.asarray([s.mass_extinction_m2_kg for s in self.species], dtype=cp.float32)
        self.invalid = cp.zeros(1, cp.int32)
        self.source = _SOURCE
        self.steps, self.provider_receipt = 0, None
        self.restart_provider_receipt = None

    def kernel(self, name, count, args):
        self.cache.raw_kernel(name, _SOURCE, module_key="hexcore.chem_transport")(
            ((count + 127)//128,), (128,), args)

    def step(self, atmosphere, *, rho, z, kh, dt):
        import cupy as cp
        I, F = self._np.int32, self._np.float32
        rho, z, kh = (cp.ascontiguousarray(a, dtype=cp.float32) for a in (rho, z, kh))
        expected = (self.nlev, self.ncells)
        if rho.shape != expected or kh.shape != expected or z.shape != (self.nlev+1, self.ncells):
            raise ValueError("chemistry density, diffusivity and vertical geometry shapes differ")
        if not math.isfinite(dt) or not 0 < dt <= 3.4028234663852886e38:
            raise ValueError("chemistry dt must be finite and positive")
        if self.boundary is not None:
            self.boundary(atmosphere, self, float(atmosphere.state.time_seconds) - dt, dt)
        flux = cp.ascontiguousarray(cp.stack([self.surface_flux[s.name] for s in self.species]), dtype=cp.float32)
        if flux.shape != (len(self.species), self.ncells) or not bool(cp.all(cp.isfinite(flux) & (flux >= 0))):
            raise ValueError("chemistry sources must be finite nonnegative kg m-2 s-1")
        self.pending_dry[...] = self.dry
        self.pending_settled[...] = self.settled
        self.pending_emitted[...] = self.emitted
        self.invalid.fill(0)
        self.kernel("chemistry_column", len(self.species)*self.ncells, (
            I(len(self.species)), I(self.nlev), I(self.ncells), I(self.solved), F(dt),
            rho, z, kh, self.vd, self.vs, flux, atmosphere.state.scalars[self.offset:],
            self.upper, self.rhs, self.pending_dry, self.pending_settled, self.pending_emitted, self.invalid))
        if int(self.invalid.get()[0]):
            raise FloatingPointError("chemistry column has invalid geometry, input or solution")

    def commit(self):
        self.dry, self.pending_dry = self.pending_dry, self.dry
        self.settled, self.pending_settled = self.pending_settled, self.settled
        self.emitted, self.pending_emitted = self.pending_emitted, self.emitted
        self.steps += 1

    def products(self, atmosphere, rho, z):
        import cupy as cp
        I = self._np.int32
        shape = (len(self.species), self.ncells)
        column, surface, aod = (cp.empty(shape, cp.float32) for _ in range(3))
        self.kernel("chemistry_products", len(self.species)*self.ncells, (
            I(len(self.species)), I(self.nlev), I(self.ncells), I(self.solved),
            rho, z, atmosphere.state.scalars[self.offset:], self.ext, column, surface, aod))
        arrays = {"aod550": cp.sum(aod, axis=0)}
        for i, s in enumerate(self.species):
            arrays.update({s.name+"_column_mass": column[i],
                           s.name+"_surface_concentration": surface[i],
                           s.name+"_aod550": aod[i],
                           s.name+"_dry_deposition": self.dry[i].astype(cp.float32),
                           s.name+"_settled_mass": self.settled[i].astype(cp.float32)})
        return arrays

    def receipt(self):
        return {"schema": SCHEMA, "config": str(self.path), "config_sha256": self.digest,
                "species": [s.__dict__ for s in self.species],
                "transport": "existing atmospheric monotonic scalar FCT",
                "mixing": "active PBL exch_h; conservative implicit finite volume",
                "dry_deposition": "species velocity; implicit lower boundary",
                "settling": "species downward velocity; implicit finite volume",
                "wet_scavenging": "not implemented", "committed_steps": self.steps,
                "provider": self.provider_receipt}


def build_column_backend(row, *, constructor, prep_geometry, kernel_cache,
                         gwdo_static=None, gwdo_kernel_cache=None, checkout=None,
                         cell_area_m2=None, seam_options=None, rollback_snapshot=True):
    from .cuda_arwen_physics_v841 import PersistentTwoPhaseCudaPhysicsBackendV841
    from .species_row import WSM6_SPECIES_ROW
    from .cuda_physics_prep_v841 import prepare_mpas_to_phys_cuda_v841

    class ChemistryBackend(PersistentTwoPhaseCudaPhysicsBackendV841):
        def _species_row(self, scalar_names):
            return WSM6_SPECIES_ROW

        def finish_step(self, *, atmosphere, scalar_names, dt, refl_10cm_due=False):
            update = super().finish_step(atmosphere=atmosphere, scalar_names=scalar_names,
                                        dt=dt, refl_10cm_due=refl_10cm_due)
            # Density must follow the RK endpoint that owns these scalar masses.
            # The phase-one PBL sounding belongs to the start of the step.
            columns = prepare_mpas_to_phys_cuda_v841(
                atmosphere, scalar_names=scalar_names,
                geometry=self._prep_geometry, kernel_cache=self._kernel_cache,
                post_rk_wsm6=True, phase2_only=True)
            fields = self._seam._driver.fields
            if "exch_h" not in fields:
                raise ValueError("active PBL publishes no exch_h for chemistry")
            kh = fields["exch_h"].reshape(self.chemistry.nlev, self.chemistry.ncells)
            self.chemistry.step(atmosphere, rho=columns.rho_dry, z=columns.z_p, kh=kh, dt=dt)
            return update

        def commit_step(self, **kwargs):
            result = super().commit_step(**kwargs)
            self.chemistry.commit()
            return result

        def diagnostic_snapshot(self, **kwargs):
            snapshot = super().diagnostic_snapshot(**kwargs)
            if self.chemistry.driver is None:
                return snapshot
            atmosphere = self.chemistry.driver.atmosphere
            columns = prepare_mpas_to_phys_cuda_v841(
                atmosphere, scalar_names=self.chemistry.row.names(),
                geometry=self._prep_geometry, kernel_cache=self._kernel_cache)
            extra = self.chemistry.products(atmosphere, columns.rho_dry, columns.z_p)
            return replace(snapshot, surface={**snapshot.surface, **extra},
                           metadata={**snapshot.metadata, "chemistry": self.chemistry.receipt()})

        def restart_state(self):
            state = dict(super().restart_state())
            state["chemistry"] = {
                "config_sha256": self.chemistry.digest,
                "dry": self.chemistry.dry.get(),
                "settled": self.chemistry.settled.get(),
                "steps": self.chemistry.steps,
                "emitted": self.chemistry.emitted.get(),
                "initial_mass_kg": self.chemistry.initial_mass_kg,
                "provider": self.chemistry.provider_receipt,
            }
            return state

        def restore_restart_state(self, payload):
            import cupy as cp
            chem = payload.get("chemistry")
            if chem is None or chem.get("config_sha256") != self.chemistry.digest:
                raise ValueError("chemistry restart config identity differs")
            arrays = {}
            for name in ("dry", "settled", "emitted"):
                array = cp.asarray(chem[name])
                target = getattr(self.chemistry, name)
                if (array.shape != target.shape or array.dtype != target.dtype
                        or not bool(cp.all(cp.isfinite(array) & (array >= 0)))):
                    raise ValueError(f"invalid chemistry restart {name}")
                arrays[name] = array
            steps = chem["steps"]
            if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
                raise ValueError("invalid chemistry restart step count")
            initial_mass = chem["initial_mass_kg"]
            if (not isinstance(initial_mass, list) or len(initial_mass) != len(self.chemistry.species)
                    or any(isinstance(x, bool) or not isinstance(x, (int, float))
                           or not math.isfinite(x) or x < 0 for x in initial_mass)):
                raise ValueError("invalid chemistry restart initial mass")
            super().restore_restart_state({k: v for k, v in payload.items() if k != "chemistry"})
            for name, array in arrays.items():
                getattr(self.chemistry, name)[...] = array
            self.chemistry.steps = steps
            self.chemistry.initial_mass_kg = list(initial_mass)
            self.chemistry.restart_provider_receipt = chem.get("provider")

        def write_ledger(self, output):
            import cupy as cp
            runtime = self.chemistry
            totals = {key: cp.sum(getattr(runtime, name)*runtime.area, axis=1).get().tolist()
                      for key, name in (("dry_deposition_kg", "dry"), ("settled_kg", "settled"),
                                        ("emitted_kg", "emitted"))}
            columns = prepare_mpas_to_phys_cuda_v841(
                runtime.driver.atmosphere, scalar_names=runtime.row.names(),
                geometry=self._prep_geometry, kernel_cache=self._kernel_cache)
            fields = runtime.products(runtime.driver.atmosphere, columns.rho_dry, columns.z_p)
            totals["final_mass_kg"] = [float(cp.sum(fields[s.name+"_column_mass"]*runtime.area))
                                       for s in runtime.species]
            totals["initial_mass_kg"] = runtime.initial_mass_kg
            totals["mass_closure_kg"] = [final+dry+settled-initial-emitted
                for final,dry,settled,initial,emitted in zip(totals["final_mass_kg"],
                    totals["dry_deposition_kg"],totals["settled_kg"],runtime.initial_mass_kg,
                    totals["emitted_kg"])]
            totals["surface_max_ug_m3"] = [float(cp.max(fields[s.name+"_surface_concentration"]))
                                          for s in runtime.species]
            Path(output, "chemistry-receipt.json").write_text(json.dumps(
                {**runtime.receipt(), **totals}, indent=2)+"\n")

    backend = ChemistryBackend(
        constructor=constructor, prep_geometry=prep_geometry, kernel_cache=kernel_cache,
        gwdo_static=gwdo_static, gwdo_kernel_cache=gwdo_kernel_cache,
        arwen_checkout=checkout, rollback_snapshot=rollback_snapshot)
    backend.chemistry = ChemistryTransport(
        _CONFIGS[row.name], cache=kernel_cache, area=cell_area_m2,
        nlev=constructor.n_levels, ncells=constructor.n_columns)
    return backend


def attach(backend, host, driver):
    """Bind provider inputs and the committed atmosphere."""
    runtime = getattr(backend, "chemistry", None)
    if runtime is None:
        return
    runtime.driver = driver
    if host.get("regional") is not None:
        raise ValueError("regional chemistry requires scalar LBC forcing over the specified and relaxation zones; sentinel-only input would leave the plume boundary undriven")
    if not runtime.raw.get("inputs"):
        raise ValueError("chemistry forecast requires dated initial composition and emission coverage in its inputs block")
    if runtime.raw.get("inputs"):
        from . import chem_emissions
        source_start = chem_emissions.instant(runtime.raw["inputs"]["valid_time"])
        from datetime import datetime, timezone
        weather_start = int(datetime.strptime(host["start_time_text"], "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
        if source_start != weather_start:
            raise ValueError("chemistry inputs and weather initialization valid times differ")
        grid_hash = host["prepared"].input_bytes["grid"]["sha256"]
        if runtime.raw["inputs"]["mesh_sha256"] != grid_hash:
            raise ValueError("chemistry inputs and weather grid SHA-256 differ")
        runtime.provider_receipt = chem_emissions.attach(backend, host)
    if (runtime.restart_provider_receipt is not None
            and runtime.restart_provider_receipt != runtime.provider_receipt):
        raise ValueError("chemistry restart source-file identity differs")
    import cupy as cp
    for s in runtime.species:
        if s.name in runtime.initial and runtime.steps == 0:
            values = cp.asarray(runtime.initial[s.name], dtype=cp.float32)
            target = driver.atmosphere.state.scalars[runtime.row.index_of(s.name)]
            if values.shape != target.shape or not bool(cp.all(cp.isfinite(values) & (values >= 0))):
                raise ValueError(f"chemistry IC for {s.name} has invalid shape or mass")
            target[...] = values
    if runtime.initial_mass_kg is None:
        from .cuda_physics_prep_v841 import prepare_mpas_to_phys_cuda_v841
        columns = prepare_mpas_to_phys_cuda_v841(driver.atmosphere,
            scalar_names=runtime.row.names(), geometry=backend._prep_geometry,
            kernel_cache=backend._kernel_cache)
        products = runtime.products(driver.atmosphere, columns.rho_dry, columns.z_p)
        runtime.initial_mass_kg = [float(cp.sum(products[s.name+"_column_mass"]*runtime.area))
                                   for s in runtime.species]


def history_units(name):
    if name.endswith(("_column_mass", "_dry_deposition", "_settled_mass")):
        return "kg m-2"
    if name.endswith("_surface_concentration"):
        return "ug m-3"
    if name == "aod550" or name.endswith("_aod550"):
        return "1"
    return None


def history_metadata(path, expected_sha256):
    path, raw, species = load_config(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError("chemistry configuration changed during the forecast")
    return json.dumps({"schema": SCHEMA, "config_sha256": digest, "basis": "dry_air",
                       "species": [s.__dict__ for s in species],
                       "aod550": "dry bulk mass-extinction proxy; no humidity-dependent growth",
                       "wet_scavenging": "not implemented"}, sort_keys=True)


def admit_arguments(args):
    if getattr(args, "chemistry_config", None) is None:
        return
    backend = getattr(args, "physics_backend", None)
    from .physics_default import arwen_suite
    # --physics-suite arwen_mpas_column alone is the ArWen default column (the road a bare run took before 0.4.6).
    suite = getattr(args, "physics_suite", None)
    if (backend not in (None, "wsm6_column") and not str(backend).startswith("chemistry_")) or (suite and suite != arwen_suite()) or getattr(args, "physics", None):
        raise ValueError("chemistry currently couples to the active default WSM6/YSU column backend; another physics selection needs its own coefficient and scalar-update contract and cannot be silently replaced")
