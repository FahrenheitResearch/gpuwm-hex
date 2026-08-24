"""Fail-closed MPAS-A v8.4.1 dry-dycore configuration.

The v8.2.3 configuration remains the default lane.  This additive subtype is
the only object that may select the v8.4.1 CPU implementation, which prevents
an old configuration from being relabelled by changing receipt text.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .driver import DryDycoreConfig
from .errors import ConfigurationRefusal

V841_SOURCE_RELEASE = "v8.4.1"
V823_SOURCE_RELEASE = "v8.2.3"
V841_GWDO_CONTRACT_SHA256 = (
    "514241e42f6154c6e00bf53a2cd050a12079d0315a7de4d8d35189c213f5ba83"
)
V841_GWDO_KERNEL_SHA256 = (
    "b334506b289c6f002a2660e6c7796361e39372f6652b8e60268c1400900ab9ec"
)
V841_GWDO_EXTERNAL_COMPOSER = "PersistentTwoPhaseCudaPhysicsBackendV841"
V841_GWDO_COMPOSITION_PHASE = "before_held_tendency_coupling"
V841_GWDO_RESULT_MODULE = "mpas_port.cuda_gwdo_v841"
V841_GWDO_RESULT_CLASS = "CudaYsuGwdoResultV841"

V841_ARWEN_BUILD_COMMIT = "fa35bfbdeb4a33a1121852fad930bb3841302856"
V841_ARWEN_AGGREGATE_FACTORY = "gpuwm.core.mpas_column_batch.run_mpas_column_batch"
V841_ARWEN_PHASE1_ORCHESTRATION = "gpuwm.core.physics.PhysicsDriver.compute"
V841_ARWEN_SOURCE_MANIFEST = (
    (
        "gpuwm/core/mpas_column_batch.py",
        "4da7cfbe3e12e201a2c3ab50909355f43ef2704dc839601ba889fe547f8df941",
    ),
    (
        "gpuwm/core/physics.py",
        "e4dc59d4d38dab927e18af85bc3fb5163216f7fd659671fa8a0e8f6a9b4fc8a8",
    ),
    (
        "gpuwm/core/microphysics.py",
        "571e0c2724fe5d2126d34d77b742b2c249c3ac91132800148c50382365de069f",
    ),
    (
        "gpuwm/core/gf.py",
        "4e4e200c6b8cf10e09c9d7ef72820b4356c298456b0f98686ad0f617dd6e98fd",
    ),
    (
        "gpuwm/core/kernels/gf.cu",
        "bc8b301453a413f71b5e9a117cba2a9cab8dcf9f566bc91c5a03ff2a60b47514",
    ),
    (
        "gpuwm/core/rrtmg_legacy.py",
        "c2acafc1e3a089d04a9b271cc5ffdd5473b6af3085a22b05742de9a8063cf0ae",
    ),
)
V841_ARWEN_H_DIABATIC_APPLIED = False


def _refuse(knob: str, value: object, reason: str, declaration: str) -> None:
    raise ConfigurationRefusal(knob, value, reason, declaration)


@dataclass(frozen=True, slots=True)
class V841DryDycoreConfig(DryDycoreConfig):
    """The first executable closed, dry, no-active-mixing v8.4.1 lane."""

    source_release: str = V841_SOURCE_RELEASE

    # v8.4.1 retains this legacy Registry sentinel but no longer consumes it.
    config_epssm: float = 0.0
    config_epssm_minimum: float = 0.1
    config_epssm_maximum: float = 0.5
    config_epssm_transition_bottom_z: float = 30_000.0
    config_epssm_transition_top_z: float = 50_000.0

    # native Registry.xml:264 config_smdiv default 0.1, and
    # mpas_atm_time_integration.F:2401 calls atm_divergence_damping_3d
    # unconditionally inside the acoustic sub-step loop.  The port's boolean has
    # no native counterpart and is descriptive only; config_smdiv is the knob.
    config_divergence_damping: bool = True
    config_smdiv: float = 0.1

    config_gpu_aware_mpi: bool = False
    config_les_model: str = "none"
    config_les_surface: str = "none"
    config_mix_scalars: bool = False
    config_surface_heat_flux: float = 0.0
    config_surface_moisture_flux: float = 0.0
    config_surface_drag_coefficient: float = 0.0

    def validate(self) -> None:
        # The v8.4.1 deformation-coefficient binary32 path IS ported
        # (mpas_port.mixing_v841 CPU authority + smagorinsky_v841_f32 CUDA
        # kernel).  Its dependent coefficients validate through the v8.4.1
        # mixing authority; the inherited v8.2.3 validator then checks the
        # remaining knobs on a neutralized no-mixing view, because its
        # split-three Smagorinsky refusal names the v8.2.3 compiled-MPAS
        # oracle, which is not this lane's authority.
        if self.config_horiz_mixing == "2d_smagorinsky":
            from .mixing_v841 import V841MixingConfig

            V841MixingConfig(
                config_horiz_mixing=self.config_horiz_mixing,
                config_len_disp=self.config_len_disp,
                config_visc4_2dsmag=self.config_visc4_2dsmag,
                config_smagorinsky_coef=self.config_smagorinsky_coef,
                config_del4u_div_factor=self.config_del4u_div_factor,
                config_h_ScaleWithMesh=self.config_h_ScaleWithMesh,
                config_mpas_cam_coef=self.config_mpas_cam_coef,
            ).validate()
            neutral_values = {
                field.name: getattr(self, field.name)
                for field in fields(type(self))
                if field.name in {f.name for f in fields(V841DryDycoreConfig)}
            }
            neutral_values["config_horiz_mixing"] = "off"
            neutral_values["config_len_disp"] = 0.0
            neutral_values["config_visc4_2dsmag"] = 0.0
            neutral_values["config_smagorinsky_coef"] = 0.0
            neutral = V841DryDycoreConfig(**neutral_values)
            DryDycoreConfig.validate(neutral)
        else:
            # ``slots=True`` creates a replacement class object, so
            # zero-argument super() is not reliable in methods retained from
            # the pre-replacement class namespace (CPython dataclasses issue
            # 90562).
            DryDycoreConfig.validate(self)
        if self.source_release != V841_SOURCE_RELEASE:
            _refuse(
                "source_release",
                self.source_release,
                "this configuration type is pinned to the v8.4.1 source semantics",
                f"source_release={V841_SOURCE_RELEASE!r}",
            )
        if not np.isfinite(self.config_epssm) or self.config_epssm != 0.0:
            _refuse(
                "config_epssm",
                self.config_epssm,
                "v8.4.1 no longer consumes the legacy scalar off-centering knob",
                "config_epssm=0.0 and the four &damping profile knobs",
            )

        minimum = float(self.config_epssm_minimum)
        maximum = float(self.config_epssm_maximum)
        bottom = float(self.config_epssm_transition_bottom_z)
        top = float(self.config_epssm_transition_top_z)
        if not np.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
            _refuse(
                "config_epssm_minimum",
                self.config_epssm_minimum,
                "the v8.4.1 off-centering profile must be bounded",
                "0<=config_epssm_minimum<=1",
            )
        if not np.isfinite(maximum) or not minimum <= maximum <= 1.0:
            _refuse(
                "config_epssm_maximum",
                self.config_epssm_maximum,
                "the v8.4.1 off-centering profile must be ordered and bounded",
                "config_epssm_minimum<=config_epssm_maximum<=1",
            )
        if not np.isfinite(bottom) or bottom < 0.0:
            _refuse(
                "config_epssm_transition_bottom_z",
                self.config_epssm_transition_bottom_z,
                "the computational-height transition must begin at or above zero",
                "config_epssm_transition_bottom_z>=0",
            )
        if not np.isfinite(top) or top <= bottom:
            _refuse(
                "config_epssm_transition_top_z",
                self.config_epssm_transition_top_z,
                "the computational-height transition must have positive width",
                "config_epssm_transition_top_z>config_epssm_transition_bottom_z",
            )

        if not isinstance(self.config_gpu_aware_mpi, (bool, np.bool_)):
            _refuse(
                "config_gpu_aware_mpi",
                self.config_gpu_aware_mpi,
                "the Registry option is logical",
                "config_gpu_aware_mpi=false",
            )
        if self.config_gpu_aware_mpi:
            _refuse(
                "config_gpu_aware_mpi",
                self.config_gpu_aware_mpi,
                "the closed serial CPU lane has no MPI halo transport",
                "config_gpu_aware_mpi=false",
            )
        if self.config_les_model != "none":
            _refuse(
                "config_les_model",
                self.config_les_model,
                "the v8.4.1 LES dissipation branches are not ported",
                "config_les_model='none'",
            )
        if self.config_les_surface != "none":
            _refuse(
                "config_les_surface",
                self.config_les_surface,
                "surface LES fluxes require a ported LES model",
                "config_les_surface='none'",
            )
        if not isinstance(self.config_mix_scalars, (bool, np.bool_)):
            _refuse(
                "config_mix_scalars",
                self.config_mix_scalars,
                "the Registry option is logical",
                "config_mix_scalars=false",
            )
        if self.config_mix_scalars:
            _refuse(
                "config_mix_scalars",
                self.config_mix_scalars,
                "the new dynamics-substep scalar dissipation branch is not ported",
                "config_mix_scalars=false",
            )
        for knob in (
            "config_surface_heat_flux",
            "config_surface_moisture_flux",
            "config_surface_drag_coefficient",
        ):
            value = float(getattr(self, knob))
            if not np.isfinite(value) or value != 0.0:
                _refuse(
                    knob,
                    getattr(self, knob),
                    "this LES-surface-only knob must not be silently ignored",
                    f"{knob}=0.0 with config_les_surface='none'",
                )

        # A zero-coefficient 2d_fixed selection is a no-mixing branch.  The
        # active v8.4.1 Smagorinsky association/order remains a separate lane.


@dataclass(frozen=True, slots=True)
class V841MpasColumnPhysicsConfig(V841DryDycoreConfig):
    """Exact real-x4 v8.4.1 CUDA dynamics plus aggregate column physics.

    This is deliberately a distinct type rather than a relaxed dry config.
    Its selector and cadence fields form the boundary contract with the
    persistent two-phase Arwen column backend; changing any one of them is a
    different, currently unproved lane.
    """

    config_dt: float = 120.0
    config_time_integration_order: int = 3
    config_number_of_sub_steps: int = 6
    config_dynamics_split_steps: int = 3
    config_split_dynamics_transport: bool = True
    config_scalar_advection: bool = True
    config_monotonic: bool = False
    config_positive_definite: bool = False
    config_apvm_upwinding: float = 0.5
    config_xnutr: float = 0.2
    config_zd: float = 22_000.0
    config_terrain_following: bool | None = True

    config_moist_physics: bool = True
    config_physics_suite: str = "arwen_mpas_column"
    config_microp_scheme: str = "mp_wsm6"
    config_convection_scheme: str = "cu_grell_freitas"
    config_lsm_scheme: str = "sf_noahmp"
    config_pbl_scheme: str = "bl_ysu"
    config_sfclayer_scheme: str = "sf_monin_obukhov_rev"
    config_radt_cld_scheme: str = "cld_fraction"
    config_radt_lw_scheme: str = "rrtmg_lw"
    config_radt_sw_scheme: str = "rrtmg_sw"
    config_gwdo_scheme: str = "off"

    config_radt_seconds: float = 600.0
    config_bldt_seconds: float = 120.0
    config_cudt_seconds: float = 120.0

    phase1_arwen_commit: str = V841_ARWEN_BUILD_COMMIT
    phase1_aggregate_factory: str = V841_ARWEN_AGGREGATE_FACTORY
    phase1_orchestration: str = V841_ARWEN_PHASE1_ORCHESTRATION
    phase1_source_manifest: tuple[tuple[str, str], ...] = V841_ARWEN_SOURCE_MANIFEST
    phase1_h_diabatic_applied: bool = V841_ARWEN_H_DIABATIC_APPLIED

    def validate(self) -> None:
        # Reuse every numerical and registry check from the pinned dry lane
        # without weakening that lane's fail-closed physics refusal.
        dry_values = {
            field.name: getattr(self, field.name)
            for field in fields(V841DryDycoreConfig)
        }
        dry_values["config_moist_physics"] = False
        dry_values["config_physics_suite"] = "none"
        V841DryDycoreConfig(**dry_values).validate()

        exact = {
            "config_horiz_mixing": "off",
            "config_coef_3rd_order": 0.25,
            "config_epssm_minimum": 0.1,
            "config_epssm_maximum": 0.5,
            "config_epssm_transition_bottom_z": 30_000.0,
            "config_epssm_transition_top_z": 50_000.0,
            "config_dt": 120.0,
            "config_time_integration_order": 3,
            "config_number_of_sub_steps": 6,
            "config_dynamics_split_steps": 3,
            "config_split_dynamics_transport": True,
            "config_scalar_advection": True,
            "config_monotonic": False,
            "config_positive_definite": False,
            "config_apvm_upwinding": 0.5,
            "config_xnutr": 0.2,
            "config_zd": 22_000.0,
            "config_smdiv": 0.1,
            "config_divergence_damping": True,
            "config_terrain_following": True,
            "config_moist_physics": True,
            "config_physics_suite": "arwen_mpas_column",
            "config_microp_scheme": "mp_wsm6",
            "config_convection_scheme": "cu_grell_freitas",
            "config_lsm_scheme": "sf_noahmp",
            "config_pbl_scheme": "bl_ysu",
            "config_sfclayer_scheme": "sf_monin_obukhov_rev",
            "config_radt_cld_scheme": "cld_fraction",
            "config_radt_lw_scheme": "rrtmg_lw",
            "config_radt_sw_scheme": "rrtmg_sw",
            "config_gwdo_scheme": "off",
            "config_radt_seconds": 600.0,
            "config_bldt_seconds": 120.0,
            "config_cudt_seconds": 120.0,
            "phase1_arwen_commit": V841_ARWEN_BUILD_COMMIT,
            "phase1_aggregate_factory": V841_ARWEN_AGGREGATE_FACTORY,
            "phase1_orchestration": V841_ARWEN_PHASE1_ORCHESTRATION,
            "phase1_source_manifest": V841_ARWEN_SOURCE_MANIFEST,
            "phase1_h_diabatic_applied": V841_ARWEN_H_DIABATIC_APPLIED,
        }
        for knob, required in exact.items():
            actual = getattr(self, knob)
            if isinstance(required, float):
                try:
                    matches = np.isfinite(float(actual)) and float(actual) == required
                except (TypeError, ValueError):
                    matches = False
            elif isinstance(required, bool):
                matches = isinstance(actual, (bool, np.bool_)) and bool(actual) is required
            elif isinstance(required, int):
                matches = (
                    isinstance(actual, (int, np.integer))
                    and not isinstance(actual, (bool, np.bool_))
                    and int(actual) == required
                )
            else:
                matches = actual == required
            if not matches:
                _refuse(
                    knob,
                    actual,
                    "the matched v8.4.1 real-x4 column-physics lane is exact",
                    f"{knob}={required!r}",
                )


@dataclass(frozen=True, slots=True)
class V841MpasColumnPhysicsGwdoConfig(V841MpasColumnPhysicsConfig):
    """Exact real-x4 lane with released external YSU gravity-wave drag."""

    config_gwdo_scheme: str = "bl_ysu_gwdo"
    gwdo_external_composer: str = V841_GWDO_EXTERNAL_COMPOSER
    gwdo_composition_phase: str = V841_GWDO_COMPOSITION_PHASE
    gwdo_contract_sha256: str = V841_GWDO_CONTRACT_SHA256
    gwdo_kernel_sha256: str = V841_GWDO_KERNEL_SHA256
    gwdo_result_module: str = V841_GWDO_RESULT_MODULE
    gwdo_result_class: str = V841_GWDO_RESULT_CLASS
    gwdo_arwen_execution_claim: bool = False

    def validate(self) -> None:
        # Validate every exact released off-lane value through its own concrete
        # type, changing only the selector that distinguishes this subtype.
        parent_values = {
            field.name: getattr(self, field.name)
            for field in fields(V841MpasColumnPhysicsConfig)
        }
        parent_values["config_gwdo_scheme"] = "off"
        V841MpasColumnPhysicsConfig(**parent_values).validate()

        exact = {
            "config_gwdo_scheme": "bl_ysu_gwdo",
            "gwdo_external_composer": V841_GWDO_EXTERNAL_COMPOSER,
            "gwdo_composition_phase": V841_GWDO_COMPOSITION_PHASE,
            "gwdo_contract_sha256": V841_GWDO_CONTRACT_SHA256,
            "gwdo_kernel_sha256": V841_GWDO_KERNEL_SHA256,
            "gwdo_result_module": V841_GWDO_RESULT_MODULE,
            "gwdo_result_class": V841_GWDO_RESULT_CLASS,
            "gwdo_arwen_execution_claim": False,
        }
        for knob, required in exact.items():
            actual = getattr(self, knob)
            if isinstance(required, bool):
                matches = isinstance(actual, (bool, np.bool_)) and bool(actual) is required
            else:
                matches = actual == required
            if not matches:
                _refuse(
                    knob,
                    actual,
                    "the released v8.4.1 external YSU-GWDO lane is exact",
                    f"{knob}={required!r}",
                )


@dataclass(frozen=True, slots=True)
class V841MpasColumnPhysicsSmagorinskyGwdoConfig(V841MpasColumnPhysicsGwdoConfig):
    """Released v8.4.1 GWDO lane plus native 2-D Smagorinsky mixing.

    The mixing block is exactly the native Registry configuration the
    natB-24h reference integrated (namelist verbatim: 2d_smagorinsky,
    len_disp 0.0 -> nominalMinDc, visc4_2dsmag 0.05, smagorinsky_coef 0.125,
    del4u_div_factor 10.0, h_ScaleWithMesh true).  Runs under this type start
    a new sub-series: mixing changes the answer, so they are not
    bit-comparable to any mixing-off arm.
    """

    config_horiz_mixing: str = "2d_smagorinsky"
    config_len_disp: float = 0.0
    config_visc4_2dsmag: float = 0.05
    config_smagorinsky_coef: float = 0.125
    config_del4u_div_factor: float = 10.0
    config_h_ScaleWithMesh: bool = True

    def validate(self) -> None:
        # Validate every exact released GWDO-lane value through its own
        # concrete type, changing only the mixing block back to that lane's
        # off values.
        parent_values = {
            field.name: getattr(self, field.name)
            for field in fields(V841MpasColumnPhysicsGwdoConfig)
        }
        parent_values["config_horiz_mixing"] = "off"
        parent_values["config_len_disp"] = 0.0
        parent_values["config_visc4_2dsmag"] = 0.0
        parent_values["config_smagorinsky_coef"] = 0.0
        parent_values["config_del4u_div_factor"] = 10.0
        parent_values["config_h_ScaleWithMesh"] = True
        V841MpasColumnPhysicsGwdoConfig(**parent_values).validate()

        exact = {
            "config_horiz_mixing": "2d_smagorinsky",
            "config_len_disp": 0.0,
            "config_visc4_2dsmag": 0.05,
            "config_smagorinsky_coef": 0.125,
            "config_del4u_div_factor": 10.0,
            "config_h_ScaleWithMesh": True,
            "config_mpas_cam_coef": 0.0,
        }
        for knob, required in exact.items():
            actual = getattr(self, knob)
            if isinstance(required, bool):
                matches = (
                    isinstance(actual, (bool, np.bool_))
                    and bool(actual) is required
                )
            elif isinstance(required, float):
                try:
                    matches = (
                        np.isfinite(float(actual)) and float(actual) == required
                    )
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = actual == required
            if not matches:
                _refuse(
                    knob,
                    actual,
                    "the v8.4.1 native-Registry Smagorinsky lane is exact",
                    f"{knob}={required!r}",
                )
        # The dependent-coefficient authority runs once more on the actual
        # values (defense in depth; the neutralized parent view above cannot
        # have seen them).
        from .mixing_v841 import V841MixingConfig

        V841MixingConfig(
            config_horiz_mixing=self.config_horiz_mixing,
            config_len_disp=self.config_len_disp,
            config_visc4_2dsmag=self.config_visc4_2dsmag,
            config_smagorinsky_coef=self.config_smagorinsky_coef,
            config_del4u_div_factor=self.config_del4u_div_factor,
            config_h_ScaleWithMesh=self.config_h_ScaleWithMesh,
            config_mpas_cam_coef=self.config_mpas_cam_coef,
        ).validate()


__all__ = [
    "V823_SOURCE_RELEASE",
    "V841_ARWEN_AGGREGATE_FACTORY",
    "V841_ARWEN_BUILD_COMMIT",
    "V841_ARWEN_H_DIABATIC_APPLIED",
    "V841_ARWEN_PHASE1_ORCHESTRATION",
    "V841_ARWEN_SOURCE_MANIFEST",
    "V841_GWDO_COMPOSITION_PHASE",
    "V841_GWDO_CONTRACT_SHA256",
    "V841_GWDO_EXTERNAL_COMPOSER",
    "V841_GWDO_KERNEL_SHA256",
    "V841_GWDO_RESULT_CLASS",
    "V841_GWDO_RESULT_MODULE",
    "V841_SOURCE_RELEASE",
    "V841DryDycoreConfig",
    "V841MpasColumnPhysicsConfig",
    "V841MpasColumnPhysicsGwdoConfig",
    "V841MpasColumnPhysicsSmagorinskyGwdoConfig",
]
