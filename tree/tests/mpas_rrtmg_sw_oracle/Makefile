FC=gfortran
FFLAGS=-O3 -cpp -DSINGLE_PRECISION -DWRF_CHEM=0 -ffree-form -ffree-line-length-none -fconvert=big-endian -fallow-argument-mismatch -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations --coverage
MODULES=mpas_kind_types.o mpas_constants.o framework_stubs.o mpas_atmphys_constants.o mpas_atmphys_functions.o module_ra_rrtmg_vinterp.o module_ra_rrtmg_lw.o module_ra_rrtmg_sw.o module_ra_rrtmg_sw_aerosols.o mpas_atmphys_rrtmg_swinit.o
all: sw_dump_tables sw_fixture_driver oracle_aer3_mpas oracle_aerosol_options oracle_aerosol_fatal oracle_solar oracle_cloud oracle_fatal sw_driver_fixture oracle_mcica oracle_mcica_single oracle_kiss_fatal oracle_ozone_batch oracle_composition oracle_reftra oracle_reftra_subnormal
%.o: %.F
	$(FC) $(FFLAGS) -c $< -o $@
%.o: %.F90
	$(FC) $(FFLAGS) -c $< -o $@
mpas_constants.o: mpas_kind_types.o
mpas_atmphys_constants.o: mpas_constants.o
mpas_atmphys_functions.o: mpas_kind_types.o framework_stubs.o
module_ra_rrtmg_vinterp.o: mpas_kind_types.o
module_ra_rrtmg_lw.o: mpas_atmphys_constants.o module_ra_rrtmg_vinterp.o framework_stubs.o
module_ra_rrtmg_sw.o: module_ra_rrtmg_lw.o
module_ra_rrtmg_sw_aerosols.o: mpas_atmphys_functions.o
mpas_atmphys_rrtmg_swinit.o: module_ra_rrtmg_sw.o

module_mp_thompson_aerosols.o: mpas_atmphys_functions.o
sw_dump_tables.o: $(MODULES) module_mp_thompson_aerosols.o
sw_dump_tables: $(MODULES) module_mp_thompson_aerosols.o sw_dump_tables.o
	$(FC) $(FFLAGS) $^ -o $@
sw_fixture_driver.o: $(MODULES) module_mp_thompson_aerosols.o
sw_fixture_driver: $(MODULES) module_mp_thompson_aerosols.o sw_fixture_driver.o
	$(FC) $(FFLAGS) $^ -o $@
oracle_aer3_mpas.o: $(MODULES) module_mp_thompson_aerosols.o
oracle_aer3_mpas: $(MODULES) module_mp_thompson_aerosols.o oracle_aer3_mpas.o
	$(FC) $(FFLAGS) $^ -o $@
oracle_aerosol_options.o: $(MODULES) module_mp_thompson_aerosols.o
oracle_aerosol_options: $(MODULES) module_mp_thompson_aerosols.o oracle_aerosol_options.o
	$(FC) $(FFLAGS) $^ -o $@

oracle_aerosol_fatal.o: $(MODULES)
oracle_aerosol_fatal: $(MODULES) oracle_aerosol_fatal.o
	$(FC) $(FFLAGS) $^ -o $@

oracle_solar.o: mpas_atmphys_constants.o
oracle_solar: mpas_kind_types.o mpas_constants.o mpas_atmphys_constants.o oracle_solar.o
	$(FC) $(FFLAGS) $^ -o $@

oracle_cloud.o: $(MODULES)
oracle_cloud: $(MODULES) oracle_cloud.o
	$(FC) $(FFLAGS) $^ -o $@

oracle_fatal.o: $(MODULES)
oracle_fatal: $(MODULES) oracle_fatal.o
	$(FC) $(FFLAGS) $^ -o $@

sw_driver_fixture.o: $(MODULES) module_mp_thompson_aerosols.o
sw_driver_fixture: $(MODULES) module_mp_thompson_aerosols.o sw_driver_fixture.o
	$(FC) $(FFLAGS) $^ -o $@

oracle_mcica.o: $(MODULES)
oracle_mcica: $(MODULES) oracle_mcica.o
	$(FC) $(FFLAGS) $^ -o $@

oracle_composition.o: $(MODULES)
oracle_composition: $(MODULES) oracle_composition.o
	$(FC) $(FFLAGS) $^ -o $@

oracle_reftra.o: $(MODULES)
oracle_reftra: $(MODULES) oracle_reftra.o
	$(FC) $(FFLAGS) $^ -o $@

oracle_reftra_subnormal.o: $(MODULES)
oracle_reftra_subnormal: $(MODULES) oracle_reftra_subnormal.o
	$(FC) $(FFLAGS) $^ -o $@

# Native defect diagnosis only; the strict reference objects remain intact.
module_ra_rrtmg_sw_aerosols_snan.o: module_ra_rrtmg_sw_aerosols.F mpas_atmphys_functions.o
	$(FC) $(FFLAGS) -finit-real=snan -c $< -o $@
oracle_aerosol_snan: $(filter-out module_ra_rrtmg_sw_aerosols.o,$(MODULES)) module_ra_rrtmg_sw_aerosols_snan.o module_mp_thompson_aerosols.o oracle_aerosol_options.o
	$(FC) $(FFLAGS) $^ -o $@

# Rust owns typed real-capture decoding, encoding and exact comparison.
oracle_words: oracle_words.rs
	rustc --edition 2021 -C opt-level=2 $< -o $@

# MT uses no four-layer KISS seed requirement; exercise one-layer overlap.
oracle_mcica_single.o: oracle_mcica.F90 $(MODULES)
	$(FC) $(FFLAGS) -DSINGLE_LAYER_ORACLE -c $< -o $@
oracle_mcica_single: $(MODULES) oracle_mcica_single.o
	$(FC) $(FFLAGS) $^ -o $@

# Mixed-column interpolation/extrapolation invokes the pristine dependency.
oracle_ozone_batch.o: mpas_kind_types.o module_ra_rrtmg_vinterp.o
oracle_ozone_batch: mpas_kind_types.o module_ra_rrtmg_vinterp.o oracle_ozone_batch.o
	$(FC) $(FFLAGS) $^ -o $@

# Defined native KISS fatal pressure order, independent of coefficient tables.
oracle_kiss_fatal.o: $(MODULES)
oracle_kiss_fatal: $(MODULES) oracle_kiss_fatal.o
	$(FC) $(FFLAGS) $^ -o $@

# Source-native unchecked seed/default-selector diagnosis; no forecast exclusion.
oracle_mcica_boundary.o: $(MODULES)
oracle_mcica_boundary: $(MODULES) oracle_mcica_boundary.o
	$(FC) $(FFLAGS) $^ -o $@

# Exact native driver option slice, including initialized default idcor.
oracle_driver_controls: oracle_driver_controls.o
	$(FC) $(FFLAGS) $^ -o $@

# Native unchecked empty/one-level ozone diagnosis; never an exclusion.
oracle_ozone_boundary.o: mpas_kind_types.o module_ra_rrtmg_vinterp.o
oracle_ozone_boundary: mpas_kind_types.o module_ra_rrtmg_vinterp.o oracle_ozone_boundary.o
	$(FC) $(FFLAGS) $^ -o $@
module_ra_rrtmg_vinterp_bounds.o: module_ra_rrtmg_vinterp.F mpas_kind_types.o
	$(FC) $(FFLAGS) -fcheck=bounds -c $< -o $@
oracle_ozone_boundary_bounds: mpas_kind_types.o module_ra_rrtmg_vinterp_bounds.o oracle_ozone_boundary.o
	$(FC) $(FFLAGS) $^ -o $@
