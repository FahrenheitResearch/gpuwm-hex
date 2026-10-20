FC = gfortran
FFLAGS = -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -fconvert=big-endian -ffree-form -cpp -fPIC --coverage
all: libsmoke_oracle.so

framework.o: framework.f90
	$(FC) $(FFLAGS) -c $<

mpas_smoke_config.o: mpas_smoke_config.F90 framework.o
	$(FC) $(FFLAGS) -c $<

mpas_smoke_init.o: mpas_smoke_init.F90 framework.o mpas_smoke_config.o
	$(FC) $(FFLAGS) -c $<

dep_data_mod.o: dep_data_mod.F90 framework.o mpas_smoke_config.o mpas_smoke_init.o
	$(FC) $(FFLAGS) -c $<

rad_data_mod.o: rad_data_mod.F90 framework.o mpas_smoke_config.o mpas_smoke_init.o dep_data_mod.o
	$(FC) $(FFLAGS) -c $<

dust_data_mod.o: dust_data_mod.F90 framework.o mpas_smoke_config.o mpas_smoke_init.o dep_data_mod.o rad_data_mod.o
	$(FC) $(FFLAGS) -c $<

seas_data_mod.o: seas_data_mod.F90 framework.o mpas_smoke_config.o mpas_smoke_init.o dep_data_mod.o rad_data_mod.o dust_data_mod.o
	$(FC) $(FFLAGS) -c $<

seas_ngac_mod.o: seas_ngac_mod.F90 framework.o mpas_smoke_config.o mpas_smoke_init.o dep_data_mod.o rad_data_mod.o dust_data_mod.o seas_data_mod.o
	$(FC) $(FFLAGS) -c $<

dep_dry_simple_mod.o: dep_dry_simple_mod.F90 framework.o mpas_smoke_config.o mpas_smoke_init.o dep_data_mod.o rad_data_mod.o dust_data_mod.o seas_data_mod.o seas_ngac_mod.o
	$(FC) $(FFLAGS) -c $<

dep_dry_mod_emerson.o: dep_dry_mod_emerson.F90 framework.o mpas_smoke_config.o mpas_smoke_init.o dep_data_mod.o rad_data_mod.o dust_data_mod.o seas_data_mod.o seas_ngac_mod.o dep_dry_simple_mod.o
	$(FC) $(FFLAGS) -c $<

module_zero_plumegen_coms.o: module_zero_plumegen_coms.F90 framework.o mpas_smoke_config.o mpas_smoke_init.o dep_data_mod.o rad_data_mod.o dust_data_mod.o seas_data_mod.o seas_ngac_mod.o dep_dry_simple_mod.o dep_dry_mod_emerson.o
	$(FC) $(FFLAGS) -c $<

module_smoke_plumerise.o: module_smoke_plumerise.F90 framework.o mpas_smoke_config.o mpas_smoke_init.o dep_data_mod.o rad_data_mod.o dust_data_mod.o seas_data_mod.o seas_ngac_mod.o dep_dry_simple_mod.o dep_dry_mod_emerson.o module_zero_plumegen_coms.o
	$(FC) $(FFLAGS) -c $<

module_plumerise.o: module_plumerise.F90 framework.o mpas_smoke_config.o mpas_smoke_init.o dep_data_mod.o rad_data_mod.o dust_data_mod.o seas_data_mod.o seas_ngac_mod.o dep_dry_simple_mod.o dep_dry_mod_emerson.o module_zero_plumegen_coms.o module_smoke_plumerise.o
	$(FC) $(FFLAGS) -c $<

module_wetdep_ls.o: module_wetdep_ls.F90 framework.o mpas_smoke_config.o mpas_smoke_init.o dep_data_mod.o rad_data_mod.o dust_data_mod.o seas_data_mod.o seas_ngac_mod.o dep_dry_simple_mod.o dep_dry_mod_emerson.o module_zero_plumegen_coms.o module_smoke_plumerise.o module_plumerise.o
	$(FC) $(FFLAGS) -c $<

module_add_emiss_burn.o: module_add_emiss_burn.F90 framework.o mpas_smoke_config.o mpas_smoke_init.o dep_data_mod.o rad_data_mod.o dust_data_mod.o seas_data_mod.o seas_ngac_mod.o dep_dry_simple_mod.o dep_dry_mod_emerson.o module_zero_plumegen_coms.o module_smoke_plumerise.o module_plumerise.o module_wetdep_ls.o
	$(FC) $(FFLAGS) -c $<

module_anthro_emissions.o: module_anthro_emissions.F90 framework.o mpas_smoke_config.o mpas_smoke_init.o dep_data_mod.o rad_data_mod.o dust_data_mod.o seas_data_mod.o seas_ngac_mod.o dep_dry_simple_mod.o dep_dry_mod_emerson.o module_zero_plumegen_coms.o module_smoke_plumerise.o module_plumerise.o module_wetdep_ls.o module_add_emiss_burn.o
	$(FC) $(FFLAGS) -c $<

module_fire_emissions.o: module_fire_emissions.F90 framework.o mpas_smoke_config.o mpas_smoke_init.o dep_data_mod.o rad_data_mod.o dust_data_mod.o seas_data_mod.o seas_ngac_mod.o dep_dry_simple_mod.o dep_dry_mod_emerson.o module_zero_plumegen_coms.o module_smoke_plumerise.o module_plumerise.o module_wetdep_ls.o module_add_emiss_burn.o module_anthro_emissions.o
	$(FC) $(FFLAGS) -c $<

dust_fengsha_mod.o: dust_fengsha_mod.F90 framework.o mpas_smoke_config.o mpas_smoke_init.o dep_data_mod.o rad_data_mod.o dust_data_mod.o seas_data_mod.o seas_ngac_mod.o dep_dry_simple_mod.o dep_dry_mod_emerson.o module_zero_plumegen_coms.o module_smoke_plumerise.o module_plumerise.o module_wetdep_ls.o module_add_emiss_burn.o module_anthro_emissions.o module_fire_emissions.o
	$(FC) $(FFLAGS) -c $<

ssalt_mod.o: ssalt_mod.F90 framework.o mpas_smoke_config.o mpas_smoke_init.o dep_data_mod.o rad_data_mod.o dust_data_mod.o seas_data_mod.o seas_ngac_mod.o dep_dry_simple_mod.o dep_dry_mod_emerson.o module_zero_plumegen_coms.o module_smoke_plumerise.o module_plumerise.o module_wetdep_ls.o module_add_emiss_burn.o module_anthro_emissions.o module_fire_emissions.o dust_fengsha_mod.o
	$(FC) $(FFLAGS) -c $<

module_smoke_diagnostics.o: module_smoke_diagnostics.F90 framework.o mpas_smoke_config.o mpas_smoke_init.o dep_data_mod.o rad_data_mod.o dust_data_mod.o seas_data_mod.o seas_ngac_mod.o dep_dry_simple_mod.o dep_dry_mod_emerson.o module_zero_plumegen_coms.o module_smoke_plumerise.o module_plumerise.o module_wetdep_ls.o module_add_emiss_burn.o module_anthro_emissions.o module_fire_emissions.o dust_fengsha_mod.o ssalt_mod.o
	$(FC) $(FFLAGS) -c $<

mpas_smoke_wrapper.o: mpas_smoke_wrapper.F90 framework.o mpas_smoke_config.o mpas_smoke_init.o dep_data_mod.o rad_data_mod.o dust_data_mod.o seas_data_mod.o seas_ngac_mod.o dep_dry_simple_mod.o dep_dry_mod_emerson.o module_zero_plumegen_coms.o module_smoke_plumerise.o module_plumerise.o module_wetdep_ls.o module_add_emiss_burn.o module_anthro_emissions.o module_fire_emissions.o dust_fengsha_mod.o ssalt_mod.o module_smoke_diagnostics.o
	$(FC) $(FFLAGS) -c $<

libsmoke_oracle.so: framework.o mpas_smoke_config.o mpas_smoke_init.o dep_data_mod.o rad_data_mod.o dust_data_mod.o seas_data_mod.o seas_ngac_mod.o dep_dry_simple_mod.o dep_dry_mod_emerson.o module_zero_plumegen_coms.o module_smoke_plumerise.o module_plumerise.o module_wetdep_ls.o module_add_emiss_burn.o module_anthro_emissions.o module_fire_emissions.o dust_fengsha_mod.o ssalt_mod.o module_smoke_diagnostics.o mpas_smoke_wrapper.o
	$(FC) $(FFLAGS) -shared -o $@ $^

constants: constants.f90 libsmoke_oracle.so
	$(FC) $(FFLAGS) -o $@ $< -L. -lsmoke_oracle -Wl,-rpath,'$$ORIGIN'

driver_slice.o: driver_slice.F90 framework.o
	$(FC) $(FFLAGS) -c $<

libdriver_slice.so: driver_slice.o framework.o driver_bridge.o
	$(FC) $(FFLAGS) -shared -o $@ $^

driver_bridge.o: driver_bridge.f90 driver_slice.o
	$(FC) $(FFLAGS) -c $<
