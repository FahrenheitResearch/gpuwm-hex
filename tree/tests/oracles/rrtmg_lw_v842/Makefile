FC=gfortran
FFLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -cpp -ffree-form -ffree-line-length-none -fconvert=big-endian -fallow-argument-mismatch -DSINGLE_PRECISION -Dmpas -DWRF_CHEM=0 --coverage
OBJS=mpas_kind_types.o mpas_constants.o stubs.o module_ra_rrtmg_vinterp.o module_ra_rrtmg_lw.o mpas_atmphys_rrtmg_lwinit.o
all: probe
%.o: src/%.F
	$(FC) $(FFLAGS) -c $< -o $@
%.o: src/%.F90
	$(FC) $(FFLAGS) -c $< -o $@
mpas_constants.o: mpas_kind_types.o
stubs.o: mpas_constants.o
module_ra_rrtmg_vinterp.o: mpas_kind_types.o
module_ra_rrtmg_lw.o: stubs.o module_ra_rrtmg_vinterp.o
mpas_atmphys_rrtmg_lwinit.o: module_ra_rrtmg_lw.o stubs.o
probe.o: $(OBJS)
probe: $(OBJS) probe.o
	$(FC) $(FFLAGS) $^ -o $@
lw_binio.o: src/lw_binio.F90
	$(FC) $(FFLAGS) -c $< -o $@
lw_dump_state.o: src/lw_dump_state.F90 lw_binio.o module_ra_rrtmg_lw.o
	$(FC) $(FFLAGS) -c $< -o $@
columns.o: src/columns.F90 mpas_atmphys_rrtmg_lwinit.o lw_dump_state.o
	$(FC) $(FFLAGS) -c $< -o $@
columns: mpas_kind_types.o mpas_constants.o stubs.o module_ra_rrtmg_vinterp.o module_ra_rrtmg_lw.o mpas_atmphys_rrtmg_lwinit.o lw_binio.o lw_dump_state.o columns.o
	$(FC) $(FFLAGS) $^ -o $@
randoms.o: src/randoms.F90 module_ra_rrtmg_lw.o lw_binio.o
	$(FC) $(FFLAGS) -c $< -o $@
randoms: $(OBJS) lw_binio.o randoms.o
	$(FC) $(FFLAGS) $^ -o $@
mcica.o: src/mcica.F90 mpas_atmphys_rrtmg_lwinit.o lw_binio.o
	$(FC) $(FFLAGS) -c $< -o $@
mcica: $(OBJS) lw_binio.o mcica.o
	$(FC) $(FFLAGS) $^ -o $@
solver.o: src/solver.F90 mpas_atmphys_rrtmg_lwinit.o lw_binio.o
	$(FC) $(FFLAGS) -c $< -o $@
solver: $(OBJS) lw_binio.o solver.o
	$(FC) $(FFLAGS) $^ -o $@
mcica_multi.o: src/mcica_multi.F90 mpas_atmphys_rrtmg_lwinit.o lw_binio.o
	$(FC) $(FFLAGS) -c $< -o $@
mcica_multi: $(OBJS) lw_binio.o mcica_multi.o
	$(FC) $(FFLAGS) $^ -o $@
fatal_paths.o: src/fatal_paths.F90 mpas_atmphys_rrtmg_lwinit.o
	$(FC) $(FFLAGS) -c $< -o $@
fatal_paths: $(OBJS) fatal_paths.o
	$(FC) $(FFLAGS) $^ -o $@
real_columns.o: src/real_columns.F90 mpas_atmphys_rrtmg_lwinit.o lw_dump_state.o
	$(FC) $(FFLAGS) -c $< -o $@
real_columns: $(OBJS) lw_binio.o lw_dump_state.o real_columns.o
	$(FC) $(FFLAGS) $^ -o $@
diffusivity.o: src/diffusivity.F90 module_ra_rrtmg_lw.o lw_binio.o
	$(FC) $(FFLAGS) -c $< -o $@
diffusivity: $(OBJS) lw_binio.o diffusivity.o
	$(FC) $(FFLAGS) $^ -o $@
setcoef_columns.o: src/setcoef_columns.F90 mpas_atmphys_rrtmg_lwinit.o lw_binio.o
	$(FC) $(FFLAGS) -c $< -o $@
setcoef_columns: $(OBJS) lw_binio.o setcoef_columns.o
	$(FC) $(FFLAGS) $^ -o $@
vinterp_columns.o: src/vinterp_columns.F90 module_ra_rrtmg_vinterp.o lw_binio.o
	$(FC) $(FFLAGS) -c $< -o $@
vinterp_columns: mpas_kind_types.o module_ra_rrtmg_vinterp.o lw_binio.o vinterp_columns.o
	$(FC) $(FFLAGS) $^ -o $@

optional_columns.o: src/optional_columns.F90 mpas_atmphys_rrtmg_lwinit.o lw_dump_state.o
	$(FC) $(FFLAGS) -c $< -o $@
optional_columns: $(OBJS) lw_binio.o lw_dump_state.o optional_columns.o
	$(FC) $(FFLAGS) $^ -o $@

subnormal_pressure.o: src/subnormal_pressure.F90 lw_binio.o
	$(FC) $(FFLAGS) -c $< -o $@
subnormal_pressure: lw_binio.o subnormal_pressure.o
	$(FC) $(FFLAGS) $^ -o $@

extended_solver.o: src/extended_solver.F90 mpas_atmphys_rrtmg_lwinit.o lw_binio.o
	$(FC) $(FFLAGS) -c $< -o $@
extended_solver: $(OBJS) lw_binio.o extended_solver.o
	$(FC) $(FFLAGS) $^ -o $@

short_mcica.o: src/short_mcica.F90 mpas_atmphys_rrtmg_lwinit.o lw_binio.o
	$(FC) $(FFLAGS) -c $< -o $@
short_mcica: $(OBJS) lw_binio.o short_mcica.o
	$(FC) $(FFLAGS) $^ -o $@

fixture_words: fixture_words.rs
	rustc --edition 2021 -O $< -o $@

# p22-rrtmg-lw: complete native one-level ozone wrapper.
one_level_ozone_columns.o: src/one_level_ozone_columns.F90 mpas_atmphys_rrtmg_lwinit.o lw_dump_state.o
	$(FC) $(FFLAGS) -c $< -o $@
one_level_ozone_columns: $(OBJS) lw_binio.o lw_dump_state.o one_level_ozone_columns.o
	$(FC) $(FFLAGS) $^ -o $@

vinterp_edges.o: src/vinterp_edges.F90 module_ra_rrtmg_vinterp.o lw_binio.o
	$(FC) $(FFLAGS) -c $< -o $@
vinterp_edges: mpas_kind_types.o module_ra_rrtmg_vinterp.o lw_binio.o vinterp_edges.o
	$(FC) $(FFLAGS) $^ -o $@
