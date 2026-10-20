FC = gfortran
FFLAGS = -O3 -fPIC -cpp -ffree-form -ffree-line-length-none -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations
all: libles.so
.PHONY: tap-records
tap-records:
	sed 's|^//!|//|' $(SUPPORT)/../../tools/native_physics_reference/oracle/oracle_words.rs > f1_words.rs
	D2_TAP_F1=$(CURDIR)/f1_words.rs rustc --edition=2021 -C opt-level=2 $(SUPPORT)/tap_records.rs -o $@
real-columns: real_columns.rs
	rustc --edition=2021 -C opt-level=2 real_columns.rs -o $@
real-module: real_module.rs real_columns.rs real_template.rs real_invoker.c
	$(CC) -O2 -c real_invoker.c -o real_invoker.o
	rustc --edition=2021 -C opt-level=2 real_module.rs -C link-arg=real_invoker.o -o $@
libles.so: stubs.f90 mpas_constants.F mpas_atm_dissipation_models.F wrappers.f90
	$(FC) $(FFLAGS) -shared stubs.f90 mpas_constants.F mpas_atm_dissipation_models.F wrappers.f90 -o $@
inspect-native: inspect_native.rs
	rustc --edition=2021 -C opt-level=2 inspect_native.rs -o $@
inspect-text: inspect_text.rs
	rustc --edition=2021 -C opt-level=2 inspect_text.rs -o $@
