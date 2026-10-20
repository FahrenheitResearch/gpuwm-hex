.PHONY: all host transport
FC = gfortran
FLAGS = -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -fconvert=big-endian -ffree-form -cpp -Dtempo_mpas -fPIC --coverage
S = ../tempo-src
objects = mpas_kind_types.o module_mp_tempo_cfgs.o module_mp_tempo_params.o module_mp_tempo_utils.o module_mp_tempo_ml.o module_mp_tempo_aerosols.o module_mp_tempo_diags.o module_mp_tempo_main.o module_mp_tempo_driver.o oracle.o
all: libtempo_oracle.so libtempo_glue.so libtempo_diagnostics.so libtempo_driver_glue.so
libtempo_oracle.so: $(objects)
	$(FC) $(FLAGS) -shared -o $@ $(objects)
rrfs: libtempo_rrfs_oracle.so libtempo_resource_transport.so libtempo_pipeline.so
pipeline: libtempo_pipeline.so
libtempo_pipeline.so: host.cpp ../../hex/tree/tests/tempo_rrfs/host_pipeline.cpp
	g++ -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -std=c++17 -fPIC -shared -I. ../../hex/tree/tests/tempo_rrfs/host_pipeline.cpp -o $@
# Bind only the test ABI to r2's published pristine full fork library. No scheme
# or framework object from the standalone oracle is linked into this witness.
libtempo_rrfs_oracle.so: oracle.o ../r2-native/libmpasphys_native_rrfs.so
	$(FC) $(FLAGS) -shared -o $@ oracle.o -L../r2-native -lmpasphys_native_rrfs -Wl,-rpath,'$$ORIGIN/../r2-native'
libtempo_resource_transport.so: ../../hex/tree/tests/tempo_rrfs/transport.rs
	rustc --edition=2021 --crate-type cdylib -O $< -o $@
%.o: $(S)/%.F90
	$(FC) $(FLAGS) -c $< -o $@
module_mp_tempo_params.o: mpas_kind_types.o
module_mp_tempo_utils.o module_mp_tempo_ml.o module_mp_tempo_aerosols.o: module_mp_tempo_params.o
module_mp_tempo_diags.o: module_mp_tempo_params.o module_mp_tempo_utils.o
module_mp_tempo_main.o: module_mp_tempo_cfgs.o module_mp_tempo_params.o module_mp_tempo_utils.o module_mp_tempo_ml.o module_mp_tempo_aerosols.o module_mp_tempo_diags.o
module_mp_tempo_driver.o: module_mp_tempo_main.o
oracle.o: ../../hex/tree/tests/tempo_rrfs/oracle.F90 module_mp_tempo_driver.o
	$(FC) $(FLAGS) -c $< -o $@
host: libtempo_host.so libtempo_audit.so
libtempo_host.so: host.cpp
	g++ -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -std=c++17 -fPIC -shared host.cpp -o $@
transport: ../../n1out/libtempo_transport.so
../../n1out/libtempo_transport.so: ../../hex/tree/tests/tempo_rrfs/transport.rs
	rustc --edition=2021 --crate-type cdylib -O $< -o $@

libtempo_glue.so: glue_oracle.F90
	$(FC) $(FLAGS) -shared $< -o $@

libtempo_diagnostics.so: diagnostic_oracle.F90 module_mp_tempo_params.o
	$(FC) $(FLAGS) -shared $< module_mp_tempo_params.o mpas_kind_types.o -o $@

libtempo_audit.so: host.cpp
	g++ -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -std=c++17 -fPIC -shared -DTEMPO_MATH_AUDIT host.cpp -o $@

libtempo_driver_glue.so: driver_glue_oracle.F90
	$(FC) $(FLAGS) -shared $< -o $@
