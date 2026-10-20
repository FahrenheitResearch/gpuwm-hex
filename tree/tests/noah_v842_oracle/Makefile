FC = gfortran
FFLAGS = -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-form -ffree-line-length-none -fconvert=big-endian -fallow-argument-mismatch -fPIC --coverage
SOURCE = $(MPAS_REF)/src/core_atmosphere/physics/physics_wrf
all: libnoah_oracle.so libnoah_host.so
stubs.o: stubs.F90
	$(FC) $(FFLAGS) -c $< -o $@
module_sf_noahlsm.F module_sf_noahlsm_glacial_only.F module_sf_noahdrv.F:
	cp $(SOURCE)/$@ $@
module_sf_noahlsm.o: module_sf_noahlsm.F stubs.o
	$(FC) $(FFLAGS) -cpp -Dmpas -c $< -o $@
module_sf_noahlsm_glacial_only.o: module_sf_noahlsm_glacial_only.F module_sf_noahlsm.o
	$(FC) $(FFLAGS) -cpp -Dmpas -c $< -o $@
module_sf_noahdrv.o: module_sf_noahdrv.F module_sf_noahlsm_glacial_only.o
	$(FC) $(FFLAGS) -cpp -Dmpas -c $< -o $@
replay.o: replay.F90 module_sf_noahlsm_glacial_only.o
	$(FC) $(FFLAGS) -c $< -o $@
native_tables.o: native_tables.F90 module_sf_noahlsm.o
	$(FC) $(FFLAGS) -cpp -c $< -o $@
driver_replay.o: driver_replay.F90 module_sf_noahdrv.o
	$(FC) $(FFLAGS) -c $< -o $@
init_replay.o: init_replay.F90 module_sf_noahlsm.o
	$(FC) $(FFLAGS) -c $< -o $@
landuse_replay.o: landuse_replay.F90
	$(FC) $(FFLAGS) -c $< -o $@
private_helpers.F90: build_helpers.py
	python build_helpers.py --private-only
private_helpers.o: private_helpers.F90 module_sf_noahlsm.o
	$(FC) $(FFLAGS) -c $< -o $@
helper_replay.o: helper_replay.F90 module_sf_noahlsm_glacial_only.o
	$(FC) $(FFLAGS) -c $< -o $@
libnoah_oracle.so: private_helpers.o helper_replay.o io_replay.o landuse_replay.o module_sf_noahdrv.o replay.o native_tables.o driver_replay.o init_replay.o
	$(FC) $(FFLAGS) -shared stubs.o module_sf_noahlsm.o module_sf_noahlsm_glacial_only.o module_sf_noahdrv.o replay.o native_tables.o driver_replay.o init_replay.o landuse_replay.o io_replay.o helper_replay.o private_helpers.o -o $@

host.cpp: helper_entry.cu build_host.py ../../src/hexcore/mpasphys/noah_v842.py
	python build_host.py
libnoah_host.so: host.cpp
	g++ -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -fno-builtin-powf -fPIC -shared $< -o $@

.PHONY: tables
tables:
	install -m 644 $(SOURCE)/files/VEGPARM.TBL $(SOURCE)/files/SOILPARM.TBL $(SOURCE)/files/GENPARM.TBL $(SOURCE)/files/LANDUSE.TBL .

# Standalone build of the Rust input bridge proposed for rw-mpas registration.
table_decode: table_decode.rs table_schema.rs
	rustc --edition=2024 -O $< -o $@

io_replay.F90: build_io_oracle.py
	python build_io_oracle.py
io_replay.o: io_replay.F90
	$(FC) $(FFLAGS) -c $< -o $@

pack_real_init: pack_real_init.rs
	rustc --edition=2024 -O $< -o $@

dump_transport: dump_transport.rs dump_layout.rs table_schema.rs
	rustc --edition=2024 -O $< -o $@

pack_real_driver: pack_real_driver.rs dump_layout.rs
	rustc --edition=2024 -O $< -o $@

compare_noah_call: compare_noah_call.rs
	rustc --edition=2024 -O $< -o $@

pack_dual_driver_input: pack_dual_driver_input.rs
	rustc --edition=2024 -O $< -o $@

# Explicit CPU-only NVRTC target; no device discovery or GPU context creation.
.PHONY: cuda-cubins
cuda-cubins:
	python precompile_cuda.py --arch $(NOAH_CUDA_ARCH) --output $(CUDA_CUBIN_OUTPUT)
