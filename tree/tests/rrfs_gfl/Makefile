.PHONY: all host
FC = gfortran
FFLAGS = -O3 -cpp -ffree-form -ffree-line-length-none -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -fdump-tree-original -fPIC --coverage
all: gfl_native.so
stubs.o: stubs.F90
	$(FC) $(FFLAGS) -c $<
module_cu_gfl_common.o: module_cu_gfl_common.F90 stubs.o
	$(FC) $(FFLAGS) -c $<
module_cu_gfl_deep.o: module_cu_gfl_deep.F90 module_cu_gfl_common.o
	$(FC) $(FFLAGS) -c $<
module_cu_gfl_sh.o: module_cu_gfl_sh.F90 module_cu_gfl_deep.o
	$(FC) $(FFLAGS) -c $<
module_cu_gfl_mpas.o: module_cu_gfl_mpas.F90 module_cu_gfl_sh.o
	$(FC) $(FFLAGS) -c $<
driver_slice.o: driver_slice.F90
	$(FC) $(FFLAGS) -c $<
gfl_native.so: topology_slice.o driver_slice.o stubs.o module_cu_gfl_common.o module_cu_gfl_deep.o module_cu_gfl_sh.o module_cu_gfl_mpas.o
	$(FC) $(FFLAGS) -shared $^ -o $@

host: gfl_host.so
gfl_host.so: host.cpp gfl_support.cuh gfl_lowered.cuh gfl_entry.cuh
	g++ -std=c++17 -O3 -fPIC -shared -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations host.cpp -o $@

topology_slice.o: topology_slice.F90
	$(FC) $(FFLAGS) -c $<

.PHONY: transport
transport: oracle_transport
oracle_transport: oracle_transport.rs
	rustc --edition=2021 -O $< -o $@
