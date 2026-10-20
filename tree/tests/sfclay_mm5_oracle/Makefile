FFLAGS = -O3 -fPIC -cpp -Dmpas -ffree-form -ffree-line-length-none -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations --coverage -fdump-tree-original -fdump-tree-optimized
all: liboracle.so libhost.so
mpas_kind_types.o: mpas_kind_types.F90
	gfortran $(FFLAGS) -c $<
ccpp_kind_types.o: ccpp_kind_types.F90
	gfortran $(FFLAGS) -c $<
sf_sfclayrev.o: sf_sfclayrev.F90 ccpp_kind_types.o
	gfortran $(FFLAGS) -c $<
sf_sfclayrev_pre.o: sf_sfclayrev_pre.F ccpp_kind_types.o
	gfortran $(FFLAGS) -c $<
module_sf_sfclayrev.o: module_sf_sfclayrev.F mpas_kind_types.o sf_sfclayrev_pre.o sf_sfclayrev.o
	gfortran $(FFLAGS) -c $<
module_sf_sfclay.o: module_sf_sfclay.F
	gfortran $(FFLAGS) -c $<
module_sf_sfcdiags.o: module_sf_sfcdiags.F
	gfortran $(FFLAGS) -c $<
oracle_wrapper.o: oracle_wrapper.F90 sf_sfclayrev.o module_sf_sfclay.o module_sf_sfclayrev.o
	gfortran $(FFLAGS) -c $<
driver_oracle.o: driver_oracle.F90 module_sf_sfcdiags.o
	gfortran $(FFLAGS) -c $<
liboracle.so: driver_oracle.o mpas_kind_types.o ccpp_kind_types.o sf_sfclayrev.o sf_sfclayrev_pre.o module_sf_sfclayrev.o module_sf_sfclay.o module_sf_sfcdiags.o oracle_wrapper.o
	gfortran --coverage -shared $^ -o $@
libhost.so: host_bridge.cpp
	g++ -O3 -fPIC -shared -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations $< -o $@
