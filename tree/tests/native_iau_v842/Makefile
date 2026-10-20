# Sources mpas_atm_iau.F and mpas_constants.F are copied from the read-only
# pinned MPAS tree by build.sh. This builds only the standalone module.
FC = gfortran
FFLAGS = -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -fconvert=big-endian -ffree-form -cpp -fPIC

all: libiau_reference.so

framework.o: framework.f90
	$(FC) $(FFLAGS) -c $<

mpas_constants.o: mpas_constants.F framework.o
	$(FC) $(FFLAGS) -c $<

mpas_atm_iau.o: mpas_atm_iau.F framework.o mpas_constants.o
	$(FC) $(FFLAGS) -c $<

wrapper.o: wrapper.f90 mpas_atm_iau.o
	$(FC) $(FFLAGS) -c $<

libiau_reference.so: framework.o mpas_constants.o mpas_atm_iau.o wrapper.o
	$(FC) -shared -o $@ $^
