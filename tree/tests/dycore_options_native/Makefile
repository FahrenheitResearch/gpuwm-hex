FC=gfortran
FLAGS=-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-form -ffree-line-length-none -fconvert=big-endian -fPIC -cpp $(D1_CPPFLAGS)
all: libdissipation.so
stubs.o: stubs.f90
	$(FC) $(FLAGS) -c $<
mpas_constants.o: mpas_constants.F stubs.o
	$(FC) $(FLAGS) -cpp -c $<
mpas_atm_dissipation_models.o: mpas_atm_dissipation_models.F mpas_constants.o
	$(FC) $(FLAGS) -cpp -c $<
libdissipation.so: stubs.o mpas_constants.o mpas_atm_dissipation_models.o
	$(FC) -shared -o $@ $^
upper_blocks.o: upper_blocks.f90 stubs.o
	$(FC) $(FLAGS) -c $<
deck: deck.f90 upper_blocks.o stubs.o mpas_constants.o mpas_atm_dissipation_models.o
	$(FC) $(FLAGS) -o $@ $^
length-deck: core_length.f90
	$(FC) $(FLAGS) -o $@ $<
subnormal-deck: subnormal_deck.f90 upper_blocks.o
	$(FC) $(FLAGS) -o $@ $^
