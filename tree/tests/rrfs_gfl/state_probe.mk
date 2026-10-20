include synthetic.mk
state_probe: state_probe.F90 native_1014ad01e035b801.o native_3b995277426d4809.o native_a26417fe934ee767.o native_8b95217d8134ef0a.o native_c67997579ae31630.o native_c480a5d0d3b35120.o native_cf16c9929a7bf8e9.o native_6e5c1503606f74a6.o
	$(FC) $(FLAGS) $^ -o $@
