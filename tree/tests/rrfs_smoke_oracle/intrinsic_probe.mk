HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
OUT ?= n1out/native-intrinsic-probe
FLAGS := -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations
all: $(OUT)/libsmoke_intrinsic_probe.so $(OUT)/intrinsic_probe.s
$(OUT):
	mkdir -p $@
$(OUT)/libsmoke_intrinsic_probe.so: $(HERE)/intrinsic_probe.F90 | $(OUT)
	gfortran $(FLAGS) -fPIC -shared $< -o $@
$(OUT)/intrinsic_probe.s: $(HERE)/intrinsic_probe.F90 | $(OUT)
	gfortran $(FLAGS) -S $< -o $@
