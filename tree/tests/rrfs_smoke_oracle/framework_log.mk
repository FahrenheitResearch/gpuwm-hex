# Test adapter links the existing pinned backend; no native reference rebuild.
OUT ?= n1out/wrapper-log-framework
DEPS ?= scratch/wrapper-log-native-deps
SOURCE := $(dir $(lastword $(MAKEFILE_LIST)))framework_log_probe.f90
$(OUT)/libr14_framework_log_probe.so: $(SOURCE)
	mkdir -p $(OUT)
	gfortran -O3 -ffree-line-length-none -fconvert=big-endian -ffree-form -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -fPIC -I$(DEPS)/modules -J$(OUT) -shared $(SOURCE) -L$(DEPS) -lmpasphys_native_rrfs -Wl,-rpath,$(abspath $(DEPS)) -o $@
