# Read-only module adapter, linked to the existing original native library.
OUT ?= n1out/native-storage-observer
DEPS ?= scratch/wrapper-log-native-deps
SOURCE := $(dir $(lastword $(MAKEFILE_LIST)))native_storage_probe.f90
$(OUT)/libr14_native_storage_probe.so: $(SOURCE)
	mkdir -p $(OUT)
	gfortran -O3 -ffree-line-length-none -fconvert=big-endian -ffree-form -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -fPIC -I$(DEPS)/modules -J$(OUT) -shared $(SOURCE) -L$(DEPS) -lmpasphys_native_rrfs -Wl,-rpath,$(abspath $(DEPS)) -o $@
