# Compile the lane reader (landed ORCL0001 plus native framework int64).
# OUT is owned lane scratch; the source checkout remains read-only.
OUT ?= n1out/native-capture-transport
SOURCE := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/extended_transport.rs)
.PHONY: all
all: $(OUT)/libmpas_oracle_transport.so
$(OUT)/libmpas_oracle_transport.so: $(SOURCE)
	mkdir -p $(OUT)
	rustc --edition=2024 --crate-type cdylib -O $(SOURCE) -o $@
