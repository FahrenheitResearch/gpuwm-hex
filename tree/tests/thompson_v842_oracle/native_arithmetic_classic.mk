.PHONY: all
all:
	python hex/tree/tests/thompson_v842_oracle/native_arithmetic_replay.py --generate-host n1out/native-arithmetic-classic
	g++ -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -std=c++17 -fPIC -shared n1out/native-arithmetic-classic/host_shared_native_log10.cpp -o n1out/native-arithmetic-classic/host_shared_native_log10.so
	python hex/tree/tests/thompson_v842_oracle/native_arithmetic_replay.py --compile n1out/native-arithmetic-classic/images
