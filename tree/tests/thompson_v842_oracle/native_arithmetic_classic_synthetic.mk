.PHONY: all
all:
	python hex/tree/tests/thompson_v842_oracle/prepare_host.py n1out/native-arithmetic-classic
	g++ -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -std=c++17 -fPIC -shared n1out/native-arithmetic-classic/host_port.cpp -o n1out/native-arithmetic-classic/host_port.so
