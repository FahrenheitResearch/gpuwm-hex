.PHONY: all
all:
	python hex/tree/tests/thompson_v842_oracle/replay_invalid_sqrt.py generate-host --directory n1out/invalid-sqrt-whole-aerosol
	g++ -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -fPIC -shared -std=c++17 -o n1out/invalid-sqrt-whole-aerosol/host.so n1out/invalid-sqrt-whole-aerosol/host.cpp
