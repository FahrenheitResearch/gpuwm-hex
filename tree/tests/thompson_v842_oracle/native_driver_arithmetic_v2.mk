.PHONY: all
all:
	python hex/tree/tests/thompson_v842_oracle/native_arithmetic_replay.py --generate-host n1out/native-driver-arithmetic-v2-classic
	g++ -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -std=c++17 -fPIC -shared n1out/native-driver-arithmetic-v2-classic/host_shared_native_log10.cpp -o n1out/native-driver-arithmetic-v2-classic/host_shared_native_log10.so
	python hex/tree/tests/thompson_v842_oracle/native_arithmetic_replay.py --compile n1out/native-driver-arithmetic-v2-classic/images
	python hex/tree/tests/thompson_v842_oracle/replay_invalid_sqrt.py generate-host --directory n1out/native-driver-arithmetic-v2-aerosol
	g++ -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -fPIC -shared -std=c++17 n1out/native-driver-arithmetic-v2-aerosol/host.cpp -o n1out/native-driver-arithmetic-v2-aerosol/host.so
	python hex/tree/tests/thompson_v842_oracle/replay_invalid_sqrt.py build --directory n1out/native-driver-arithmetic-v2-aerosol
