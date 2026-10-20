all: scratch/libsmoke_host.so
scratch/libsmoke_host.so: scratch/host.cpp scratch/port.cuh scratch/host_prefix.cuh
	g++ -std=c++17 -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -fPIC -shared -o $@ scratch/host.cpp > receipts/host-build.log 2>&1
