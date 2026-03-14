# 1-Billion-Row Challenge — Build targets
#
# Usage:
#   make             — build all targets
#   make baseline    — single-threaded CPU baseline (g++)
#   make baseline_mt — multi-threaded CPU baseline (g++ + pthreads)
#   make solution    — CUDA GPU solution (nvcc)
#   make clean       — remove binaries

CXX      := g++
NVCC     := nvcc
CXXFLAGS := -O3
NVFLAGS  := -O3 -arch=sm_70

.PHONY: all clean

all: baseline baseline_mt solution

baseline: baseline.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

baseline_mt: baseline_mt.cpp
	$(CXX) $(CXXFLAGS) -pthread -o $@ $<

solution: solution.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

clean:
	rm -f baseline baseline_mt solution
