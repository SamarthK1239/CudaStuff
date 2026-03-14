# CudaStuff

Collection of CUDA and systems-performance experiments, currently centered on a 1 Billion Row Challenge implementation with:

- single-threaded CPU baseline
- multi-threaded CPU baseline
- CUDA GPU solution
- data generator and benchmark script

## Project Layout

```text
billion-row-challenge/
	baseline.cpp       # single-threaded CPU implementation
	baseline_mt.cpp    # multi-threaded CPU implementation
	solution.cu        # CUDA implementation
	main.py            # synthetic dataset generator
	benchmark.sh       # build + run + verify + compare timings
	Makefile           # build targets
```

## Prerequisites

- Linux
- `g++` with pthread support
- NVIDIA CUDA toolkit (`nvcc`)
- Python 3 (for dataset generation)
- Optional: `numactl` (used by benchmark script when available)

## Build

From the repository root:

```bash
cd billion-row-challenge
make
```

Available targets:

```bash
make baseline
make baseline_mt
make solution
make clean
```

## Generate Input Data

Default (1 billion rows):

```bash
cd billion-row-challenge
python3 main.py
```

Smaller test dataset:

```bash
python3 main.py --rows 10000000 --output measurements-10m.txt
```

Input format per line:

```text
StationName;temperature
```

## Run Implementations

```bash
cd billion-row-challenge
./baseline measurements.txt
./baseline_mt measurements.txt
./solution measurements.txt
```

## Benchmark End-to-End

The benchmark script builds all targets, runs CPU and GPU versions, verifies output equality, and prints a timing summary.

```bash
cd billion-row-challenge
./benchmark.sh
```

Use a custom data file:

```bash
./benchmark.sh measurements-10m.txt
```

## Notes

- Large generated/input text datasets (for example `measurements.txt`) are intentionally ignored by git.
- Local build binaries are also ignored via `.gitignore`.
- The CUDA build in this repo currently targets `sm_70` (V100-class GPUs) in `billion-row-challenge/Makefile`.