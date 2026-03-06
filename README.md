# GPU Compute Projects (CUDA / OpenCL)

This repository contains GPU-accelerated projects focused on understanding and benchmarking parallel computing using **CUDA and OpenCL**.
Each project is self-contained and comes with its own build system.
The goal is mainly experimental: validating assumptions about GPU architecture, scheduling, memory, and performance through practical code.

---

## Current Project

- [`VectorAndMatricesOperations/`](https://github.com/Noctiluce/gpu_stuff/tree/main/VectorAndMatricesOperations) – Vector and matrix operations to explore GPU behavior, occupancy, and memory optimization.
- [`GPUConvolution/`](https://github.com/Noctiluce/gpu_stuff/tree/main/GPUConvolution) – CUDA implementation of a 2D convolution, optimized and profiled with NVIDIA Nsight Compute.
- [`Histogram/`](https://github.com/Noctiluce/gpu_stuff/tree/main/Histogram) – Histogram focused on warp scheduling, atomic contention, shared memory usage, and memory hierarchy behavior.

---

## Build

Each project has its own CMake configuration.

Example:

```bash
cd project_name
mkdir build && cd build
cmake ..
make
```
