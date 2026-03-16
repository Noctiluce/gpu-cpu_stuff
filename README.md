# GPU Compute Projects (CUDA / OpenCL)

This repository contains GPU-accelerated projects focused on understanding and benchmarking parallel computing using **CUDA and OpenCL**.
Each project is self-contained and comes with its own build system.
The goal is mainly experimental: validating assumptions about GPU architecture, scheduling, memory, and performance through practical code.

---

## Current Project

- [`VectorAndMatricesOperations/`](https://github.com/Noctiluce/gpu_stuff/tree/main/VectorAndMatricesOperations) – A small hands-on project to visualize how GPUs actually behave with threads, blocks, warps, and SMs.
- [`GPUConvolution/`](https://github.com/Noctiluce/gpu_stuff/tree/main/GPUConvolution) – Hand-written CUDA implementation of a 2D convolution (up to 7×7 kernel), optimized and profiled with NVIDIA Nsight Compute.
- [`Histogram/`](https://github.com/Noctiluce/gpu_stuff/tree/main/Histogram) – This project explores histogram computation on GPU using CUDA, with a focus on understanding warp scheduling, atomic contention, shared memory usage, and memory hierarchy behavior.
- [`False Sharing/`](https://github.com/Noctiluce/gpu-cpu_stuff/tree/main/FalseSharing) – A micro-benchmark in C++ demonstrating how false sharing can significantly degrade the performance of multithreaded applications.
- [`Cache Misses/`](https://github.com/Noctiluce/gpu-cpu_stuff/tree/main/CacheMissesImpactLatency) – C++ benchmarks illustrating the impact of cache misses on latency, with concrete HFT (High-Frequency Trading) use cases.
- [`Basic CNN/`](https://github.com/Noctiluce/gpu-cpu_stuff/tree/main/BasicCNN) – A convolutional neural network implemented entirely from scratch in C++ and CUDA. No ML framework, no cuDNN, no autograd. Every component - forward pass, backpropagation, Adam optimiser, and GPU kernels - is written by hand.
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
