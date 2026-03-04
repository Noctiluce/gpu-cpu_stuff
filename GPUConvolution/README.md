# 🚀 CUDA 2D Convolution — ×400 Speedup vs CPU

Hand-written CUDA implementation of a 2D convolution (up to 7×7 kernel), optimized and profiled with NVIDIA Nsight Compute.

No frameworks. Pure CUDA.

---

## 📌 Overview

This project implements a 2D convolution on both:

- CPU (baseline C++ implementation)
- GPU (custom CUDA kernel)

The goal is to analyze architectural performance differences and identify the main optimization levers on GPU.

---

## ⚙️ Key Optimization Decisions

### 1️⃣ Shared Memory Tiling with Halo

Each block loads a tile (block + border) into SRAM (shared memory) instead of hitting global memory (DRAM) on every access.

For a 7×7 convolution, each pixel contributes to up to 49 output values. Without tiling, that's up to 49 redundant global memory reads for the same pixel across neighboring threads. With tiling, the data is loaded once into shared memory and reused by all threads in the block.

```
Global Memory (DRAM)  ~600 cycles latency
Shared Memory (SRAM)  ~20 cycles latency
```

This is the single highest-impact optimization in the kernel.

---

### 2️⃣ Thread Block Shape — 32×8 over 16×16

A `32×8` block aligns the X dimension to the warp size (32 threads), ensuring coalesced global memory access on every load.

With a `16×16` block, each warp spans two rows of the image. Since image data is stored row-major in memory, the second half of the warp accesses a non-contiguous address range — breaking coalescing and doubling the number of memory transactions.

```
16×16 → warp spans 2 rows → 2 memory transactions per warp
32×8  → warp spans 1 row  → 1 memory transaction per warp
```

This single choice impacts every memory transaction in the kernel.

---

### 3️⃣ Latency Hiding via Warp-Level Parallelism

The RTX 2070 Super has 40 SMs, each capable of running up to 32 warps concurrently. When a warp stalls on a global memory load during tile population, the scheduler immediately switches to another warp at zero cost.

The `32×8` block shape combined with a large grid ensures enough warps in flight to keep all SMs busy. This is why throughput scales almost linearly from 1k to 6k images, but drops off at 256px — at small resolutions, the grid is too small to saturate all SMs and some warps remain idle.

```
256px  → grid too small → SMs underutilized → ×139 speedup
6000px → grid saturates all SMs             → ×418 speedup
```

---

## 📊 Performance Results

Measured using **NVIDIA Nsight Compute**
Hardware: **RTX 2070 Super**
Kernel: 7×7 edge sharpen
Scene: Sponza

|Resolution	|CPU (ms)	| GPU (ms)	 |Speedup|
|-----------|-----------|-----------|-------|
| 6000px	| 836	    | 2.00	     | ×418  |
| 4000px	| 367	    | 0.89	     | ×412  |
| 2000px	| 92	    | 0.23	      | ×400  |
| 1000px	| 22	    | 0.06	     | ×366  |
|  512px	| 5.67	    | 0.02	     | ×285  |
|  256px	| 1.39	    | 0.01	     | ×139  |

**Average speedup: ×412**

---

## 🧠 Observations

- The workload is primarily memory-bound.
- Shared memory is the dominant performance factor.
- GPU scales near-linearly with image size.
- On small images, kernel launch overhead reduces speedup.

---

## 🎯 What This Project Demonstrates

- GPU acceleration is architecture-driven, not magic.
- Memory hierarchy understanding is critical.
- Proper tiling dominates arithmetic optimizations.
- Small kernels benefit strongly from constant memory.
