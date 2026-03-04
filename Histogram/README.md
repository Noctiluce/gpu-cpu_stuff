# CUDA RGB Histogram – Performance Exploration

This project explores histogram computation on GPU using CUDA, with a focus on understanding warp scheduling, atomic contention, shared memory usage, and memory hierarchy behavior.

The goal is not just to compute a histogram, but to study how different architectural decisions impact performance on modern NVIDIA GPUs.

The test case uses a 6K × 4K RGB image (24 million pixels) with 256 bins per channel.

---

## Project Overview

Histogram computation is a deceptively simple problem. In practice, it reveals important GPU performance characteristics such as atomic contention, warp-level execution behavior, shared memory bank conflicts, occupancy limits, and memory latency hiding.

Three progressively optimized CUDA kernels were implemented and benchmarked.

---

## Implementations

### Naive GPU Kernel

Each thread directly updates the global histogram using `atomicAdd` in global memory.

This version is straightforward but heavily limited by atomic contention.

Observed runtime (6K × 4K, 256 bins):  
~12 ms

---

### Shared Memory Per Block

Each block maintains a private histogram in shared memory. Threads update the shared histogram using `atomicAdd`, and at the end of the block execution, results are merged into global memory.

This significantly reduces global atomic contention.

Observed runtime:  
~2.8 ms

---

### Warp-Private Histograms + Strided Loop

Each warp maintains a private histogram in shared memory, reducing intra-block contention further.

Additional optimizations include:

- Strided loop to increase workload per thread
- Read-only cache loads
- Fast float-to-int conversion
- Reduced synchronization pressure

Observed runtime:  
~1.22 ms

---

## CPU Baseline

A reference CPU implementation was used for comparison.

On the same 6K × 4K image:

CPU runtime: ~150 ms  
Optimized GPU runtime: ~1.22 ms

This represents roughly two orders of magnitude speedup.

---

## RGB Histogram Support

The project computes separate histograms for R, G, and B channels, and generates:

- Individual histograms per channel (CPU and GPU)
- A combined RGB histogram image for CPU
- A combined RGB histogram image for GPU

The combined RGB histogram visualizes all three channel distributions in a single image.

---


## Key Takeaways

Performance in histogram kernels is dominated by atomic contention rather than raw memory bandwidth.

Moving atomics from global memory to shared memory drastically improves runtime.

Warp-private histograms reduce contention further by isolating updates at warp granularity.

Loop striding improves occupancy and helps hide latency by increasing arithmetic intensity per thread.

Optimizing GPU code is not only about FLOPS. It requires understanding how warps are scheduled and how the memory hierarchy interacts with contention.

---

## Motivation

Histogram computation is widely used in computer vision, image processing, and scientific computing.

Although conceptually simple, it serves as a compact and powerful case study for understanding real GPU performance bottlenecks.

This project highlights how architectural awareness leads to meaningful performance gains.