# False Sharing Benchmark - C++

A micro-benchmark in C++ demonstrating how **false sharing** can significantly degrade the performance of multithreaded applications.

This project compares two scenarios:

1. Threads updating independent variables located on the **same cache line** (false sharing)
2. Threads updating independent variables located on **separate cache lines** (properly aligned)

The benchmark highlights how **memory layout alone** can dramatically impact performance in low-latency systems.

---

## What is False Sharing?

Modern CPUs do not operate directly on individual variables in memory.  
Instead, memory is transferred between RAM and CPU caches in fixed-size blocks called **cache lines** (typically **64 bytes**).

When multiple threads update variables that reside on the **same cache line**, even if the variables are logically independent, the CPU cache coherence protocol forces the cache line to **bounce between cores**.

This results in:

- frequent cache invalidations
- expensive cache coherence traffic
- severe performance degradation

This phenomenon is known as **false sharing**.

```
Cache line (64 bytes)
┌──────────────────────────────────────────────────────┐
│  counter[0] │ counter[1] │ counter[2] │ counter[3]   │
└──────────────────────────────────────────────────────┘
        ↑             ↑             ↑             ↑
    Thread 0      Thread 1      Thread 2      Thread 3
```

Even though each thread updates a different counter, they all compete for the **same cache line**.

---

## Avoiding False Sharing

False sharing can be avoided by ensuring that each frequently-written variable resides on its **own cache line**, using:

- `alignas(64)`
- manual padding
- `std::hardware_destructive_interference_size`

```cpp
struct alignas(64) PaddedCounter {
    volatile int value;
};
```

Each counter now occupies its own cache line, preventing cross-core invalidations:

```
Cache line 0      Cache line 1      Cache line 2      Cache line 3
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│ counter[0] │    │ counter[1] │    │ counter[2] │    │ counter[3] │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
```

---

## Benchmark Overview

### 1. False Sharing

Counters are stored contiguously in memory. Multiple threads update different elements of the same array, which share the same cache line.

```cpp
struct FalseSharingCounters {
    volatile int counters[NUM_THREADS];
};
```

### 2. No False Sharing

Each counter is aligned to its own cache line. Each thread updates a separate counter on a different cache line.

```cpp
struct alignas(64) PaddedCounter {
    volatile int value;
};
```

---

## Ensuring a Fair Benchmark

Both experiments perform **exactly the same work**:

- same number of threads
- same number of increments
- same final result (`NUM_THREADS × ITERATIONS`)

The program verifies correctness at runtime. The **only difference between the two experiments is memory layout**.

---

## Preventing Compiler Optimizations

Counters are declared `volatile` to prevent the compiler from collapsing the increment loop into a single addition. Without this, the compiler could transform:

```cpp
for (...) counter++;
// into:
counter += ITERATIONS;
```

Using `volatile` ensures the program performs **real memory writes**, allowing cache line contention to be observed.

---

## Configuration

| Constant      | Default value | Description                     |
|---------------|---------------|---------------------------------|
| `NUM_THREADS` | `4`           | Number of concurrent threads    |
| `ITERATIONS`  | `100 000 000` | Increment iterations per thread |

---

## Build & Run

```bash
mkdir build
cd build
cmake ..
make -j
./falseSharing
```
---

**Example output on a 4-core machine:** **5.27x faster**
```
False sharing time:     10003.93 ms
No false sharing time:  190.306 ms

Expected result:         400000000
False sharing result:    400000000
No false sharing result: 400000000

Results verified
```

Both experiments produce the same result, but the **false sharing case is significantly slower**.

---

## Why This Matters

False sharing can easily appear in real-world systems:

- lock-free queues
- per-thread counters
- statistics collectors
- logging systems
- high-frequency trading engines

In latency-sensitive systems, even small layout mistakes can produce **multi-fold performance degradation**. Understanding how CPU caches behave is critical when designing **high-performance multithreaded systems**.

---

## Key Takeaways

- CPUs operate on **cache lines**, not individual variables
- Writing to one variable can invalidate nearby variables on the same cache line
- Independent variables can still cause contention
- Proper alignment and padding can dramatically improve performance
- Memory layout is a critical component of low-latency system design

---

## License

MIT License