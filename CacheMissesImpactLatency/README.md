# Cache Misses Latency benchmarks

C++ benchmarks illustrating the impact of **cache misses** on latency, with concrete HFT (High-Frequency Trading) use cases.

## Benchmarks

### AoS vs SoA   `--aos-soa`

Simulates a real filter over 50 million orders: find all orders whose price exceeds a threshold.

| Layout | Structure | Cache behavior |
|--------|-----------|----------------|
| **AoS** (Array of Structs) | `Order[]`   64 bytes/element (1 cache line) | Loads all 8 fields just to read `price` → 7/8 bytes wasted |
| **SoA** (Struct of Arrays) | `price[]`, `qty[]`, ... | Iterates only `price[]` → 8 useful values per cache line |

**Expected result**: SoA ~8x faster for this type of filter.

**Given result**:
```
AoS vs SoA :
Same sum
Slower AoS: 126.956 ms
Faster SoA: 15.7641 ms
Speedup:    8.0535x
```

---

### Sequential vs Random   `--seq-rand`

Compares sequential access over a 50M integer array against random access (indices shuffled via `std::shuffle`).

| Access | Cache behavior |
|--------|----------------|
| **Sequential** | Hardware prefetcher active → near-zero misses |
| **Random** | Near-systematic cache miss → DRAM latency (~100 ns/access) |


**Given result**:
```
Sequential vs Random access : 
Same sum
Slower Random:     380.472 ms
Faster Sequential: 94.8575 ms
Speedup:           4.01099x
```

---

## Build

```bash
mkdir build
cd build
cmake ..
make -j
./cacheMissesImpactLatency
```

## Usage

```bash
# Run all benchmarks
./cacheMissesImpactLatency

# AoS vs SoA only
./cacheMissesImpactLatency --aos-soa

# Sequential vs Random only
./cacheMissesImpactLatency --seq-rand

# Help
./cacheMissesImpactLatency --help
```

## Profiling with `perf`

```bash
# General statistics
sudo perf stat ./cacheMissesImpactLatency

# Cache references & misses (L1/LLC)
sudo perf stat -e cache-references,cache-misses ./cacheMissesImpactLatency

# L1 data cache detail
sudo perf stat -e L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses ./cacheMissesImpactLatency
```

## Key Concepts

**Cache line**: the minimum transfer unit between CPU and memory, typically **64 bytes**.
Loading a single `double` costs as much as loading 8 contiguous `double`s.

**Prefetcher**: the CPU predicts sequential access patterns and preloads cache lines ahead of time.
Random access defeats this mechanism → memory stalls.

**AoS vs SoA**: when an algorithm only accesses *a subset of fields*,
SoA maximizes useful data density per cache line and reduces cache pressure.
