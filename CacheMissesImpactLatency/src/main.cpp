// benchmarks/sequential_vs_random.cpp
//sudo perf stat -e cache-references,cache-misses ./cacheMissesImpactLatency
//sudo perf stat ./cacheMissesImpactLatency
//sudo perf stat -e L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses ./cacheMissesImpactLatency
//
// Usage:
//   ./cacheMissesImpactLatency                         > run all benchmarks
//   ./cacheMissesImpactLatency --aos-soa               > run only AoS vs SoA
//   ./cacheMissesImpactLatency --seq-rand              > run only Sequential vs Random
//   ./cacheMissesImpactLatency --aos-soa --seq-rand    > run both (same as default)

#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <string>

template<typename F>
double benchmark(F&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

// ─── AoS vs SoA ─────────────────────────────────────────────────────────────
// Real-world HFT case: scan all orders to find those above a price threshold.
// AoS: loads all 8 fields per order just to compare price → 7/8 bytes wasted.
// SoA: iterates price[] only → cache lines are 100% useful data.

constexpr size_t N = 50'000'000;
constexpr double THRESHOLD = N * 0.01 * 0.5; // ~50% of orders pass the filter

struct Order {
    double price;      //  8 bytes
    double qty;        //  8 bytes
    double timestamp;  //  8 bytes
    double volume;     //  8 bytes
    double vwap;       //  8 bytes
    double bid;        //  8 bytes
    double ask;        //  8 bytes
    double spread;     //  8 bytes
    // 64 bytes == 1 cache line 9full)
};
static_assert(sizeof(Order) == 64, "Order must be exactly 64 bytes");

struct OrdersSOA {
    std::vector<double> price;
    std::vector<double> qty;
    std::vector<double> timestamp;
    std::vector<double> volume;
    std::vector<double> vwap;
    std::vector<double> bid;
    std::vector<double> ask;
    std::vector<double> spread;
};

void runAosSoa() {
    // Init AoS
    std::vector<Order> aosVector(N);
    for (size_t i = 0; i < N; i++) {
        aosVector[i].price     = static_cast<double>(i) * 0.01;
        aosVector[i].qty       = static_cast<double>(i) * 0.02;
        aosVector[i].timestamp = static_cast<double>(i);
        aosVector[i].volume    = static_cast<double>(i) * 100.0;
        aosVector[i].vwap      = static_cast<double>(i) * 0.015;
        aosVector[i].bid       = static_cast<double>(i) * 0.009;
        aosVector[i].ask       = static_cast<double>(i) * 0.011;
        aosVector[i].spread    = 0.002;
    }

    // Init SoA
    OrdersSOA soaOrders;
    soaOrders.price.resize(N);
    soaOrders.qty.resize(N);
    soaOrders.timestamp.resize(N);
    soaOrders.volume.resize(N);
    soaOrders.vwap.resize(N);
    soaOrders.bid.resize(N);
    soaOrders.ask.resize(N);
    soaOrders.spread.resize(N);
    for (size_t i = 0; i < N; i++) {
        soaOrders.price[i]     = static_cast<double>(i) * 0.01;
        soaOrders.qty[i]       = static_cast<double>(i) * 0.02;
        soaOrders.timestamp[i] = static_cast<double>(i);
        soaOrders.volume[i]    = static_cast<double>(i) * 100.0;
        soaOrders.vwap[i]      = static_cast<double>(i) * 0.015;
        soaOrders.bid[i]       = static_cast<double>(i) * 0.009;
        soaOrders.ask[i]       = static_cast<double>(i) * 0.011;
        soaOrders.spread[i]    = 0.002;
    }

    volatile size_t countAOS = 0;
    volatile size_t countSOA = 0;

    // Warm up
    for (size_t i = 0; i < 1000; i++) if (aosVector[i].price > THRESHOLD) countAOS++;
    for (size_t i = 0; i < 1000; i++) if (soaOrders.price[i] > THRESHOLD) countSOA++;

    // AoS: filter orders above threshold, loads full 64-byte order object per element
    auto aosTime = benchmark([&] {
        size_t c = 0;
        for (size_t i = 0; i < N; i++)
            if (aosVector[i].price > THRESHOLD) c++;
        countAOS = c;
    });

    // SoA: same filter, iterates only price[], 8 values per cache line
    auto soaTime = benchmark([&] {
        size_t c = 0;
        for (size_t i = 0; i < N; i++)
            if (soaOrders.price[i] > THRESHOLD) c++;
        countSOA = c;
    });

    std::cout << "=== AoS vs SoA (filter: price > threshold) ===\n";
    std::cout << (countAOS == countSOA ? "Same sum" : "Different sum") << "\n";
    std::cout << "Slower AoS: " << aosTime * 1000.0 << " ms\n";
    std::cout << "Faster SoA: " << soaTime * 1000.0 << " ms\n";
    std::cout << "Speedup:    " << aosTime / soaTime << "x\n\n";
}

void runSeqRand() {
    std::vector<int> data(N, 1);
    std::vector<size_t> indices(N);
    for (size_t i = 0; i < N; i++) indices[i] = i;
    std::shuffle(indices.begin(), indices.end(), std::mt19937{42});

    volatile long long sumSeq  = 0;
    volatile long long sumRand = 0;

    auto seqTime = benchmark([&] {
        for (size_t i = 0; i < N; i++)
            sumSeq += data[i];
    });

    auto randTime = benchmark([&] {
        for (size_t i = 0; i < N; i++)
            sumRand += data[indices[i]];
    });

    std::cout << "=== Sequential vs Random access ===\n";
    std::cout << (sumSeq == sumRand ? "Same sum" : "Different sum") << "\n";
    std::cout << "Slower Random:     " << randTime * 1000.0 << " ms\n";
    std::cout << "Faster Sequential: " << seqTime  * 1000.0 << " ms\n";
    std::cout << "Speedup:           " << randTime / seqTime << "x\n\n";
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "  (no args)    Run all benchmarks\n"
              << "  --aos-soa    Run AoS vs SoA benchmark\n"
              << "  --seq-rand   Run Sequential vs Random access benchmark\n";
}

int main(int argc, char* argv[]) {
    bool runAll    = (argc == 1);
    bool doAosSoa = false;
    bool doSeqRand = false;

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--aos-soa")
            doAosSoa = true;
        else if (arg == "--seq-rand")
            doSeqRand = true;
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (runAll || doAosSoa)  runAosSoa();
    if (runAll || doSeqRand) runSeqRand();

    return 0;
}