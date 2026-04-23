// Compile : g++ -O3 -o bench main.cpp
// Run     : ./bench

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <span>

constexpr int N = 100'000'000;
constexpr int RUNS = 5;
using ms = std::chrono::duration<double, std::milli>;

double sum_naive(std::span<const double> a)
{
    double result = 0.0;
    for (int i = 0; i < a.size(); ++i)
        result += a[i];   //
    return result;
}

double sum_ilp4(std::span<const double> a)
{
    double r0 = 0.0, r1 = 0.0, r2 = 0.0, r3 = 0.0;
    for (int i = 0; i < a.size(); i += 4) {
        r0 += a[i];
        r1 += a[i + 1];
        r2 += a[i + 2];
        r3 += a[i + 3];
    }
    return r0 + r1 + r2 + r3;
}


template<typename Fn>
double bench(const char* label, Fn fn, std::span<const double> data, int runs)
{
    double result = 0.0;
    double best_ms = 1e18;

    for (int r = 0; r < runs; ++r) {
        auto start = std::chrono::high_resolution_clock::now();
        result = fn(data);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = ms(end - start).count();
        if (elapsed < best_ms) best_ms = elapsed;
    }

    double ns_per_elem = (best_ms * 1e6) / data.size();
    std::printf("  %-20s  best=%7.1f ms   %.3f ns/elem   result=%.2f\n",
                label, best_ms, ns_per_elem, result);
    return best_ms;
}

int main()
{
    // Allocation et initialisation du tableau
    std::vector<double> data(N, 1);

    std::puts("Dependency Chain Benchmark");
    std::printf("  addition of %d doubles\n", N);
    std::printf("  best out of %d runs\n\n", RUNS);

    double t_slow = bench("sum_naive (slow)", sum_naive,  data,  RUNS);
    double t_fast = bench("sum_ilp4  (fast)", sum_ilp4,   data,  RUNS);

    std::puts("");
    std::printf("  Speedup : %.2fx\n", t_slow / t_fast);
    return 0;
}