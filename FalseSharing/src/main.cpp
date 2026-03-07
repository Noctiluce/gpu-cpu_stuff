#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include <cassert>

constexpr size_t NUM_THREADS = 4;
constexpr size_t ITERATIONS = 100'000'000;

struct FalseSharingCounters {
    volatile int counters[NUM_THREADS];
};

struct alignas(64) PaddedCounter {
    volatile int value;
};

long long runFalseSharing() {
    FalseSharingCounters data{};

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            for (size_t i = 0; i < ITERATIONS; ++i) {
                data.counters[t]++;
            }
        });
    }

    for (auto& th : threads) th.join();

    auto end = std::chrono::high_resolution_clock::now();

    long long total = 0;
    for (auto c : data.counters)
        total += c;

    std::cout << "False sharing time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";

    return total;
}

long long runNoFalseSharing() {
    PaddedCounter counters[NUM_THREADS]{};

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            for (size_t i = 0; i < ITERATIONS; ++i) {
                counters[t].value++;
            }
        });
    }

    for (auto& th : threads) th.join();

    auto end = std::chrono::high_resolution_clock::now();

    long long total = 0;
    for (auto& c : counters)
        total += c.value;

    std::cout << "No false sharing time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";

    return total;
}

int main() {

    const auto expected = NUM_THREADS * ITERATIONS;

    auto result1 = runFalseSharing();
    auto result2 = runNoFalseSharing();

    std::cout << "\nExpected result: " << expected << "\n";
    std::cout << "False sharing result: " << result1 << "\n";
    std::cout << "No false sharing result: " << result2 << "\n";

    assert(result1 == expected);
    assert(result2 == expected);

    std::cout << "\nResults verified\n";
}