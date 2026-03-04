#pragma once
#include <chrono>

struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    void reset(){ start = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms(){
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double,std::milli>(end-start).count();
    }
};
