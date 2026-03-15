#pragma once
#include <vector>
#include <cassert>
#include <numeric>
#include <stdexcept>


struct Tensor {
    std::vector<int>   shape;
    std::vector<float> data;

    Tensor() = default;

    explicit Tensor(std::vector<int> shape_, float fill = 0.f)
        : shape(std::move(shape_))
        , data(num_elements(), fill)
    {}

    int num_elements() const {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>{});
    }

    float& at(int i)             { return data[i]; }
    float  at(int i) const       { return data[i]; }

    float& at(int c, int h, int w) {
        assert(shape.size() == 3);
        return data[c * shape[1] * shape[2] + h * shape[2] + w];
    }
    float at(int c, int h, int w) const {
        assert(shape.size() == 3);
        return data[c * shape[1] * shape[2] + h * shape[2] + w];
    }

    float& at1(int i)       { assert(shape.size() == 1); return data[i]; }
    float  at1(int i) const { assert(shape.size() == 1); return data[i]; }
};
