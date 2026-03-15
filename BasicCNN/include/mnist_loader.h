#pragma once
#include "tensor.h"
#include <vector>
#include <string>


struct MNISTSample {
    Tensor image;   // dim : [1, 28, 28], val : [0, 1]
    int    label;   // actualValue
};


std::vector<MNISTSample> load_mnist(
    const std::string& imagesPath,
    const std::string& labelsPath,
    int maxSamples = 10
);


void print_mnist_ascii(const MNISTSample& sample);
