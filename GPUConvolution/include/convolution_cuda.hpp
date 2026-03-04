#pragma once
#include "image.hpp"
#include <vector>
#include "utils.hpp"

void conv_cuda(const Image& in, Image& out, const std::vector<float>& kernel, int K, Timer* t);

