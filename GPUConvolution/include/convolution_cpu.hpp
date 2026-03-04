#pragma once
#include "image.hpp"
#include <vector>

void conv_cpu(const Image& in, Image& out, const std::vector<float>& kernel, int K);
std::vector<float> make_blur_kernel(int K);
std::vector<float> make_edge_kernel(int K);
std::vector<float> make_sharpen_kernel(int K);
std::vector<float> make_sharpen_7x7_kernel();