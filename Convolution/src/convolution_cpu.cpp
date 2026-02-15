#include "convolution_cpu.hpp"
#include <algorithm>
#include <cstring>

void conv_cpu(const Image& in, Image& out, const std::vector<float>& k, int K) {
    const int R = K / 2;
    const int width = in.width;
    const int height = in.height;
    const int channels = in.channels;

    std::memset(out.data.data(), 0, width * height * channels * sizeof(float));

    for(int y = R; y < height - R; y++) {
        for(int x = R; x < width - R; x++) {
            const int out_idx_base = (y * width + x) * channels;

            for(int c = 0; c < channels; c++) {
                float sum = 0.f;
                for(int ky = -R; ky <= R; ky++) {
                    const int iy = y + ky;
                    const int in_row_offset = iy * width * channels;
                    const int k_row_offset = (ky + R) * K;
                    for(int kx = -R; kx <= R; kx++) {
                        const int ix = x + kx;
                        const float v = in.data[in_row_offset + ix * channels + c];
                        const float w = k[k_row_offset + (kx + R)];
                        sum += v * w;
                    }
                }

                out.data[out_idx_base + c] = sum ;
            }
        }
    }
}

std::vector<float> make_blur_kernel(int K) {
    const float weight = 1.0f / (K * K);
    return std::vector<float>(K * K, weight);
}

std::vector<float> make_edge_kernel(int K) {
    std::vector<float> k(K * K, -1.0f);
    const int center = (K / 2) * K + K / 2;
    k[center] = static_cast<float>(K * K - 1);
    return k;
}

std::vector<float> make_sharpen_kernel(int K) {
    std::vector<float> k(K * K, -1.0f);
    const int center = (K / 2) * K + K / 2;
    k[center] = static_cast<float>(K * K);
    return k;
}

std::vector<float> make_sharpen_7x7_kernel() {
    return {
        -0.125f, -0.125f, -0.125f, -0.125f, -0.125f, -0.125f, -0.125f,
        -0.125f,  0.0f,    0.0f,    0.0f,    0.0f,    0.0f,   -0.125f,
        -0.125f,  0.0f,    0.25f,   0.25f,   0.25f,   0.0f,   -0.125f,
        -0.125f,  0.0f,    0.25f,   1.2f,    0.25f,   0.0f,   -0.125f,
        -0.125f,  0.0f,    0.25f,   0.25f,   0.25f,   0.0f,   -0.125f,
        -0.125f,  0.0f,    0.0f,    0.0f,    0.0f,    0.0f,   -0.125f,
        -0.125f, -0.125f, -0.125f, -0.125f, -0.125f, -0.125f, -0.125f
    };
}
