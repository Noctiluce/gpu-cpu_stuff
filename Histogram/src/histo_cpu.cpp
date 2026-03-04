#include "histo_cpu.hpp"
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