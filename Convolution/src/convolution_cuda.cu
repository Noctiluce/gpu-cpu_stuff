#include "convolution_cuda.hpp"
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code!=cudaSuccess){
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " " << file << ":" << line << std::endl;
        if(abort) exit(code);
    }
}

__constant__ float d_kernel[49]; // max 7x7

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height,
    int channels,
    int K)
{
    const int R = K / 2;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;

    int x = bx + tx;
    int y = by + ty;

    int tileW = blockDim.x + 2 * R;
    int tileH = blockDim.y + 2 * R;

    extern __shared__ float tile[];

    // tile + halo
    for (int c = 0; c < channels; ++c)
    {
        for (int dy = ty; dy < tileH; dy += blockDim.y)
        {
            for (int dx = tx; dx < tileW; dx += blockDim.x)
            {
                int globalX = bx + dx - R;
                int globalY = by + dy - R;

                float value = 0.0f;

                if (globalX >= 0 && globalX < width &&
                    globalY >= 0 && globalY < height)
                {
                    value = input[(globalY * width + globalX) * channels + c];
                }

                tile[(dy * tileW + dx) * channels + c] = value;
            }
        }
    }

    __syncthreads();

    // conv
    if (x >= R && x < width - R &&
        y >= R && y < height - R)
    {
        for (int c = 0; c < channels; ++c)
        {
            float sum = 0.0f;

#pragma unroll
            for (int ky = -R; ky <= R; ++ky)
            {
#pragma unroll
                for (int kx = -R; kx <= R; ++kx)
                {
                    float v = tile[((ty + R + ky) * tileW + (tx + R + kx)) * channels + c];
                    float w = d_kernel[(ky + R) * K + (kx + R)];
                    sum += v * w;
                }
            }

            output[(y * width + x) * channels + c] = sum;
        }
    }
}

void conv_cuda(const Image& in, Image& out, const std::vector<float>& kernel, int K, Timer * t){
    int width = in.width;
    int height = in.height;
    int channels = in.channels;
    size_t img_size = width*height*channels*sizeof(float);

    static float* d_in = nullptr;
    static float* d_out = nullptr;
    if(!d_in) CUDA_CHECK(cudaMalloc(&d_in,img_size));
    if(!d_out) CUDA_CHECK(cudaMalloc(&d_out,img_size));

    CUDA_CHECK(cudaMemcpy(d_in,in.data.data(),img_size,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel,kernel.data(),K*K*sizeof(float)));

    dim3 block(16,16);
    dim3 grid((width+block.x-1)/block.x,(height+block.y-1)/block.y);
    size_t smem = (block.x+2*(K/2))*(block.y+2*(K/2))*channels*sizeof(float);

    t->reset();
    conv2d_kernel<<<grid,block,smem>>>(d_in,d_out,width,height,channels,K);
    auto timeKernel = t->elapsed_ms();
    std::cout << "kernel: " << timeKernel << " ms\n";

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out.data.data(),d_out,img_size,cudaMemcpyDeviceToHost));
    std::cout << "total : " <<  t->elapsed_ms() << " ms\n";

}
