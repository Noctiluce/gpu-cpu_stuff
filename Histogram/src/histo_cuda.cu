#include "histo_cuda.hpp"
#include <cuda_runtime.h>
#include <iostream>

#define BINS 256

__global__ void histogram_kernel_naive(const float* img_data, int* hist, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    int bin = static_cast<int>(img_data[idx] * (BINS - 1));
    atomicAdd(&hist[bin], 1);
}

__global__ void histogram_kernel_shared(const float* img_data, int* hist, int size) {
    __shared__ int local_hist[BINS];

    if (threadIdx.x < BINS)
    local_hist[threadIdx.x] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        {
        float val = __ldg(&img_data[idx]);
        int bin = static_cast<int>(val * 255.0f);
        atomicAdd(&local_hist[bin], 1);
    }
    __syncthreads();

    if (threadIdx.x < BINS)
        atomicAdd(&hist[threadIdx.x], local_hist[threadIdx.x]);
}

std::vector<int> conv_cuda(const Image& in){
    int size = in.data.size();
    float* d_data;
    int* d_hist;
    cudaMalloc(&d_data, size * sizeof(float));
    cudaMalloc(&d_hist, BINS * sizeof(int));
    cudaMemset(d_hist, 0, BINS * sizeof(int));

    cudaMemcpy(d_data, in.data.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    //histogram_kernel_naive<<<blocks, threads>>>(d_data, d_hist, size);
    histogram_kernel_shared<<<blocks, threads>>>(d_data, d_hist, size);

    std::vector<int> hist(BINS);
    cudaMemcpy(hist.data(), d_hist, BINS * sizeof(int), cudaMemcpyDeviceToHost);


    cudaFree(d_data);
    cudaFree(d_hist);


    return hist;
}