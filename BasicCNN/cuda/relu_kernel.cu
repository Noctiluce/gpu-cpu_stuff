#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void relu_kernel(float* data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = data[i] > 0.f ? data[i] : 0.f;
}

extern "C" void launch_relu(float* d_data, int n)
{
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(d_data, n);
}
