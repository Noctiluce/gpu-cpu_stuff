#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>



__global__ void maxpool_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int B, int C, int H_in, int W_in, int pool_size)
{
    int H_out = H_in / pool_size;
    int W_out = W_in / pool_size;

    int b  = blockIdx.x;
    int c  = blockIdx.y;
    int hw = blockIdx.z * blockDim.x + threadIdx.x;

    if (b >= B || c >= C || hw >= H_out * W_out) return;

    int h = hw / W_out;
    int w = hw % W_out;

    float mx = -FLT_MAX;
    for (int ph = 0; ph < pool_size; ++ph)
        for (int pw = 0; pw < pool_size; ++pw) {
            int in_idx = b * (C * H_in * W_in)
                       + c * (H_in * W_in)
                       + (h * pool_size + ph) * W_in
                       + (w * pool_size + pw);
            float v = input[in_idx];
            if (v > mx) mx = v;
        }

    int out_idx = b * (C * H_out * W_out)
                + c * (H_out * W_out)
                + h * W_out + w;
    output[out_idx] = mx;
}

extern "C" void launch_maxpool(
    const float* d_input,
    float*       d_output,
    int B, int C, int H_in, int W_in, int pool_size)
{
    int H_out    = H_in / pool_size;
    int W_out    = W_in / pool_size;
    int hw_total = H_out * W_out;

    int threads = min(hw_total, 256);
    int blocks_z = (hw_total + threads - 1) / threads;

    dim3 grid (B, C, blocks_z);
    dim3 block(threads);

    maxpool_kernel<<<grid, block>>>(d_input, d_output, B, C, H_in, W_in, pool_size);
}
