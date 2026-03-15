#include <cuda_runtime.h>
#include <device_launch_parameters.h>



__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float*       __restrict__ output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int ksize)
{
    int H_out = H_in - ksize + 1;
    int W_out = W_in - ksize + 1;

    int b    = blockIdx.x;
    int o    = blockIdx.y;
    int hw   = blockIdx.z * blockDim.x + threadIdx.x;

    if (b >= B || o >= C_out || hw >= H_out * W_out) return;

    int h = hw / W_out;
    int w = hw % W_out;

    float sum = bias[o];

    for (int ci = 0; ci < C_in; ++ci)
        for (int ky = 0; ky < ksize; ++ky)
            for (int kx = 0; kx < ksize; ++kx) {
                int in_idx = b  * (C_in  * H_in  * W_in)
                           + ci * (H_in  * W_in)
                           + (h + ky) * W_in + (w + kx);

                int w_idx  = o  * (C_in  * ksize * ksize)
                           + ci * (ksize * ksize)
                           + ky * ksize + kx;

                sum += input[in_idx] * weights[w_idx];
            }

    int out_idx = b  * (C_out * H_out * W_out)
                + o  * (H_out * W_out)
                + h  * W_out + w;
    output[out_idx] = sum;
}

extern "C" void launch_conv2d(
    const float* d_input,
    const float* d_weights,
    const float* d_bias,
    float*       d_output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int ksize)
{
    int H_out   = H_in - ksize + 1;
    int W_out   = W_in - ksize + 1;
    int hw_total = H_out * W_out;

    int threads = min(hw_total, 256);
    int blocks_z = (hw_total + threads - 1) / threads;

    dim3 grid (B, C_out, blocks_z);
    dim3 block(threads);

    conv2d_kernel<<<grid, block>>>(
        d_input, d_weights, d_bias, d_output,
        B, C_in, H_in, W_in, C_out, ksize);
}
