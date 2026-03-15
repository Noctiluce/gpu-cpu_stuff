#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>


__global__ void linear_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float*       __restrict__ output,
    int B, int in_f, int out_f)
{
    int b = blockIdx.x;
    int o = blockIdx.y;

    if (b >= B || o >= out_f) return;

    float sum = bias[o];
    const float* x = input   + b * in_f;
    const float* w = weights + o * in_f;

    for (int i = 0; i < in_f; ++i)
        sum += x[i] * w[i];

    output[b * out_f + o] = sum;
}

extern "C" void launch_linear(
    const float* d_input,
    const float* d_weights,
    const float* d_bias,
    float*       d_output,
    int B, int in_f, int out_f)
{
    dim3 grid(B, out_f);
    linear_kernel<<<grid, 1>>>(d_input, d_weights, d_bias, d_output, B, in_f, out_f);
}

__global__ void softmax_kernel(float* data, int B, int n_classes)
{
    extern __shared__ float smem[];

    int b = blockIdx.x;
    int i = threadIdx.x;

    if (b >= B || i >= n_classes) return;

    float* x = data + b * n_classes;

    smem[i] = x[i];
    __syncthreads();

    if (i == 0) {
        float mx = smem[0];
        for (int j = 1; j < n_classes; ++j)
            if (smem[j] > mx) mx = smem[j];
        smem[n_classes] = mx;
    }
    __syncthreads();

    float mx = smem[n_classes];
    smem[i] = __expf(smem[i] - mx);
    __syncthreads();

    if (i == 0) {
        float s = 0.f;
        for (int j = 0; j < n_classes; ++j) s += smem[j];
        smem[n_classes] = s;
    }
    __syncthreads();

    x[i] = smem[i] / smem[n_classes];
}

extern "C" void launch_softmax(float* d_data, int B, int n_classes)
{
    size_t smem_size = (n_classes + 1) * sizeof(float);
    softmax_kernel<<<B, n_classes, smem_size>>>(d_data, B, n_classes);
}
