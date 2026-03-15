#pragma once
#include "tensor.h"
#include "layers.h"
#include "mnist_loader.h"   // ← manquait
#include <vector>
#include <string>
#include <cuda_runtime.h>



//////////////////////////
//////////////////////////

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while(0)



//////////////////////////
//////////////////////////

struct GPUWeights {
    float* d_weights = nullptr;
    float* d_bias    = nullptr;
    int    n_weights = 0;
    int    n_bias    = 0;

    GPUWeights() = default;
    GPUWeights(const std::vector<float>& w, const std::vector<float>& b);
    ~GPUWeights();

    // non-copyable
    GPUWeights(const GPUWeights&)            = delete;
    GPUWeights& operator=(const GPUWeights&) = delete;

    // movable
    GPUWeights(GPUWeights&&) noexcept;
    GPUWeights& operator=(GPUWeights&&) noexcept;
};



//////////////////////////
//////////////////////////

struct GPUSequential {
    // GPU weight blocks (one per Conv2D or Linear in the network)
    std::vector<GPUWeights> gpu_weights;

    // Network shape parameters (copied from CPU net)
    struct ConvParams  { int in_ch, out_ch, k, H_in, W_in; };
    struct LinearParams{ int in_f, out_f; };

    std::vector<ConvParams>   conv_params;
    std::vector<LinearParams> linear_params;

    // Upload weights from a trained CPU Sequential
    void upload_from(const Sequential& cpu_net);

    // Run inference on a batch of images (batch_size × 1 × 28 × 28)
    // Returns logits (batch_size × 10) on CPU
    std::vector<float> forward_batch(const std::vector<float>& images_flat,
                                     int batch_size) const;
};



//////////////////////////
//////////////////////////

void evaluate_gpu(const Sequential&               cpu_net,
                  const std::vector<MNISTSample>& test_data,
                  double                          cpu_elapsed_ms,
                  bool                            show_samples = true,
                  int                             num_to_show  = 5,
                  int                             batch_size   = 128);
