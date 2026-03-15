#include "gpu_layers.h"
#include "mnist_loader.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstring>

extern "C" {
void launch_conv2d (const float*, const float*, const float*, float*,
                    int, int, int, int, int, int);
void launch_relu   (float*, int);
void launch_maxpool(const float*, float*, int, int, int, int, int);
void launch_linear (const float*, const float*, const float*, float*,
                    int, int, int);
void launch_softmax(float*, int, int);
}


GPUWeights::GPUWeights(const std::vector<float>& w, const std::vector<float>& b)
    : n_weights((int)w.size()), n_bias((int)b.size())
{
    CUDA_CHECK(cudaMalloc(&d_weights, n_weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias,    n_bias    * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_weights, w.data(), n_weights * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias,    b.data(), n_bias    * sizeof(float), cudaMemcpyHostToDevice));
}

GPUWeights::~GPUWeights() {
    if (d_weights) cudaFree(d_weights);
    if (d_bias)    cudaFree(d_bias);
}

GPUWeights::GPUWeights(GPUWeights&& o) noexcept
    : d_weights(o.d_weights), d_bias(o.d_bias)
    , n_weights(o.n_weights), n_bias(o.n_bias)
{
    o.d_weights = nullptr; o.d_bias = nullptr;
}

GPUWeights& GPUWeights::operator=(GPUWeights&& o) noexcept {
    if (this != &o) {
        if (d_weights) cudaFree(d_weights);
        if (d_bias)    cudaFree(d_bias);
        d_weights = o.d_weights; d_bias = o.d_bias;
        n_weights = o.n_weights; n_bias = o.n_bias;
        o.d_weights = nullptr;   o.d_bias = nullptr;
    }
    return *this;
}


void GPUSequential::upload_from(const Sequential& cpu_net) {
    gpu_weights.clear();
    conv_params.clear();
    linear_params.clear();

    int H = 28, W = 28;

    for (const auto& layer : cpu_net.layers) {
        if (auto* cv = dynamic_cast<const Conv2D*>(layer.get())) {
            conv_params.push_back({cv->in_channels, cv->out_channels,
                                   cv->kernel_size, H, W});
            gpu_weights.emplace_back(cv->weights, cv->bias);
            H = H - cv->kernel_size + 1;
            W = W - cv->kernel_size + 1;
        }
        else if (auto* mp = dynamic_cast<const MaxPool2D*>(layer.get())) {
            H /= mp->pool_size;
            W /= mp->pool_size;
        }
        else if (auto* ln = dynamic_cast<const Linear*>(layer.get())) {
            linear_params.push_back({ln->in_features, ln->out_features});
            gpu_weights.emplace_back(ln->weights, ln->bias);
        }
    }
}

std::vector<float> GPUSequential::forward_batch(
    const std::vector<float>& images_flat, int B) const
{
    // -- Upload images ----------------------------------------------------
    float* d_buf = nullptr;
    size_t img_bytes = images_flat.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_buf, img_bytes));
    CUDA_CHECK(cudaMemcpy(d_buf, images_flat.data(), img_bytes, cudaMemcpyHostToDevice));

    float* d_current = d_buf;

    int C = 1, H = 28, W = 28;

    int conv_idx   = 0;
    int linear_idx = 0;
    int w_idx = 0;


    auto do_conv = [&]() {
        const auto& p  = conv_params[conv_idx++];
        int Ho = p.H_in - p.k + 1;
        int Wo = p.W_in - p.k + 1;
        size_t out_bytes = (size_t)B * p.out_ch * Ho * Wo * sizeof(float);
        float* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, out_bytes));

        launch_conv2d(d_current, gpu_weights[w_idx].d_weights,
                      gpu_weights[w_idx].d_bias, d_out,
                      B, p.in_ch, p.H_in, p.W_in, p.out_ch, p.k);
        w_idx++;

        if (d_current != d_buf) cudaFree(d_current);
        else                    cudaFree(d_buf), d_buf = nullptr;
        d_current = d_out;
        C = p.out_ch; H = Ho; W = Wo;
    };

    auto do_relu = [&]() {
        launch_relu(d_current, B * C * H * W);
    };

    auto do_pool = [&](int ps) {
        int Ho = H / ps, Wo = W / ps;
        float* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, (size_t)B * C * Ho * Wo * sizeof(float)));
        launch_maxpool(d_current, d_out, B, C, H, W, ps);
        cudaFree(d_current);
        d_current = d_out;
        H = Ho; W = Wo;
    };

    auto do_linear = [&]() {
        const auto& p  = linear_params[linear_idx++];
        float* d_out   = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, (size_t)B * p.out_f * sizeof(float)));
        launch_linear(d_current, gpu_weights[w_idx].d_weights,
                      gpu_weights[w_idx].d_bias, d_out,
                      B, p.in_f, p.out_f);
        w_idx++;
        cudaFree(d_current);
        d_current = d_out;
        C = p.out_f; H = 1; W = 1;
    };

    do_conv();            
    do_relu();
    do_pool(2);           
    do_conv();            
    do_relu();
    do_pool(2);           
    do_linear();          
    do_relu();
    do_linear();          

    launch_softmax(d_current, B, 10);

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> result(B * 10);
    CUDA_CHECK(cudaMemcpy(result.data(), d_current,
                          B * 10 * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_current);

    return result;
}

void evaluate_gpu(const Sequential&               cpu_net,
                  const std::vector<MNISTSample>& test_data,
                  double                          cpu_elapsed_ms,
                  bool                            show_samples,
                  int                             num_to_show,
                  int                             batch_size)
{
    std::cout << "\n==================================================\n";
    std::cout <<   "              GPU Evaluation (batch=" << batch_size << ")\n";
    std::cout <<   "==================================================\n\n";

    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::cout << "  GPU: " << prop.name
              << "  (" << prop.multiProcessorCount << " SMs, "
              << prop.totalGlobalMem / (1024*1024) << " MB)\n\n";

    GPUSequential gpu_net;
    gpu_net.upload_from(cpu_net);

    int n = (int)test_data.size();

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    int correct = 0;
    std::vector<int> class_correct(10, 0), class_total(10, 0);
    std::vector<int> all_preds(n);

    CUDA_CHECK(cudaEventRecord(ev_start));

    for (int start = 0; start < n; start += batch_size) {
        int B = std::min(batch_size, n - start);

        std::vector<float> flat(B * 28 * 28);
        for (int b = 0; b < B; ++b) {
            const auto& img = test_data[start + b].image;
            std::copy(img.data.begin(), img.data.end(),
                      flat.begin() + b * 28 * 28);
        }

        auto probs = gpu_net.forward_batch(flat, B);

        for (int b = 0; b < B; ++b) {
            const float* p   = probs.data() + b * 10;
            int pred = (int)(std::max_element(p, p + 10) - p);
            int label = test_data[start + b].label;
            all_preds[start + b] = pred;
            class_total[label]++;
            if (pred == label) { ++correct; class_correct[label]++; }
        }
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float gpu_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop));
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    float acc = (float)correct / n * 100.f;
    double speedup = cpu_elapsed_ms / gpu_ms;

    std::cout << "==================================================\n";
    std::cout << "  Test accuracy     : " << std::fixed << std::setprecision(2)
              << acc << "%  (" << correct << "/" << n << ")\n\n";

    std::cout << "  *------------------------------------------*\n";
    std::cout << "  |  CPU inference : " << std::setw(8) << std::setprecision(2)
              << cpu_elapsed_ms    << " ms  ("
              << std::setprecision(3) << cpu_elapsed_ms / n << " ms/img)   |\n";
    std::cout << "  |  GPU inference : " << std::setw(8) << std::setprecision(2)
              << gpu_ms            << " ms  ("
              << std::setprecision(3) << gpu_ms / n << " ms/img)   |\n";
    std::cout << "  |  Speedup       :   " << std::setw(5) << std::setprecision(1)
              << speedup << "×" << "                        |\n";
    std::cout << "  *------------------------------------------*\n\n";

    std::cout << "  Per-class accuracy:\n";
    for (int d = 0; d < 10; ++d) {
        float ca = class_total[d] > 0
                   ? (float)class_correct[d] / class_total[d] * 100.f : 0.f;
        int bar = (int)(ca / 100.f * 20.f);
        std::cout << "  " << d << " |"
                  << std::string(bar, '#')
                  << std::string(20 - bar, ' ')
                  << "| " << std::setw(5) << std::setprecision(1) << ca << "%\n";
    }
    std::cout << "==================================================\n\n";
}
