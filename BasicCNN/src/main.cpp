#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
#include <string>

#include "tensor.h"
#include "layers.h"
#include "mnist_loader.h"

#ifdef USE_CUDA
#include "gpu_layers.h"
#endif

using Clock = std::chrono::high_resolution_clock;

// ─────────────────────────────────────────────────────────────────────────────
//  Network definition
// ─────────────────────────────────────────────────────────────────────────────
Sequential build_cnn() {
    Sequential net;
    net.add<Conv2D>  (1,   8,  3, 1)
       .add<ReLU>    ()
       .add<MaxPool2D>(2)
       .add<Conv2D>  (8,  16,  3, 2)
       .add<ReLU>    ()
       .add<MaxPool2D>(2)
       .add<Flatten> ()
       .add<Linear>  (400, 128, 3)
       .add<ReLU>    ()
       .add<Linear>  (128,  10, 4);
    return net;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Progress bar
// ─────────────────────────────────────────────────────────────────────────────
static void print_progress(int current, int total, float loss, float acc) {
    int w = 30, filled = (int)((float)current / total * w);
    std::cout << "\r  [";
    for (int i = 0; i < w; ++i) std::cout << (i < filled ? '#' : ' ');
    std::cout << "] " << std::setw(5) << current << "/" << total
              << "  loss: " << std::fixed << std::setprecision(4) << loss
              << "  acc: "  << std::setprecision(1) << acc * 100.f << "%   "
              << std::flush;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Training
// ─────────────────────────────────────────────────────────────────────────────
struct TrainConfig { int epochs = 5; float lr = 1e-3f; int batch_log = 2000; };

void train(Sequential& net, std::vector<MNISTSample>& data, const TrainConfig& cfg)
{
    std::mt19937 rng(0);
    std::vector<int> idx(data.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::cout << "\n==================================================\n";
    std::cout <<   "                  Training (CPU)                  \n";
    std::cout <<   "==================================================\n";

    for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
        std::shuffle(idx.begin(), idx.end(), rng);
        float loss_sum = 0.f;
        int   correct  = 0;
        int   n        = (int)data.size();
        auto  t0       = Clock::now();

        for (int step = 0; step < n; ++step) {
            const auto& s         = data[idx[step]];
            auto [logits, caches] = net.forwardTrain(s.image);
            auto [loss, d_logits] = softmaxCrossEntropy(logits, s.label);
            loss_sum += loss;
            int pred = (int)(std::max_element(logits.data.begin(), logits.data.end()) - logits.data.begin());
            if (pred == s.label) ++correct;
            net.backward(d_logits, caches);
            net.adamStep(cfg.lr);
            if ((step + 1) % cfg.batch_log == 0 || step == n - 1)
                print_progress(step + 1, n, loss_sum / (step + 1), (float)correct / (step + 1));
        }

        double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
        std::cout << "\n  Epoch " << epoch << "/" << cfg.epochs << " - loss: " << std::fixed << std::setprecision(4) << loss_sum / n << "  train_acc: " << std::setprecision(1) << (float)correct / n * 100.f << "%" << "  (" << std::setprecision(1) << elapsed << "s)\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  CPU Evaluation  - returns elapsed ms
// ─────────────────────────────────────────────────────────────────────────────
double evaluate_cpu(const Sequential& net, const std::vector<MNISTSample>& test_data, bool show_samples = true, int num_to_show = 5)
{
    std::cout << "\n==================================================\n";
    std::cout <<   "              CPU Evaluation                       \n";
    std::cout <<   "==================================================\n\n";

    int correct = 0, n = (int)test_data.size();
    Softmax sm;
    std::vector<int> class_correct(10, 0), class_total(10, 0);

    auto t0 = Clock::now();
    for (int i = 0; i < n; ++i) {
        const auto& s = test_data[i];
        Tensor logits = net.forward(s.image);
        Tensor probs  = sm.forward(logits);
        int pred = (int)(std::max_element(probs.data.begin(), probs.data.end()) - probs.data.begin());
        class_total[s.label]++;
        if (pred == s.label) {
            ++correct;
            class_correct[s.label]++;
        }

        if (show_samples && i < num_to_show) {
            print_mnist_ascii(s);
            std::cout << "  Predicted : " << pred
                      << (pred == s.label ? "  good" : " bad  (true: " + std::to_string(s.label) + ")") << "\n";
            std::cout << "  Confidence: " << std::fixed << std::setprecision(1) << probs.data[pred] * 100.f << "%\n\n";
            for (int d = 0; d < 10; ++d) {
                int bar = (int)(probs.data[d] * 30.f);
                std::cout << "  " << d << " |" << std::string(bar, d == pred ? '#' : '-') << std::string(30 - bar, ' ') << "| " << std::setw(5) << std::setprecision(2) << probs.data[d] * 100.f << "%\n";
            }
            std::cout << "\n";
        }
    }

    double ms  = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    float  acc = (float)correct / n * 100.f;

    std::cout << "==================================================\n";
    std::cout << "Test accuracy : " << std::fixed << std::setprecision(2) << acc << "%  (" << correct << "/" << n << ")\n";
    std::cout << "Inference time: " << std::setprecision(3) << ms / n << " ms/image  (" << std::setprecision(1) << ms << " ms total)\n";
    std::cout << "==================================================\n\n";

    std::cout << "  Per-class accuracy (CPU):\n";
    for (int d = 0; d < 10; ++d) {
        float ca = class_total[d] > 0 ? (float)class_correct[d] / class_total[d] * 100.f : 0.f;
        int bar = (int)(ca / 100.f * 20.f);
        std::cout << "  " << d << " |" << std::string(bar, '#') << std::string(20 - bar, ' ') << "| " << std::setw(5) << std::setprecision(1) << ca << "%\n";
    }
    std::cout << "\n";
    return ms;
}

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
    int         epochs      = 5;
    float       lr          = 1e-3f;
    std::string weight_file = "weights.bin";
    bool        load_only   = true;

    if (argc >= 2) epochs      = std::stoi(argv[1]);
    if (argc >= 3) lr          = std::stof(argv[2]);
    if (argc >= 4) weight_file = argv[3];
    if (argc >= 5) load_only   = std::string(argv[4]) == "--eval";

    Sequential net = build_cnn();
    net.summary();

    // Load mnists files
    std::vector<MNISTSample> train_data, test_data;
    {
        try {
            std::cout << "Loading training data (60 000 images)...\n";
            train_data = load_mnist("../data/train-images-idx3-ubyte",
                                    "../data/train-labels-idx1-ubyte", 60000);
            std::cout << "Loading test data    (10 000 images)...\n";
            test_data  = load_mnist("../data/t10k-images-idx3-ubyte",
                                    "../data/t10k-labels-idx1-ubyte",  10000);
            std::cout << "done\n";
        }
        catch (const std::exception& e) {
            std::cerr << "\n[ERROR] " << e.what()
                      << "\nPlace MNIST IDX files in data/\n\n";
            return 1;
        }
    }

    // Load weights (if asked)
    if (load_only) {
        try {
            net.loadWeights(weight_file);
            std::cout << "Loaded weights from " << weight_file << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[WARN] " << e.what() << "\n";
        }
    }

    // train
    if (!load_only) {
        train(net, train_data, {epochs, lr, 2000});
        try {
            net.saveWeights(weight_file);
            std::cout << "\n  Weights saved to: " << weight_file << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[WARN] " << e.what() << "\n";
        }
    }

    // CPU eval
    double cpu_ms = evaluate_cpu(net, test_data, true, 0);

#ifdef USE_CUDA
    // GPU eval
    evaluate_gpu(net, test_data, cpu_ms, false, 0, 256);
#else
    std::cout << "[INFO] Compiled without CUDA - GPU evaluation skipped.\n\n";
#endif

    return 0;
}

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
