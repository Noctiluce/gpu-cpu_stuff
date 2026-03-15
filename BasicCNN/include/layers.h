#pragma once
#include "tensor.h"
#include <vector>
#include <string>
#include <memory>

struct Cache { virtual ~Cache() = default; };

struct ForwardResult {
    Tensor                 output;
    std::shared_ptr<Cache> cache;
};



//////////////////////////
//////////////////////////

struct AdamState {
    std::vector<float> m, v;
    int t = 0;

    void init(int n) { m.assign(n, 0.f); v.assign(n, 0.f); }

    void step(std::vector<float>& params,
              const std::vector<float>& grads,
              float lr     = 1e-3f,
              float beta1  = 0.9f,
              float beta2  = 0.999f,
              float eps    = 1e-8f);
};



//////////////////////////
//////////////////////////

struct Layer {
    virtual ~Layer() = default;

    virtual Tensor        forward      (const Tensor& input) const = 0;
    virtual ForwardResult forwardTrain(const Tensor& input)       = 0;
    virtual Tensor        backward     (const Tensor& d_out,
                                        const Cache&  cache)       = 0;
    virtual void          adamStep    (float lr)                  {}
    virtual std::string   name()       const = 0;
    virtual int           numParams() const { return 0; }
};



//////////////////////////
//////////////////////////
struct Conv2DCache : Cache { Tensor input; };

struct Conv2D : Layer {
    int in_channels, out_channels, kernel_size;
    std::vector<float> weights, bias, dW, db;
    AdamState          adam_W, adam_b;

    Conv2D(int in_ch, int out_ch, int k, unsigned seed = 42);

    Tensor        forward      (const Tensor& x) const override;
    ForwardResult forwardTrain(const Tensor& x)       override;
    Tensor        backward     (const Tensor& d, const Cache& c) override;
    void          adamStep    (float lr)                        override;
    std::string   name()       const override;
    int           numParams() const override { return (int)(weights.size()+bias.size()); }

private:
    inline float  w (int o,int i,int ky,int kx) const;
    inline float& w (int o,int i,int ky,int kx);
    inline float  dw(int o,int i,int ky,int kx) const;
    inline float& dw(int o,int i,int ky,int kx);
};



//////////////////////////
//////////////////////////
struct ReLUCache : Cache { Tensor input; };

struct ReLU : Layer {
    Tensor        forward      (const Tensor& x) const override;
    ForwardResult forwardTrain(const Tensor& x)       override;
    Tensor        backward     (const Tensor& d, const Cache& c) override;
    std::string   name()       const override { return "ReLU"; }
};



//////////////////////////
//////////////////////////
struct MaxPoolCache : Cache { Tensor input; int pool_size; };

struct MaxPool2D : Layer {
    int pool_size;
    explicit MaxPool2D(int ps = 2) : pool_size(ps) {}

    Tensor        forward      (const Tensor& x) const override;
    ForwardResult forwardTrain(const Tensor& x)       override;
    Tensor        backward     (const Tensor& d, const Cache& c) override;
    std::string   name()       const override;
};

//////////////////////////
//////////////////////////
struct FlattenCache : Cache { std::vector<int> orig_shape; };

struct Flatten : Layer {
    Tensor        forward      (const Tensor& x) const override;
    ForwardResult forwardTrain(const Tensor& x)       override;
    Tensor        backward     (const Tensor& d, const Cache& c) override;
    std::string   name()       const override { return "Flatten"; }
};



//////////////////////////
//////////////////////////
struct LinearCache : Cache { Tensor input; };

struct Linear : Layer {
    int in_features, out_features;
    std::vector<float> weights, bias, dW, db;
    AdamState          adam_W, adam_b;

    Linear(int in_f, int out_f, unsigned seed = 42);

    Tensor        forward      (const Tensor& x) const override;
    ForwardResult forwardTrain(const Tensor& x)       override;
    Tensor        backward     (const Tensor& d, const Cache& c) override;
    void          adamStep    (float lr)                        override;
    std::string   name()       const override;
    int           numParams() const override { return (int)(weights.size()+bias.size()); }
};



//////////////////////////
//////////////////////////
struct Softmax : Layer {
    Tensor        forward      (const Tensor& x) const override;
    ForwardResult forwardTrain(const Tensor& x)       override;
    Tensor        backward     (const Tensor& d, const Cache& c) override;
    std::string   name()       const override { return "Softmax"; }
};



//////////////////////////
//////////////////////////
struct Sequential {
    std::vector<std::unique_ptr<Layer>> layers;

    template<typename T, typename... Args>
    Sequential& add(Args&&... args) {
        layers.push_back(std::make_unique<T>(std::forward<Args>(args)...));
        return *this;
    }

    Tensor forward(const Tensor& input) const;

    // Forward that also captures intermediate activations (for visualisation).
    // activations[i] = output of layer i, in order.
    Tensor forwardInspect(const Tensor& input,
                           std::vector<Tensor>& activations) const;

    std::pair<Tensor, std::vector<std::shared_ptr<Cache>>>
    forwardTrain(const Tensor& input);

    void backward(const Tensor& dLoss,
                  const std::vector<std::shared_ptr<Cache>>& caches);

    void adamStep(float lr);
    void summary()  const;
    void saveWeights(const std::string& path) const;
    void loadWeights(const std::string& path);
};



//////////////////////////
//////////////////////////

std::pair<float, Tensor> softmaxCrossEntropy(const Tensor& logits, int label);
