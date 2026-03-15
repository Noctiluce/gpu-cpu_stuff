#include "layers.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <limits>
#include <random>

static std::vector<float> kaimingUniform(int fanIn, int n, unsigned seed) {
    std::mt19937 rng(seed);
    float bound = std::sqrt(2.f / static_cast<float>(fanIn));
    std::uniform_real_distribution<float> dist(-bound, bound);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

//////////////////////////
//////////////////////////

void AdamState::step(std::vector<float>& params, const std::vector<float>& grads, float lr, float beta1, float beta2, float eps)
{
    ++t;
    float bc1 = 1.f - std::pow(beta1, (float)t);
    float bc2 = 1.f - std::pow(beta2, (float)t);
    for (int i = 0; i < (int)params.size(); ++i) {
        m[i] = beta1 * m[i] + (1.f - beta1) * grads[i];
        v[i] = beta2 * v[i] + (1.f - beta2) * grads[i] * grads[i];
        float m_hat = m[i] / bc1;
        float v_hat = v[i] / bc2;
        params[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

//////////////////////////
//////////////////////////

Conv2D::Conv2D(int in_ch, int out_ch, int k, unsigned seed)
    : in_channels(in_ch), out_channels(out_ch), kernel_size(k)
{
    int fan_in = in_ch * k * k;
    weights = kaimingUniform(fan_in, out_ch * in_ch * k * k, seed);
    bias.assign(out_ch, 0.f);
    dW.assign(weights.size(), 0.f);
    db.assign(bias.size(),    0.f);
    adam_W.init((int)weights.size());
    adam_b.init((int)bias.size());
}

float  Conv2D::w (int o,int i,int ky,int kx) const {
    return weights[o*(in_channels*kernel_size*kernel_size)+i*(kernel_size*kernel_size)+ky*kernel_size+kx];
}
float& Conv2D::w (int o,int i,int ky,int kx) {
    return weights[o*(in_channels*kernel_size*kernel_size)+i*(kernel_size*kernel_size)+ky*kernel_size+kx];
}
float  Conv2D::dw(int o,int i,int ky,int kx) const {
    return dW[o*(in_channels*kernel_size*kernel_size)+i*(kernel_size*kernel_size)+ky*kernel_size+kx];
}
float& Conv2D::dw(int o,int i,int ky,int kx) {
    return dW[o*(in_channels*kernel_size*kernel_size)+i*(kernel_size*kernel_size)+ky*kernel_size+kx];
}

static Tensor conv2d_compute(const Tensor& input, int in_channels, int out_channels, int kernel_size, const std::vector<float>& weights, const std::vector<float>& bias)
{
    int H = input.shape[1], W = input.shape[2];
    int Ho = H - kernel_size + 1, Wo = W - kernel_size + 1;
    Tensor output({out_channels, Ho, Wo});
    auto wIdx = [&](int o,int i,int ky,int kx){
        return weights[o*(in_channels*kernel_size*kernel_size)+i*(kernel_size*kernel_size)+ky*kernel_size+kx];
    };
    for (int o = 0; o < out_channels; ++o)
        for (int h = 0; h < Ho; ++h)
            for (int ww = 0; ww < Wo; ++ww) {
                float s = bias[o];
                for (int i = 0; i < in_channels; ++i)
                    for (int ky = 0; ky < kernel_size; ++ky)
                        for (int kx = 0; kx < kernel_size; ++kx)
                            s += input.at(i, h+ky, ww+kx) * wIdx(o,i,ky,kx);
                output.at(o, h, ww) = s;
            }
    return output;
}

Tensor Conv2D::forward(const Tensor& x) const {
    return conv2d_compute(x, in_channels, out_channels, kernel_size, weights, bias);
}

ForwardResult Conv2D::forwardTrain(const Tensor& x) {
    auto c = std::make_shared<Conv2DCache>();
    c->input = x;
    return { conv2d_compute(x, in_channels, out_channels, kernel_size, weights, bias), c };
}

Tensor Conv2D::backward(const Tensor& dOut, const Cache& i_cache) {
    const auto& c   = static_cast<const Conv2DCache&>(i_cache);
    const Tensor& x = c.input;
    int H = x.shape[1], W = x.shape[2];
    int Ho = dOut.shape[1], Wo = dOut.shape[2];

    Tensor dIn({in_channels, H, W}, 0.f);
    std::fill(dW.begin(), dW.end(), 0.f);
    std::fill(db.begin(), db.end(), 0.f);

    for (int o = 0; o < out_channels; ++o)
        for (int h = 0; h < Ho; ++h)
            for (int ww = 0; ww < Wo; ++ww) {
                float g = dOut.at(o, h, ww);
                db[o] += g;
                for (int i = 0; i < in_channels; ++i)
                    for (int ky = 0; ky < kernel_size; ++ky)
                        for (int kx = 0; kx < kernel_size; ++kx) {
                            dw(o,i,ky,kx)           += g * x.at(i, h+ky, ww+kx);
                            dIn.at(i, h+ky, ww+kx) += g * w(o,i,ky,kx);
                        }
            }
    return dIn;
}

void Conv2D::adamStep(float lr) {
    adam_W.step(weights, dW, lr);
    adam_b.step(bias,    db, lr);
}

std::string Conv2D::name() const {
    return "Conv2D(" + std::to_string(in_channels) + "->"
         + std::to_string(out_channels)
         + ", k=" + std::to_string(kernel_size) + ")";
}

//////////////////////////
//////////////////////////

Tensor ReLU::forward(const Tensor& x) const
{
    Tensor out = x;
    for (auto& v : out.data) v = std::max(0.f, v);
    return out;
}
ForwardResult ReLU::forwardTrain(const Tensor& x)
{
    auto c = std::make_shared<ReLUCache>();
    c->input = x;
    Tensor out = x;
    for (auto& v : out.data) v = std::max(0.f, v);
    return {out, c};
}
Tensor ReLU::backward(const Tensor& dOut, const Cache& i_cache)
{
    const auto& c = static_cast<const ReLUCache&>(i_cache);
    Tensor dIn = dOut;
    for (int i = 0; i < (int)dIn.data.size(); ++i)
        if (c.input.data[i] <= 0.f) dIn.data[i] = 0.f;
    return dIn;
}

//////////////////////////
//////////////////////////

static Tensor maxpool_compute(const Tensor& x, int ps) {
    int C = x.shape[0], H = x.shape[1], W = x.shape[2];
    int Ho = H/ps, Wo = W/ps;
    Tensor out({C, Ho, Wo});
    for (int c = 0; c < C; ++c)
        for (int h = 0; h < Ho; ++h)
            for (int w = 0; w < Wo; ++w) {
                float mx = -std::numeric_limits<float>::infinity();
                for (int ph = 0; ph < ps; ++ph)
                    for (int pw = 0; pw < ps; ++pw)
                        mx = std::max(mx, x.at(c, h*ps+ph, w*ps+pw));
                out.at(c,h,w) = mx;
            }
    return out;
}

Tensor MaxPool2D::forward (const Tensor& x) const { return maxpool_compute(x, pool_size); }
ForwardResult MaxPool2D::forwardTrain(const Tensor& x) {
    auto c = std::make_shared<MaxPoolCache>();
    c->input = x; c->pool_size = pool_size;
    return {maxpool_compute(x, pool_size), c};
}
Tensor MaxPool2D::backward(const Tensor& dIn, const Cache& i_cache) {
    const auto& c  = static_cast<const MaxPoolCache&>(i_cache);
    int ps = c.pool_size;
    int C  = c.input.shape[0], H = c.input.shape[1], W = c.input.shape[2];
    int Ho = H/ps, Wo = W/ps;
    Tensor dOut(c.input.shape, 0.f);
    for (int ch = 0; ch < C; ++ch)
        for (int h = 0; h < Ho; ++h)
            for (int w = 0; w < Wo; ++w) {
                float mx  = -std::numeric_limits<float>::infinity();
                int   mh  = h*ps, mww = w*ps;
                for (int ph = 0; ph < ps; ++ph)
                    for (int pw = 0; pw < ps; ++pw) {
                        float v = c.input.at(ch, h*ps+ph, w*ps+pw);
                        if (v > mx) { mx = v; mh = h*ps+ph; mww = w*ps+pw; }
                    }
                dOut.at(ch, mh, mww) += dIn.at(ch, h, w);
            }
    return dOut;
}
std::string MaxPool2D::name() const {
    return "MaxPool2D(" + std::to_string(pool_size) + "x" + std::to_string(pool_size) + ")";
}

//////////////////////////
//////////////////////////


Tensor Flatten::forward(const Tensor& x) const {
    Tensor o;
    o.shape = {x.num_elements()};
    o.data = x.data;
    return o;
}
ForwardResult Flatten::forwardTrain(const Tensor& x)
{
    auto c = std::make_shared<FlattenCache>();
    c->orig_shape = x.shape;
    Tensor o; o.shape = {x.num_elements()}; o.data = x.data;
    return {o, c};
}
Tensor Flatten::backward(const Tensor& dOut, const Cache& i_cache)
{
    const auto& c = static_cast<const FlattenCache&>(i_cache);
    Tensor dIn; dIn.shape = c.orig_shape; dIn.data = dOut.data;
    return dIn;
}

//////////////////////////
//////////////////////////

Linear::Linear(int in_f, int out_f, unsigned seed) : in_features(in_f), out_features(out_f)
{
    weights = kaimingUniform(in_f, out_f * in_f, seed);
    bias.assign(out_f, 0.f);
    dW.assign(weights.size(), 0.f);
    db.assign(bias.size(),    0.f);
    adam_W.init((int)weights.size());
    adam_b.init((int)bias.size());
}

Tensor Linear::forward(const Tensor& x) const
{
    Tensor out({out_features});
    for (int o = 0; o < out_features; ++o) {
        float s = bias[o];
        for (int i = 0; i < in_features; ++i)
            s += x.data[i] * weights[o * in_features + i];
        out.data[o] = s;
    }
    return out;
}
ForwardResult Linear::forwardTrain(const Tensor& x) {
    auto c = std::make_shared<LinearCache>();
    c->input = x;
    return {forward(x), c};
}
Tensor Linear::backward(const Tensor& dOut, const Cache& i_cache) {
    const auto& c = static_cast<const LinearCache&>(i_cache);
    Tensor dIn({in_features}, 0.f);
    std::fill(dW.begin(), dW.end(), 0.f);
    std::fill(db.begin(), db.end(), 0.f);
    for (int o = 0; o < out_features; ++o)
    {
        db[o] += dOut.data[o];
        for (int i = 0; i < in_features; ++i)
        {
            dW[o * in_features + i] += dOut.data[o] * c.input.data[i];
            dIn.data[i]            += dOut.data[o] * weights[o * in_features + i];
        }
    }
    return dIn;
}
void Linear::adamStep(float lr) {
    adam_W.step(weights, dW, lr);
    adam_b.step(bias,    db, lr);
}
std::string Linear::name() const {
    return "Linear(" + std::to_string(in_features) + "->"
         + std::to_string(out_features) + ")";
}

//////////////////////////
//////////////////////////

Tensor Softmax::forward(const Tensor& x) const
{
    Tensor out = x;
    float mx = *std::max_element(out.data.begin(), out.data.end());
    float s  = 0.f;
    for (auto& v : out.data) {
        v = std::exp(v - mx);
        s += v;
    }
    for (auto& v : out.data)
        v /= s;
    return out;
}
ForwardResult Softmax::forwardTrain(const Tensor& x) { return {forward(x), nullptr}; }
Tensor        Softmax::backward     (const Tensor& d, const Cache&) { return d; }

//////////////////////////
//////////////////////////

std::pair<float, Tensor> softmaxCrossEntropy(const Tensor& logits, int label)
{
    const float mx = *std::max_element(logits.data.begin(), logits.data.end());
    std::vector<float> p(logits.data.size());
    float s = 0.f;
    for (int i = 0; i < (int)p.size(); ++i)
    {
        p[i] = std::exp(logits.data[i] - mx);
        s += p[i];
    }
    for (auto& v : p)
        v /= s;

    float loss = -std::log(std::max(p[label], 1e-7f));

    Tensor dLogits({(int)p.size()});
    dLogits.data = p;
    dLogits.data[label] -= 1.f;
    return {loss, dLogits};
}

//////////////////////////
//////////////////////////

Tensor Sequential::forward(const Tensor& input) const
{
    Tensor result = input;
    for (const auto& l : layers)
        result = l->forward(result);
    return result;
}

Tensor Sequential::forwardInspect(const Tensor& input, std::vector<Tensor>& activations) const
{
    activations.clear();
    activations.reserve(layers.size());
    Tensor result = input;
    for (const auto& l : layers)
    {
        result = l->forward(result);
        activations.push_back(result);
    }
    return result;
}

std::pair<Tensor, std::vector<std::shared_ptr<Cache>>> Sequential::forwardTrain(const Tensor& input)
{
    Tensor result = input;
    std::vector<std::shared_ptr<Cache>> caches;
    caches.reserve(layers.size());
    for (auto& l : layers)
        {
        auto [out, cache] = l->forwardTrain(result);
        result = std::move(out);
        caches.emplace_back(std::move(cache));
    }
    return {result, caches};
}

void Sequential::backward(const Tensor& dLoss, const std::vector<std::shared_ptr<Cache>>& caches)
{
    Tensor grad = dLoss;
    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i)
    {
        if (caches[i])
            grad = layers[i]->backward(grad, *caches[i]);
    }
}

void Sequential::adamStep(float lr)
{
    for (auto& l : layers)
        l->adamStep(lr);
}

void Sequential::summary() const
{
    int total = 0;
    std::cout << "\n **** Network summary : \n";
    for (size_t i = 0; i < layers.size(); ++i)
    {
        int p = layers[i]->numParams();
        total += p;
        std::cout << "[" << std::setw(2) << i+1 << "] " << std::left << std::setw(28) << layers[i]->name() << std::right << std::setw(8) << p << " params\n";
    }
    std::cout << "parameter count == " << total << "\n\n";
}

void Sequential::saveWeights(const std::string& path) const
{
    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open file for writing: " + path);

    for (const auto& l : layers)
    {
        auto* cv = dynamic_cast<Conv2D*>(l.get());
        if (cv)
        {
            f.write(reinterpret_cast<const char*>(cv->weights.data()),cv->weights.size() * sizeof(float));
            f.write(reinterpret_cast<const char*>(cv->bias.data()),cv->bias.size() * sizeof(float));
        }
        auto* ln = dynamic_cast<Linear*>(l.get());
        if (ln)
        {
            f.write(reinterpret_cast<const char*>(ln->weights.data()),ln->weights.size() * sizeof(float));
            f.write(reinterpret_cast<const char*>(ln->bias.data()),ln->bias.size() * sizeof(float));
        }
    }
}

void Sequential::loadWeights(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open weights file: " + path);

    for (auto& l : layers)
    {
        auto* cv = dynamic_cast<Conv2D*>(l.get());
        if (cv)
        {
            f.read(reinterpret_cast<char*>(cv->weights.data()),cv->weights.size() * sizeof(float));
            f.read(reinterpret_cast<char*>(cv->bias.data()),cv->bias.size() * sizeof(float));
        }
        auto* ln = dynamic_cast<Linear*>(l.get());
        if (ln)
        {
            f.read(reinterpret_cast<char*>(ln->weights.data()),ln->weights.size() * sizeof(float));
            f.read(reinterpret_cast<char*>(ln->bias.data()),ln->bias.size() * sizeof(float));
        }
    }
}
