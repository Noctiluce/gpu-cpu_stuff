#include "mnist_loader.h"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <string>
#include <cstdint>

static int32_t read_be_int32(std::ifstream& f) {
    uint8_t b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return (int32_t)((b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]);
}

std::vector<MNISTSample> load_mnist(
    const std::string& imagesPath,
    const std::string& labelsPath,
    int maxSamples)
{
    std::ifstream img_f(imagesPath, std::ios::binary);
    if (!img_f)
        throw std::runtime_error("Cannot open images file: " + imagesPath);

    std::ifstream lbl_f(labelsPath, std::ios::binary);
    if (!lbl_f)
        throw std::runtime_error("Cannot open labels file: " + labelsPath);

    int32_t img_magic = read_be_int32(img_f);
    if (img_magic != 0x00000803)
        throw std::runtime_error("Bad magic number in images file");

    int32_t num_images = read_be_int32(img_f);
    int32_t rows       = read_be_int32(img_f);
    int32_t cols       = read_be_int32(img_f);

    int32_t lbl_magic = read_be_int32(lbl_f);
    if (lbl_magic != 0x00000801)
        throw std::runtime_error("Bad magic number in labels file");

    int32_t num_labels = read_be_int32(lbl_f);

    if (num_images != num_labels)
        throw std::runtime_error("Image / label count mismatch");

    int n = std::min((int32_t)maxSamples, num_images);
    int pixels = rows * cols;

    std::vector<MNISTSample> samples;
    samples.reserve(n);

    for (int s = 0; s < n; ++s) {

        std::vector<uint8_t> raw(pixels);
        img_f.read(reinterpret_cast<char*>(raw.data()), pixels);

        Tensor img({1, rows, cols});
        for (int i = 0; i < pixels; ++i)
            img.data[i] = raw[i] / 255.f;


        uint8_t lbl;
        lbl_f.read(reinterpret_cast<char*>(&lbl), 1);

        MNISTSample sample;
        sample.image = std::move(img);
        sample.label = (int)lbl;
        samples.push_back(std::move(sample));
    }

    return samples;
}

void print_mnist_ascii(const MNISTSample& sample) {
    const int H = sample.image.shape[1];
    const int W = sample.image.shape[2];

    std::cout << "\n  True label: " << sample.label << "\n\n";
    std::cout << "  +" << std::string(W, '-') << "+\n";

    for (int h = 0; h < H; ++h) {
        std::cout << "  |";
        for (int w = 0; w < W; ++w) {
            float v = sample.image.at(0, h, w);
            if      (v > 0.75f) std::cout << '#';
            else if (v > 0.40f) std::cout << '+';
            else if (v > 0.15f) std::cout << '.';
            else                std::cout << ' ';
        }
        std::cout << "|\n";
    }
    std::cout << "  +" << std::string(W, '-') << "+\n\n";
}
