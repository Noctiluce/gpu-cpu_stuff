#pragma once
#include <string>
#include <vector>

struct Image {
    int width;
    int height;
    int channels;
    std::vector<float> data;
};

Image load_image(const std::string& path);
void save_image(const std::string& path, const Image& img);



// access formulas
// Interleaved → data[pixel * channels + canal]
// Planar → data[canal * (width*height) + pixel]

// RGB RGB RGB  →  RRR GGG BBB
Image to_planar(const Image& src);

// RRR GGG BBB  →  RGB RGB RGB
Image to_interleaved(const Image& src);