#include "image.hpp"
#include "histo_cpu.hpp"
#include "histo_cuda.hpp"
#include "utils.hpp"
#include <iostream>
#include <filesystem>
#include <algorithm>


std::vector<int> histogram_cpu(const Image& img, int bins = 256) {
    std::vector<int> hist(bins, 0);

    for (float val : img.data) {
        int bin = static_cast<int>(val * (bins - 1));
        hist[bin]++;
    }

    return hist;
}


Image histogram_to_image(const std::vector<int>& hist, int img_width = 256, int img_height = 256) {
    Image img;
    img.width = img_width;
    img.height = img_height;
    img.channels = 1;
    img.data.resize(img_width * img_height, 1.0f);

    int max_val = *std::max_element(hist.begin(), hist.end());

    for (int x = 0; x < img_width; ++x) {
        int col_height = static_cast<int>((hist[x] / static_cast<float>(max_val)) * img_height);
        for (int y = 0; y < col_height; ++y) {
            int row = img_height - 1 - y;
            img.data[row * img_width + x] = 0.0f;
        }
    }

    return img;
}

Image combine_rgb_histograms(const std::vector<int>& hist_r,
                             const std::vector<int>& hist_g,
                             const std::vector<int>& hist_b,
                             int width = 256, int height = 256) {
    Image hist_img;
    hist_img.width = width;
    hist_img.height = height;
    hist_img.channels = 3;
    hist_img.data.resize(width * height * 3, 0.0f);

    int max_val = 1;
    for(int i = 0; i < width; ++i) {
        max_val = std::max(max_val, std::max(hist_r[i], std::max(hist_g[i], hist_b[i])));
    }

    for(int x = 0; x < width; ++x) {
        int r_height = static_cast<int>((hist_r[x] / (float)max_val) * (height-1));
        int g_height = static_cast<int>((hist_g[x] / (float)max_val) * (height-1));
        int b_height = static_cast<int>((hist_b[x] / (float)max_val) * (height-1));

        for(int y = 0; y < height; ++y) {
            int idx = (y * width + x) * 3;
            if(y >= height - r_height) hist_img.data[idx + 0] = 1.0f; // R
            if(y >= height - g_height) hist_img.data[idx + 1] = 1.0f; // G
            if(y >= height - b_height) hist_img.data[idx + 2] = 1.0f; // B
        }
    }

    return hist_img;
}

int main() {
    std::filesystem::path cwd = std::filesystem::current_path();
    Image imgSrc = load_image(cwd.string()+"/../data/crab_6k_RGB_saturated.png");

    const char* channels_name[3] = {"R", "G", "B"};
    std::vector<int> hist_cpu_channels[3];
    std::vector<int> hist_gpu_channels[3];

    for (int c = 0; c < 3; ++c) {
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Processing channel: " << channels_name[c] << std::endl;

        Image imgMono;
        imgMono.width = imgSrc.width;
        imgMono.height = imgSrc.height;
        imgMono.channels = 1;
        imgMono.data.resize(imgMono.width * imgMono.height);
        int total_pixels = imgSrc.width * imgSrc.height;
        for (int i = 0; i < total_pixels; ++i) {
            imgMono.data[i] = imgSrc.data[i * imgSrc.channels + c];
        }


        auto start = std::chrono::high_resolution_clock::now();
        hist_cpu_channels[c] = histogram_cpu(imgMono);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Temps CPU : "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                  << " µs" << std::endl;

        Image hist_img_cpu = histogram_to_image(hist_cpu_channels[c]);
        save_image(cwd.string()+"/../results/histogram_cpu_" + std::string(channels_name[c]) + ".png", hist_img_cpu);


        start = std::chrono::high_resolution_clock::now();
        hist_gpu_channels[c] = conv_cuda(imgMono);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "Temps GPU : "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << " ms" << std::endl;

        Image hist_img_gpu = histogram_to_image(hist_gpu_channels[c]);
        save_image(cwd.string()+"/../results/histogram_gpu_" + std::string(channels_name[c]) + ".png", hist_img_gpu);
    }


    Image hist_img_cpu_rgb = combine_rgb_histograms(hist_cpu_channels[0],
                                                    hist_cpu_channels[1],
                                                    hist_cpu_channels[2]);
    save_image(cwd.string()+"/../results/histogram_cpu_RGB.png", hist_img_cpu_rgb);


    Image hist_img_gpu_rgb = combine_rgb_histograms(hist_gpu_channels[0],
                                                    hist_gpu_channels[1],
                                                    hist_gpu_channels[2]);
    save_image(cwd.string()+"/../results/histogram_gpu_RGB.png", hist_img_gpu_rgb);

    std::cout << "All histograms generated successfully!" << std::endl;
    return 0;
}