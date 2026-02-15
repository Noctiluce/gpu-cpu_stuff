#include "image.hpp"
#include "convolution_cpu.hpp"
#include "convolution_cuda.hpp"
#include "utils.hpp"
#include <iostream>
#include <filesystem>

int main(){
    std::filesystem::path cwd = std::filesystem::current_path();

    Image img = load_image(cwd.string()+"/../data/peppers.png");
    Image cpu_out = img;
    Image gpu_out = img;

    int K = 7;
    auto kernel = make_sharpen_7x7_kernel();

    Timer t;
    t.reset();
    conv_cpu(img,cpu_out,kernel,K);
    std::cout << "CPU Time: " << t.elapsed_ms() << " ms\n";

    t.reset();
    conv_cuda(img,gpu_out,kernel,K, &t);
    std::cout << "GPU Time: " << t.elapsed_ms() << " ms\n";

    const int width = img.width;
    const int height = img.height;
    const int channels = img.channels;

    for (int i = 0; i < width* height* channels; ++i) {
        if (abs(cpu_out.data[i] - gpu_out.data[i]) > 1e-5 ) // TODO speak of that
        {
            std::cout << "Results are not the same " << i << std::endl;
        }
    }

    save_image(cwd.string()+"/../results/cpu.png",cpu_out);
    save_image(cwd.string()+"/../results/gpu.png",gpu_out);
}
