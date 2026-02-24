#include "image.hpp"
#include "convolution_cpu.hpp"
#include "convolution_cuda.hpp"
#include "utils.hpp"
#include <iostream>
#include <filesystem>

int main(){
    std::filesystem::path cwd = std::filesystem::current_path();

    std::vector<std::string> imageNameVec{"sponza_6k.png","sponza_4k.png","sponza_2k.png","sponza_1k.png","sponza_512.png","sponza_256.png"};
    for (auto imageName : imageNameVec)
    {
        std::cout <<"\n"<<  imageName << " : " << std::endl;
        Image img = load_image(cwd.string()+"/../data/"+imageName);
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

        save_image(cwd.string()+"/../results/cpu_"+imageName,cpu_out);
        save_image(cwd.string()+"/../results/gpu_"+imageName,gpu_out);
    }

}
