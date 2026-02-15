#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "image.hpp"

Image load_image(const std::string& path) {
    int w,h,c;
    unsigned char* img = stbi_load(path.c_str(), &w, &h, &c, 0);

    Image out;
    out.width = w;
    out.height = h;
    out.channels = c;
    out.data.resize(w*h*c);

    for(int i=0;i<w*h*c;i++)
        out.data[i] = img[i]/255.0f;

    stbi_image_free(img);
    return out;
}

void save_image(const std::string& path, const Image& img) {
    std::vector<unsigned char> buffer(img.width*img.height*img.channels);
    for(int i=0;i<img.width*img.height*img.channels;i++)
        buffer[i] = static_cast<unsigned char>(std::min(1.0f,std::max(0.0f,img.data[i]))*255);

    if(img.channels==3)
        stbi_write_png(path.c_str(), img.width, img.height, 3, buffer.data(), img.width*3);
    else if(img.channels==1)
        stbi_write_png(path.c_str(), img.width, img.height, 1, buffer.data(), img.width);
}
