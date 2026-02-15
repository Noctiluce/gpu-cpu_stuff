cmake_minimum_required(VERSION 3.18)

project(gpu_convolution LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


set(CMAKE_CUDA_ARCHITECTURES native)

if (MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -O3 -march=native)
endif()

set(SOURCES
        src/main.cpp
        src/image.cpp
        src/convolution_cpu.cpp
        src/convolution_cuda.cu
)

include_directories(
        include
        include/external  # pour stb_image
)
add_executable(gpu_convolution ${SOURCES})

set_target_properties(gpu_convolution PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

if (NOT MSVC)
    target_compile_options(gpu_convolution PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math
            -Xptxas -v
            >)
endif()
