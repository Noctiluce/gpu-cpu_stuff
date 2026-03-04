cmake_minimum_required(VERSION 3.18)

project(gpu_convolution LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# RTX 2070 Super
set(CMAKE_CUDA_ARCHITECTURES 75)

option(NATIVE_OPT "Enable native CPU optimizations" ON)

if (MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -O3)
    if(NATIVE_OPT)
        add_compile_options(-march=native)
    endif()
endif()

set(SOURCES
        src/main.cpp
        src/image.cpp
        src/convolution_cpu.cpp
        src/convolution_cuda.cu
)

add_executable(gpu_convolution ${SOURCES})

target_include_directories(gpu_convolution PRIVATE
        include
        include/external
)

set_target_properties(gpu_convolution PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

target_compile_options(gpu_convolution PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        -lineinfo
        -Xptxas -v
        -Xptxas -dlcm=ca
        >
)
