cmake_minimum_required(VERSION 3.18)

project(mnist_cnn LANGUAGES CXX CUDA)
set(USE_CUDA ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Fix NVCC dependency bug (CMake + CUDA 12)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -MD")

# RTX 2070 Super
set(CMAKE_CUDA_ARCHITECTURES 75)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "CUDA enabled - GPU evaluation active")

set(COMMON_SOURCES
    src/layers.cpp
    src/mnist_loader.cpp
)

# mnist_cnn
set(CUDA_SOURCES
    cuda/conv_kernel.cu
    cuda/relu_kernel.cu
    cuda/pool_kernel.cu
    cuda/fc_kernel.cu
    cuda/gpu_layers.cu
)

add_executable(mnist_cnn
    src/main.cpp
    ${COMMON_SOURCES}
    ${CUDA_SOURCES}
)

target_compile_definitions(mnist_cnn PRIVATE USE_CUDA)

set_target_properties(mnist_cnn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(mnist_cnn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -MD
        -O3
        --use_fast_math
        -lineinfo
        -Xptxas=-v
        -Xptxas=-dlcm=ca
        -Xcompiler=-O3
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -march=native
    >
)

target_include_directories(mnist_cnn PRIVATE include)

# cnn_viewer
find_package(SFML 2.5 COMPONENTS graphics window system QUIET)

if(SFML_FOUND)
    message(STATUS "SFML found — building cnn_viewer")

    add_executable(cnn_viewer
        src/cnn_viewer.cpp
        ${COMMON_SOURCES}
    )

    target_include_directories(cnn_viewer PRIVATE include)

    target_link_libraries(cnn_viewer PRIVATE
        sfml-graphics
        sfml-window
        sfml-system
    )

    target_compile_options(cnn_viewer PRIVATE
        $<$<CONFIG:Release>:-O3 -march=native>
    )
else()
    message(STATUS "SFML not found — skipping cnn_viewer")
endif()





# -------------------------------------------------------------
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/data)

message(STATUS "Build type  : ${CMAKE_BUILD_TYPE}")
message(STATUS "MNIST data  : ${CMAKE_BINARY_DIR}/data/")
