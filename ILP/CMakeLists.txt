cmake_minimum_required(VERSION 3.18)

project(ILP LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

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
)

add_executable(ILP ${SOURCES})

target_include_directories(ILP PRIVATE
        include
        include/external
)

