cmake_minimum_required(VERSION 3.18)
project(Cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(CUDAToolkit REQUIRED)

# Shared include/flags/libs for every sample under dev/
add_library(cuda_dev_common INTERFACE)
target_include_directories(cuda_dev_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/dev
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_options(cuda_dev_common INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        -lineinfo
    >
)
target_link_libraries(cuda_dev_common INTERFACE
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
)

file(GLOB CUDA_DEV_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/dev/*.cu
)

foreach(src ${CUDA_DEV_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE cuda_dev_common)
endforeach()
