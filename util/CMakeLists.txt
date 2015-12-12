cmake_minimum_required(VERSION 2.8)
find_package("CUDA" REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=compute_20;-code=sm_20;-std=c++11;-O3)

Include(ParallelUtils.cmake)
enable_cuda_support()
cuda_add_executable(convolution.out 2Dconvolution.cu 2Dconvolution.h CudaTimer.cuh)
