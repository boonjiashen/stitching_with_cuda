cmake_minimum_required(VERSION 2.8)

# Otherwise we can't use C++11 features
#set(CMAKE_CXX_FLAGS "-std=c++11")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=compute_20;-code=sm_20;-std=c++11)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=compute_35;-code=sm_35,compute_35;-m=32)
Include(../ParallelUtils.cmake)
enable_cuda_support()
cuda_add_executable(tryMatrix.out tryMatrix.cu ../Matrix.h)

# Otherwise we can't use C++11 features
#set(CMAKE_CXX_FLAGS "-std=c++11")

#add_executable( feature_matching feature_matching.cpp )
#target_link_libraries( feature_matching ${OpenCV_LIBS} )

#add_subdirectory(testUnifiedMemory)
