cmake_minimum_required(VERSION 2.8)

MESSAGE( STATUS "Apple ?             " ${APPLE} )
MESSAGE( STATUS "Unix ?              " ${UNIX} )
MESSAGE( STATUS "CMAKE_SYSTEM_NAME   " ${CMAKE_SYSTEM_NAME})

# Source: http://www.openguru.com/2009/04/cmake-detecting-platformoperating.html
# Darwin corresponds to Mac OS X
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(OpenCV_DIR "/usr/local/Cellar/opencv3/3.0.0/share/OpenCV")
    include_directories(/usr/local/Cellar/opencv3/3.0.0/include)
    file(GLOB OpenCV_LIBS /usr/local/Cellar/opencv3/3.0.0/lib/*) # The following variable is visible even in subdirs

# For Euler
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    find_package( OpenCV REQUIRED )  # This has to come before the following two definitions
    include_directories(/home/jiashen/opencv/include)
    FILE(GLOB OpenCV_LIBS /home/jiashen/opencv/lib/*) # The following variable is visible even in subdirs
endif()

#include( GenericDebugger.cmake )

# Otherwise we can't use C++11 features
#set(CMAKE_CXX_FLAGS "-std=c++11")

#add_executable( feature_matching feature_matching.cpp )
#target_link_libraries( feature_matching ${OpenCV_LIBS} )

#add_subdirectory(testUnifiedMemory)
#add_subdirectory(testMatrix)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=compute_20;-code=sm_20;-std=c++11)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=compute_35;-code=sm_35,compute_35;-m=32)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
Include(ParallelUtils.cmake)
enable_cuda_support()
cuda_add_executable(tryCPUFeatureMatcher.out tryCPUFeatureMatcher.cu cpuFeatureMatcher.h Matrix.h)
#cuda_add_executable(gpuFeatureMatcher.out gpuFeatureMatcher.cu
    #cpuFeatureMatcher.h Matrix.h)

# Otherwise we can't use C++11 features
set(CMAKE_CXX_FLAGS "-std=c++11")

#add_executable( feature_matching feature_matching.cpp )
#target_link_libraries( feature_matching ${OpenCV_LIBS} )

#add_subdirectory(testUnifiedMemory)
#add_subdirectory(tryMemoryAllocationError)
