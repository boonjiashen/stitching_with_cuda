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

#add_executable( feature_matching feature_matching.cpp )
#target_link_libraries( feature_matching ${OpenCV_LIBS} )

#add_subdirectory(tryMatrix)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=compute_20;-code=sm_20;-std=c++11)
Include(ParallelUtils.cmake)
enable_cuda_support()
cuda_add_executable(timeGPUFeatureMatcher.out timeGPUFeatureMatcher.cu
    gpuFeatureMatcher.cuh cpuFeatureMatcher.h Matrix.h)

# Otherwise we can't use C++11 features
set(CMAKE_CXX_FLAGS "-std=c++11")
