cmake_minimum_required(VERSION 3.4)
set(OpenCV_DIR /usr/local/Cellar/opencv3/3.0.0/share/OpenCV)
find_package(OpenCV 3 REQUIRED)
add_executable(hello_world hello_world.cpp)
