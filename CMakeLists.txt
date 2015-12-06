cmake_minimum_required(VERSION 2.8)

#include( GenericDebugger.cmake )

find_package( OpenCV REQUIRED )
include_directories(/home/jiashen/opencv/include)

# The following variable is visible even in subdirs
FILE(GLOB OpenCV_LIBS /home/jiashen/opencv/lib/*)

#SET(OpenCV_LIBS "${OpenCV_LIBS}")

add_subdirectory(samples)
