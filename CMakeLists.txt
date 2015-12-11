cmake_minimum_required(VERSION 2.8)
set(OpenCV_DIR "/usr/local/Cellar/opencv3/3.0.0/share/OpenCV")

#include( GenericDebugger.cmake )

find_package( OpenCV REQUIRED )

#include_directories(/home/jiashen/opencv/include)
#FILE(GLOB OpenCV_LIBS /home/jiashen/opencv/lib/*) # The following variable is visible even in subdirs

include_directories(/usr/local/Cellar/opencv3/3.0.0/include)
file(GLOB OpenCV_LIBS /usr/local/Cellar/opencv3/3.0.0/lib/*) # The following variable is visible even in subdirs

#SET(OpenCV_LIBS "${OpenCV_LIBS}")

add_executable( feature_matching feature_matching.cpp )
target_link_libraries( feature_matching ${OpenCV_LIBS} )
