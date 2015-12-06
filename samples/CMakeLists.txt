cmake_minimum_required(VERSION 2.8)

find_package( OpenCV REQUIRED )
include_directories(/home/jiashen/opencv/include)
FILE(GLOB OpenCV_LIBS /home/jiashen/opencv/lib/*)

add_executable( DisplayImage DisplayImage.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )

add_executable( stitching stitching.cpp )
target_link_libraries( stitching ${OpenCV_LIBS} )
