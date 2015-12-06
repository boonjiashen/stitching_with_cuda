cmake_minimum_required(VERSION 2.8)

#include( GenericDebugger.cmake )

find_package( OpenCV REQUIRED )
include_directories(/home/jiashen/opencv/include)
FILE(GLOB OpenCV_LIBS /home/jiashen/opencv/lib/*)

#add_executable( DisplayImage DisplayImage.cpp )
#target_link_libraries( DisplayImage ${OpenCV_LIBS} )

#cmake_minimum_required(VERSION 2.8)
##set(OpenCV_DIR /usr/local/Cellar/opencv3/3.0.0/share/OpenCV)
#find_package(OpenCV REQUIRED)

#cmake_minimum_required(VERSION 2.8)
#include_directories(~/opencv/include)
#FILE(GLOB libs ~/opencv/lib/*)

##add_executable( DisplayImage DisplayImage.cpp )
##add_executable(hello_world hello_world.cpp)
##target_link_libraries( DisplayImage ${libs})
add_subdirectory(samples)

#add_executable(stitching stitching.cpp)
#target_link_libraries( stitching ${OpenCV_LIBS} )
##target_link_libraries( DisplayImage ${OpenCV_LIBS} )
##target_link_libraries( stitching ${libs})
