cmake_minimum_required(VERSION 2.8)

#IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
#set(OpenCV_DIR "/usr/local/Cellar/opencv3/3.0.0/share/OpenCV")

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
    include_directories(/home/jiashen/opencv/include)
    FILE(GLOB OpenCV_LIBS /home/jiashen/opencv/lib/*) # The following variable is visible even in subdirs
endif()

#include( GenericDebugger.cmake )

find_package( OpenCV REQUIRED )

#include_directories(/home/jiashen/opencv/include)
#FILE(GLOB OpenCV_LIBS /home/jiashen/opencv/lib/*) # The following variable is visible even in subdirs

#SET(OpenCV_LIBS "${OpenCV_LIBS}")

add_executable( feature_matching feature_matching.cpp )
target_link_libraries( feature_matching ${OpenCV_LIBS} )
