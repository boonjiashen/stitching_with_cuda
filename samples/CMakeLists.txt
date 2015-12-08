cmake_minimum_required(VERSION 2.8)

#MESSAGE( STATUS "OpenCV_LIBS:         " ${OpenCV_LIBS} )

#add_executable( DisplayImage DisplayImage.cpp )
#target_link_libraries( DisplayImage ${OpenCV_LIBS} )

add_executable( stitching stitching.cpp )
target_link_libraries( stitching ${OpenCV_LIBS} )

add_executable( stitching_detailed stitching_detailed.cpp )
target_link_libraries( stitching_detailed ${OpenCV_LIBS} )
