cmake_minimum_required(VERSION 2.8)
project( pointCloud )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

FIND_PACKAGE(PCL 1.3 REQUIRED) 
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable( pointCloud pointCloud.cpp )
target_link_libraries( pointCloud ${OpenCV_LIBS} ${PCL_LIBRARIES})
