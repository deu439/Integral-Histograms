cmake_minimum_required(VERSION 2.6)
project(integralhistogram)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(hist main.cpp)
add_executable(detection detection.cpp)
target_link_libraries( hist ${OpenCV_LIBS})
target_link_libraries( detection ${OpenCV_LIBS})
