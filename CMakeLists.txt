cmake_minimum_required(VERSION 2.6)
project(integralhistogram)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(hist main.cpp)
target_link_libraries( hist ${OpenCV_LIBS})
