cmake_minimum_required(VERSION 3.5)
project(scan_matching)

set(CMAKE_CXX_STANDARD 11)

find_package (Eigen3 REQUIRED NO_MODULE)

set(SOURCE_FILES main.cpp)

include_directories(${EIGEN3_INCLUDE_DIR})
add_executable(scan_matching ${SOURCE_FILES})
