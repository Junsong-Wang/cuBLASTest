cmake_minimum_required(VERSION 3.4.0)
project(match_test LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(CUDA REQUIRED)
find_package(cuBLAS REQUIRED)
find_package(OpenCV REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -Wall -Wno-unknown-pragmas -DSYS_LINUX -DVS_TRANMIST -pthread")
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75;-std=c++11;)

include_directories(src)
FILE(GLOB_RECURSE sourcefiles "src/*.cpp" "src/*.cu")

CUDA_ADD_EXECUTABLE(match ${sourcefiles})
MESSAGE(${OpenCV_LIBS})
target_link_libraries(match ${OpenCV_LIBS} cuda cublas cudart)
