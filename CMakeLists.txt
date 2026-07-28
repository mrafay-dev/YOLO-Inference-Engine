cmake_minimum_required(VERSION 3.20)
project(YOLOv8nInference CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")

set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/lib64/cmake/onnxruntime ${CMAKE_PREFIX_PATH})
link_directories(${CMAKE_SOURCE_DIR}/lib64)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto")  

find_package(onnxruntime REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(src)




