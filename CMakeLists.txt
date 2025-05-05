cmake_minimum_required(VERSION 3.30)
project(untitled1)

set(CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc")
find_package(OpenCV REQUIRED)


add_executable(untitled1 main.cpp)

target_link_libraries(untitled1 ${OpenCV_LIBS})
