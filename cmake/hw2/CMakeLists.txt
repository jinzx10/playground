cmake_minimum_required(VERSION 2.8)
#project(hello_world2)

include_directories(include)

#set(SOURCES src/fact.cpp src/main.cpp)
file(GLOB SOURCES src/*.cpp)

add_executable(hw2 ${SOURCES})
