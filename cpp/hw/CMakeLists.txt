cmake_minimum_required(VERSION 3.10)
project(hw DESCRIPTION "hello world!")

# Set language standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Options
option(ENABLE_PROFILING "Function call counter and time profiling" ON)

# Prerequisites
find_package(MPI REQUIRED COMPONENTS CXX)

# External dependencies
include(cmake/Dependencies.cmake)

# Compiler flags
if (ENABLE_PROFILING)
    add_compile_definitions(ENABLE_PROFILING)
endif()

# Add source directories
add_subdirectory(src)

