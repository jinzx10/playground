cmake_minimum_required(VERSION 3.10)
project(hw DESCRIPTION "hello world!")

# Set language standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(ENABLE_PROFILING "Function call counter and time profiling" ON)

# Add the main application executable target
add_executable(${PROJECT_NAME} src/main.cpp)


add_subdirectory(src/util)

# Link the main executable against the libraries defined in the subdirectories
target_link_libraries(${PROJECT_NAME} PRIVATE util)


if (ENABLE_PROFILING)
    add_compile_definitions(ENABLE_PROFILING)
endif()

## --- UNIT TEST BUILD ---
## 3. CONFIGURE TESTING (Using Google Test)
#
## Search for Google Test package
## NOTE: You must install GTest globally or include it as a submodule/external project
#find_package(GTest REQUIRED)
#
## Enable testing for CTest (CMake's built-in test runner)
#enable_testing()
#
## Define the unit test executable target
#add_executable(unit_tests
#    ${TEST_SOURCE_DIR}/unit/*.cpp
#    ${TEST_SOURCE_DIR}/unit/*/*.cpp # Include unit test files
#)
#
## Link the unit test executable to the necessary libraries
#target_link_libraries(unit_tests
#    GTest::gtest_main  # Links GTest framework and its main() function
#    ${PROJECT_NAME}    # Links the main application's code (if needed)
#)
#
## 4. REGISTER TESTS WITH CTEST
#
## This command tells CTest how to run the test executable
#add_test(NAME run_unit_tests COMMAND unit_tests)
