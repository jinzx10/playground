add_library(para STATIC
    parallel_config.cpp
)

target_include_directories(para PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(para PUBLIC MPI::MPI_CXX)
