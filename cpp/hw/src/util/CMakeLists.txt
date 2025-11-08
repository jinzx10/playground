add_library(util STATIC
    profiler_aggregator.cpp
    log.cpp
    toml.cpp
)

target_include_directories(util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(util PUBLIC spdlog::spdlog MPI::MPI_CXX tomlplusplus::tomlplusplus)
