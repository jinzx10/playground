add_library(util STATIC profiler_aggregator.cpp)

# Define which headers should be made available to other targets (like the main executable)
target_include_directories(util
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR} # Exposes headers in this directory (utility.h)
)
