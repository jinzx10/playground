#ifndef FUNCTION_PROFILER_H
#define FUNCTION_PROFILER_H

#ifdef ENABLE_PROFILING

#include <string>
#include <chrono>
#include "profiler_aggregator.h"

class FunctionProfiler {
private:
    std::string name_;
    std::chrono::steady_clock::time_point start_time_;

public:
    FunctionProfiler(const std::string& function_name) 
        : name_(function_name), start_time_(std::chrono::steady_clock::now()) {}

    ~FunctionProfiler() {
        auto duration = std::chrono::steady_clock::now() - start_time_;
        ProfilerAggregator::log(name_, duration);
    }
};

// A convenience macro to simplify insertion into functions
#define PROFILE_FUNCTION() FunctionProfiler function_profiler(__func__)

#else

#define PROFILE_FUNCTION() void(0)

#endif

#endif
