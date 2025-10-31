#ifndef PROFILER_AGGREGATOR_H
#define PROFILER_AGGREGATOR_H

#include <chrono>
#include <map>
#include <mutex>
#include <string>

class ProfilerAggregator {
private: // singleton
    ProfilerAggregator() = default;
    ~ProfilerAggregator() { print(); }

    using duration_t = std::chrono::nanoseconds;
    struct FunctionStats {
        size_t call_count = 0;
        duration_t total_duration{0};
    };

    std::map<std::string, FunctionStats> data_;
    std::mutex data_mutex_; // for thread-safe access

public:
    static ProfilerAggregator& get() {
        static ProfilerAggregator instance;
        return instance;
    }
    static void log(const std::string& name, duration_t duration);
    static void print();
};

#endif
