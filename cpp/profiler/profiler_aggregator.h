#ifndef PROFILER_AGGREGATOR_H
#define PROFILER_AGGREGATOR_H

#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <cstdio>

class ProfilerAggregator {
private:
    struct FunctionStats {
        size_t call_count = 0;
        std::chrono::nanoseconds total_duration{0};
    };

    std::map<std::string, FunctionStats> data_;

    // Mutex for thread-safe access
    std::mutex data_mutex_;

    ProfilerAggregator() = default;
    ~ProfilerAggregator() { print(); }

public:
    static ProfilerAggregator& get_instance() {
        static ProfilerAggregator instance;
        return instance;
    }
    static void log(const std::string& name, std::chrono::nanoseconds duration);
    static void print();
};

#endif
