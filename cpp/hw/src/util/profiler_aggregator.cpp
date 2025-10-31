#include "profiler_aggregator.h"
#include <cstdio>

void ProfilerAggregator::log(const std::string& name, duration_t duration) {
    ProfilerAggregator& instance = ProfilerAggregator::get();
    std::lock_guard<std::mutex> lock(instance.data_mutex_);

    FunctionStats& stats = instance.data_[name];
    stats.call_count++;
    stats.total_duration += duration;
}

void ProfilerAggregator::print() {
    ProfilerAggregator& instance = ProfilerAggregator::get();
    std::lock_guard<std::mutex> lock(instance.data_mutex_);

    std::printf("\n");
    std::printf("------------------ FUNCTION PROFILING RESULTS ------------------\n");
    std::printf("  Count  | Total Duration (s) | Avg Duration (s) | Function Name\n");
    std::printf("----------------------------------------------------------------\n");

    for (const auto& pair : instance.data_) {
        const auto& name = pair.first;
        const auto& stats = pair.second;

        // convert to seconds
        double total_s = std::chrono::duration<double>(stats.total_duration).count();
        double avg_s = total_s / stats.call_count;

        std::printf(" %7zu | %18.3f | %16.3f | %s\n",
                    stats.call_count, total_s, avg_s, name.c_str());
    }
    std::printf("----------------------------------------------------------------\n");
}
