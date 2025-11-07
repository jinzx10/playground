#include "profiler_aggregator.h"
#include "util/log.h"
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

    Log::info("");
    Log::info("-------------------- FUNCTION PROFILING RESULTS --------------------");
    Log::info("   Count   | Total Duration (s) | Avg Duration (s) | Function Name  ");
    Log::info("--------------------------------------------------------------------");

    for (const auto& pair : instance.data_) {
        const auto& name = pair.first;
        const auto& stats = pair.second;

        // convert to seconds
        double total_s = std::chrono::duration<double>(stats.total_duration).count();
        double avg_s = total_s / stats.call_count;

        Log::info(" {:9} | {:18.3f} | {:16.3f} | {}",
                    stats.call_count, total_s, avg_s, name.c_str());
    }
    Log::info("--------------------------------------------------------------------");
}
