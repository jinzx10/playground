#include "log.h"

#include "para/parallel_config.h"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include <string>
#include <cmath>

namespace {
    int num_digits(int n) {
        return static_cast<int>(std::floor(std::log10(n))) + 1;
    }
}

void Log::init() {
    int rank = ParallelConfig::get().world_rank();

    // (1) Create sinks
    std::vector<spdlog::sink_ptr> sinks;

    // (1.1) All processes get an individual, all-inclusive file sink
    // with filename "rank-<rank>.log" (width aligned)
    int rank_width = num_digits(ParallelConfig::get().world_size()-1);
    std::string filename = fmt::format("rank-{:0{}d}.log", rank, rank_width);

    // true: overwrite; false: append
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename, true);

    // Log every message up from debug level to the file
    file_sink->set_level(spdlog::level::debug);

    // [time.millisec] [level] message
    file_sink->set_pattern("[%T.%e] [%l] %v");

    sinks.push_back(file_sink);

    // (1.2) ONLY rank-0 gets a console sink
    if (rank == 0) {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

        // Keep console less verbose by including messages up from "info",
        // i.e., trace/debug-level messages are excluded.
        console_sink->set_level(spdlog::level::info);
        console_sink->set_pattern("%v");

        sinks.push_back(console_sink);
    }

    // (2) Create and set the logger as default
    std::string logger_name = "TBD"; // not actually used so far
    auto logger = std::make_shared<spdlog::logger>(logger_name, sinks.begin(), sinks.end());
    logger->set_level(spdlog::level::debug);
    spdlog::set_default_logger(logger);

    // (3) Give the first message
    debug("Logging starts on rank {}", rank);
    flush();
}


void Log::flush() {
    spdlog::default_logger()->flush();
}
