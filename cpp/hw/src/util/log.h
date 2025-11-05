#ifndef LOG_H
#define LOG_H

#include "spdlog/spdlog.h"

namespace Log {

    using spdlog::error;
    using spdlog::warn;
    using spdlog::info;
    using spdlog::debug;

    void init();
    void flush();
}

#endif
