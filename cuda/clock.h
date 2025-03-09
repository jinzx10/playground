#ifndef CLOCK_H
#define CLOCK_H

#include <chrono>

#define CLOCK(call, msg)                                        \
{                                                               \
    using iclock = std::chrono::high_resolution_clock;          \
    auto start = iclock::now();                                 \
    call;                                                       \
    std::chrono::duration<double> dur = iclock::now() - start;  \
    printf("%s: %8.3f\n", msg, dur.count());                    \
}


#endif
