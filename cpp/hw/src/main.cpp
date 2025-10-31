#include "util/function_profiler.h"
#include <thread>   

void complexCalculation() {
    PROFILE_FUNCTION();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

void databaseQuery() {
    PROFILE_FUNCTION();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

int main() {
    PROFILE_FUNCTION();
    complexCalculation();
    complexCalculation();
    complexCalculation();
    databaseQuery();

    ProfilerAggregator::get_instance().print();
    
    return 0;
}
