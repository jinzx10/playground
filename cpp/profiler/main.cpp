#include "function_profiler.h" // Assuming you put the setup in a header file
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
    complexCalculation();
    complexCalculation();
    complexCalculation();
    databaseQuery();

    //ProfilerAggregator::get_instance().print();
    
    // The results are automatically printed when ProfilerAggregator's 
    // static instance is destroyed (at program exit).
    return 0;
}
