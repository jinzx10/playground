#include "util/function_profiler.h"
#include <thread>   
#include <mpi.h>

void complexCalculation() {
    PROFILE_FUNCTION();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

void databaseQuery() {
    PROFILE_FUNCTION();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

int main(int argc, char** argv) {
    PROFILE_FUNCTION();

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0) {
        freopen("/dev/null", "w", stdout);
    }

    complexCalculation();
    complexCalculation();
    complexCalculation();
    databaseQuery();

    //ProfilerAggregator::get_instance().print();
    
    MPI_Finalize();
    return 0;
}
