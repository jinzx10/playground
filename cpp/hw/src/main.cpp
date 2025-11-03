#include "util/function_profiler.h"
#include "para/parallel_config.h"
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

    // Initialize MPI with thread support
    int requested = MPI_THREAD_MULTIPLE;
    int provided;
    MPI_Init_thread(&argc, &argv, requested, &provided);
    if (provided < requested) {
        std::fprintf(stderr, "Could not obtain requested MPI thread support level\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    ParallelConfig& para = ParallelConfig::get();

    // Suppress the output other than the root process
    //if (para.world_rank() != 0) std::freopen("/dev/null", "w", stdout);

    para.setup(2, 2, 1);

    complexCalculation();
    complexCalculation();
    complexCalculation();
    databaseQuery();

    para.free();
    MPI_Finalize();
    return 0;
}
