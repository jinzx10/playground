#include "para/parallel_config.h"
#include "util/function_profiler.h"
#include "util/log.h"
#include "util/toml.h"

#include <mpi.h>
#include <thread> // merely used in test code; not necessary
#include <iostream>

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

    // Initialize logger
    Log::init();

    // Check thread support
    if (provided < requested) {
        Log::error("Could not obtain requested MPI thread support level");
        Log::flush();
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Image, kpool and bpool setup
    ParallelConfig::get().setup(2, 2, 1);

    complexCalculation();
    complexCalculation();
    complexCalculation();
    databaseQuery();

    Toml::table tbl = Toml::parse_and_bcast("sp_pbe0.toml");
    std::string prefix = *(tbl["env"]["out_prefix"].value<std::string>());
    Log::info("out_prefix is {}", prefix);


    ParallelConfig::get().free();
    Log::flush();
    MPI_Finalize();
    return 0;
}
