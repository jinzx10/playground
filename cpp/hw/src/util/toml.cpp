#include "toml.h"

#include "log.h"
#include "para/parallel_config.h"

#include <sstream>

namespace {

    Toml::table parse(const std::string& filename) {
        // Parse a TOML file into a toml::table
        toml::parse_result result = toml::parse_file(filename);
        if (!result) {
            Log::error("Failed to parse TOML file: {}", filename);
            Log::flush();
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        Log::debug("TOML file \"{}\" parsed successfully", filename);
        return result.table();
    }

    void bcast(toml::table& tbl) {
        // Broadcast a toml::table from rank-0 to all other ranks

        // Serialize the table to a string
        std::stringstream ss;
        ss << tbl;
        std::string content = ss.str();

        // Broadcast the size of the string
        int buffer_size = static_cast<int>(content.size());
        MPI_Bcast(&buffer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Broadcast the string
        if (ParallelConfig::get().world_rank() != 0) content.resize(buffer_size);
        MPI_Bcast(content.data(), buffer_size, MPI_CHAR, 0, MPI_COMM_WORLD);
        tbl = toml::parse(content).table();
    }
}

Toml::table Toml::parse_and_bcast(const std::string& filename) {
    toml::table tbl;
    if (ParallelConfig::get().world_rank() == 0) tbl = ::parse(filename);
    bcast(tbl);
    return tbl;
}
