#include "parallel_config.h"

#include <cstdio>
#include <cassert>

namespace {
    void split_comm(
        MPI_Comm parent_comm,
        int num_child,
        MPI_Comm& child_comm,
        int& child_rank,
        int& child_size,
        const std::string& info
    ) {
        int parent_size = 0;
        int parent_rank = 0;
        MPI_Comm_size(parent_comm, &parent_size);
        MPI_Comm_rank(parent_comm, &parent_rank);

        // sanity check
        assert(num_child <= parent_size);
        if (parent_size % num_child != 0) {
            std::fprintf(
                stderr,
                "%s warning: %s: parent_size (%d) "
                "is not divisible by num_child (%d)\n",
                __func__, info.c_str(), parent_size, num_child
            );
        }

        // Calculate split parameters (block partition)
        // color controls subset assignment, key controls ordering within subset
        int sz_block = parent_size / num_child + (parent_size % num_child != 0);
        int color = parent_rank / sz_block;
        int key = parent_rank % sz_block;

        // Split and store results
        MPI_Comm_split(parent_comm, color, key, &child_comm);
        MPI_Comm_rank(child_comm, &child_rank);
        MPI_Comm_size(child_comm, &child_size);
    }
} // anonymous namespace


ParallelConfig::ParallelConfig() {
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
}


void ParallelConfig::free() {
    if (bpool_comm_ != MPI_COMM_NULL) MPI_Comm_free(&bpool_comm_);
    if (kpool_comm_ != MPI_COMM_NULL) MPI_Comm_free(&kpool_comm_);
    if (image_comm_ != MPI_COMM_NULL) MPI_Comm_free(&image_comm_);
    image_comm_ = kpool_comm_ = bpool_comm_ = MPI_COMM_NULL;
}


void ParallelConfig::do_setup(
    int num_images,
    int kpools_per_image,
    int bpools_per_kpool
) {
    split_comm(
        MPI_COMM_WORLD,
        num_images,
        image_comm_,
        image_rank_,
        image_size_,
        "world -> images"
    );

    split_comm(
        image_comm_,
        kpools_per_image,
        kpool_comm_,
        kpool_rank_,
        kpool_size_,
        "image -> kpools"
    );

    split_comm(
        kpool_comm_,
        bpools_per_kpool,
        bpool_comm_,
        bpool_rank_,
        bpool_size_,
        "kpool -> bands"
    );
}

