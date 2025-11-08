#include "parallel_config.h"

#include "util/log.h"

namespace {

    void split(
        MPI_Comm comm,
        int num_sub,
        MPI_Comm& intra_sub_comm,
        int& intra_sub_rank,
        int& intra_sub_size,
        MPI_Comm& inter_sub_comm,
        int& inter_sub_rank,
        int& inter_sub_size,
        const std::string& info
    ) {
        int size = 0;
        int rank = 0;
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);

        // sanity check
        // TODO: do we need it to be divisible?
        if (size % num_sub != 0) {
            Log::error("{}: {}: {} is not divisible by {}.",
                        __func__, info, size, num_sub);
            Log::flush();
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int size_sub = size / num_sub;

        // MPI_Comm_split(comm, color, key, sub_comm);
        // color controls subset assignment
        // key controls ordering within subset

        // The intra communicator is a block partition of the processes
        int intra_color = rank / size_sub;
        MPI_Comm_split(comm, intra_color, rank, &intra_sub_comm);
        MPI_Comm_rank(intra_sub_comm, &intra_sub_rank);
        MPI_Comm_size(intra_sub_comm, &intra_sub_size);

        // The inter communicator is a cyclic partition of the processes
        int inter_color = rank % size_sub;
        MPI_Comm_split(comm, inter_color, rank, &inter_sub_comm);
        MPI_Comm_rank(inter_sub_comm, &inter_sub_rank);
        MPI_Comm_size(inter_sub_comm, &inter_sub_size);

        // Here the term "intra" and "inter" are used based on whether
        // the processes in sub-communicators have contiguous original rank.
        // Suppose there are 8 processes and num_sub is 2, then
        // the intra communicator works for the communication within
        // {0,1,2,3} or {4,5,6,7}, and the inter communicator works for
        // the communication within {0,4}, {1,5}, {2,6}, {3,7}.
        //
        //      <-- intra -->
        //      0   1   2   3   ^
        //                      | inter
        //      4   5   6   7   v
        //
        // The parallelization scheme will assume that intra-communications
        // are faster than inter-communications. This may not be true in reality,
        // i.e., processes with close ranks might be far away in the physical
        // topology. Users might need to pass extra launching arguments to `mpirun`
        // for better performance.
    }
} // anonymous namespace


ParallelConfig::ParallelConfig() {
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
}


void ParallelConfig::free() {
    if (intra_bpool_comm_ != MPI_COMM_NULL) MPI_Comm_free(&intra_bpool_comm_);
    if (intra_kpool_comm_ != MPI_COMM_NULL) MPI_Comm_free(&intra_kpool_comm_);
    if (intra_image_comm_ != MPI_COMM_NULL) MPI_Comm_free(&intra_image_comm_);
    intra_image_comm_ = intra_kpool_comm_ = intra_bpool_comm_ = MPI_COMM_NULL;

    if (inter_bpool_comm_ != MPI_COMM_NULL) MPI_Comm_free(&inter_bpool_comm_);
    if (inter_kpool_comm_ != MPI_COMM_NULL) MPI_Comm_free(&inter_kpool_comm_);
    if (inter_image_comm_ != MPI_COMM_NULL) MPI_Comm_free(&inter_image_comm_);
    inter_image_comm_ = inter_kpool_comm_ = inter_bpool_comm_ = MPI_COMM_NULL;
}


void ParallelConfig::setup(
    int num_images,
    int kpools_per_image,
    int bpools_per_kpool
) {
    split(
        MPI_COMM_WORLD,
        num_images,
        intra_image_comm_,
        intra_image_rank_,
        intra_image_size_,
        inter_image_comm_,
        inter_image_rank_,
        inter_image_size_,
        "world -> images"
    );

    split(
        intra_image_comm_,
        kpools_per_image,
        intra_kpool_comm_,
        intra_kpool_rank_,
        intra_kpool_size_,
        inter_kpool_comm_,
        inter_kpool_rank_,
        inter_kpool_size_,
        "image -> kpools"
    );

    split(
        intra_kpool_comm_,
        bpools_per_kpool,
        intra_bpool_comm_,
        intra_bpool_rank_,
        intra_bpool_size_,
        inter_bpool_comm_,
        inter_bpool_rank_,
        inter_bpool_size_,
        "kpool -> bpools"
    );
}

