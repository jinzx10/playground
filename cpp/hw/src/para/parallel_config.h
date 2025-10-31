#ifndef PARALLEL_CONFIG_H
#define PARALLEL_CONFIG_H

#include <mpi.h>
#include <mutex>

class ParallelConfig {
private: // singleton
    ParallelConfig() = default;
    ~ParallelConfig() { free(); }

    MPI_Comm image_comm_ = MPI_COMM_NULL;
    MPI_Comm kpool_comm_ = MPI_COMM_NULL;
    MPI_Comm bpool_comm_ = MPI_COMM_NULL;

    int world_size_ = -1;
    int image_size_ = -1;
    int kpool_size_ = -1;
    int bpool_size_ = -1;

    int world_rank_ = -1;
    int image_rank_ = -1;
    int kpool_rank_ = -1;
    int bpool_rank_ = -1;

    std::once_flag init_flag_;

    void do_setup(int num_images, int kpools_per_image, int bpools_per_kpool);

public:
    static ParallelConfig& get() {
        static ParallelConfig instance;
        return instance;
    }

    void setup(int num_images, int kpools_per_image, int bpools_per_kpool) {
        std::call_once(
            init_flag_,
            &ParallelConfig::do_setup,
            this, num_images, kpools_per_image, bpools_per_kpool
        );
    }

    void free();

    MPI_Comm image_comm() { return image_comm_; }
    MPI_Comm kpool_comm() { return kpool_comm_; }
    MPI_Comm bpool_comm() { return bpool_comm_; }

    int world_size() { return world_size_; }
    int image_size() { return image_size_; }
    int kpool_size() { return kpool_size_; }
    int bpool_size() { return bpool_size_; }

    int world_rank() { return world_rank_; }
    int image_rank() { return image_rank_; }
    int kpool_rank() { return kpool_rank_; }
    int bpool_rank() { return bpool_rank_; }
};

#endif
