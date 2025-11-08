#ifndef PARALLEL_CONFIG_H
#define PARALLEL_CONFIG_H

#include <mpi.h>

class ParallelConfig {
private: // singleton
    ParallelConfig();
    ~ParallelConfig() { free(); }

    int world_size_ = -1;
    int world_rank_ = -1;

    MPI_Comm intra_image_comm_ = MPI_COMM_NULL;
    MPI_Comm intra_kpool_comm_ = MPI_COMM_NULL;
    MPI_Comm intra_bpool_comm_ = MPI_COMM_NULL;

    int intra_image_size_ = -1;
    int intra_kpool_size_ = -1;
    int intra_bpool_size_ = -1;

    int intra_image_rank_ = -1;
    int intra_kpool_rank_ = -1;
    int intra_bpool_rank_ = -1;

    MPI_Comm inter_image_comm_ = MPI_COMM_NULL;
    MPI_Comm inter_kpool_comm_ = MPI_COMM_NULL;
    MPI_Comm inter_bpool_comm_ = MPI_COMM_NULL;

    int inter_image_size_ = -1;
    int inter_kpool_size_ = -1;
    int inter_bpool_size_ = -1;

    int inter_image_rank_ = -1;
    int inter_kpool_rank_ = -1;
    int inter_bpool_rank_ = -1;

public:
    static ParallelConfig& get() {
        static ParallelConfig instance;
        return instance;
    }

    void setup(int num_images, int kpools_per_image, int bpools_per_kpool);
    void free();

    int world_size() const { return world_size_; }
    int world_rank() const { return world_rank_; }

    MPI_Comm intra_image_comm() const { return intra_image_comm_; }
    MPI_Comm intra_kpool_comm() const { return intra_kpool_comm_; }
    MPI_Comm intra_bpool_comm() const { return intra_bpool_comm_; }

    int intra_image_size() const { return intra_image_size_; }
    int intra_kpool_size() const { return intra_kpool_size_; }
    int intra_bpool_size() const { return intra_bpool_size_; }

    int intra_image_rank() const { return intra_image_rank_; }
    int intra_kpool_rank() const { return intra_kpool_rank_; }
    int intra_bpool_rank() const { return intra_bpool_rank_; }

    MPI_Comm inter_image_comm() const { return inter_image_comm_; }
    MPI_Comm inter_kpool_comm() const { return inter_kpool_comm_; }
    MPI_Comm inter_bpool_comm() const { return inter_bpool_comm_; }

    int inter_image_size() const { return inter_image_size_; }
    int inter_kpool_size() const { return inter_kpool_size_; }
    int inter_bpool_size() const { return inter_bpool_size_; }

    int inter_image_rank() const { return inter_image_rank_; }
    int inter_kpool_rank() const { return inter_kpool_rank_; }
    int inter_bpool_rank() const { return inter_bpool_rank_; }
};

#endif
