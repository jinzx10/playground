#include "parallel_2d.h"
#include "scalapack_connector.h"
#include "blacs_connector.h"
#include <unistd.h>

int main() {

    //===========================================
    //              setup
    //===========================================

    MPI_Init(nullptr, nullptr);

    int nprocs = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //*******************************************
    // create & distribute a global matrix
    //*******************************************
    
    std::vector<double> A_global;
    std::vector<double> A_dist;

    // global size
    int nrow = 7;
    int ncol = 9;

    if (rank == 0) {
        // global matrix
        A_global.resize(nrow * ncol);
        for (int i = 0; i < nrow; i++) {
            for (int j = 0; j < ncol; j++) {
                A_global[i + j * nrow] = i + 0.1 * j;
            }
        }

        // print the global matrix
        std::cout << "Global matrix:" << std::endl;
        for (int i = 0; i < nrow; i++) {
            for (int j = 0; j < ncol; j++) {
                printf("%5.1f", A_global[i + j * nrow]);
            }
            printf("\n");
        }
    }
    if (rank == 0) { printf("\n"); }

    Parallel_2D p2d_global;
    p2d_global.init(nrow, ncol, std::max(nrow, ncol), MPI_COMM_WORLD);

    // block size
    int nb = 2;

    Parallel_2D p2d_dist;
    p2d_dist.set(nrow, ncol, nb, p2d_global.comm_2D, p2d_global.blacs_ctxt);

    // allocate space for local matrix
    A_dist.resize(p2d_dist.nloc);

    // distribute the global matrix
    Cpdgemr2d(nrow, ncol, A_global.data(), 1, 1, p2d_global.desc, A_dist.data(), 1, 1, p2d_dist.desc, p2d_global.blacs_ctxt);

    // print the local matrix
    for (int irank = 0; irank < nprocs; irank++) {
        if (rank == irank) {
            printf("fully distributed matrix on rank = %i/%i\n", rank, nprocs);
            for (int i = 0; i < p2d_dist.nrow; i++) {
                for (int j = 0; j < p2d_dist.ncol; j++) {
                    printf("%5.1f", A_dist[i + j * p2d_dist.nrow]);
                }
                printf("\n");
            }
        }
        usleep(10000);
    }
    if (rank == 0) { printf("\n"); }


    //===========================================
    //              pool
    //===========================================

    std::vector<double> A_pool;

    // number of pools
    int npool = 2;

    // split the communicator
    int ipool = rank % npool;
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, ipool, rank, &comm);

    // rank within the pool
    int rank_pool = 0;
    MPI_Comm_rank(comm, &rank_pool);

    Parallel_2D p2d_pool;
    p2d_pool.init(nrow, ncol, nb, comm);


    for (int irank = 0; irank < nprocs; irank++) {
        if (rank == irank) {
            printf("ipool = %i   rank in pool = %i  redistributed matrix size = (%i,%i)\n",
                    ipool, rank_pool, p2d_pool.nrow, p2d_pool.ncol);
        }
        usleep(10000);
    }
    if (rank == 0) { printf("\n"); }

    // allocate space for local matrix
    A_pool.resize(p2d_pool.nloc);


    // see the following pages for details:
    // https://www.netlib.org/scalapack/slug/node164.html
    // https://www.netlib.org/scalapack/slug/node167.html
    int desc_pool[9];
    std::copy(p2d_pool.desc, p2d_pool.desc + 9, desc_pool);
    if (ipool != 0) {
        desc_pool[1] = -1; // this is the key!
    }

    // redistribute the global matrix
    Cpdgemr2d(nrow, ncol, A_dist.data(), 1, 1, p2d_dist.desc, A_pool.data(), 1, 1, desc_pool, p2d_dist.blacs_ctxt);

    // print the redistributed matrix
    for (int irank = 0; irank < nprocs; irank++) {
        if (rank == irank) {
            printf("ipool = %i   rank in pool = %i\n", ipool, rank_pool);
            for (int i = 0; i < p2d_pool.nrow; i++) {
                for (int j = 0; j < p2d_pool.ncol; j++) {
                    printf("%5.1f", A_pool[i + j * p2d_pool.nrow]);
                }
                printf("\n");
            }
        }
        usleep(10000);
    }

    MPI_Finalize();
}
