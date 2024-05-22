#include "block_cyclic_handle.h"

#include <cstdio>
#include <cassert>
#include <unistd.h>


int main()
{
    MPI_Init(nullptr, nullptr);

    int nprocs, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    BlockCyclicHandle bcd(7, 11, 10, 12);

    for (int i = 0; i < nprocs; i++) {
        if (i == rank) {
            printf("rank = %i    pcoord = (%i,%i)    local size : %i x %i\n",
                    rank, bcd.ip(), bcd.jp(), bcd.ml(), bcd.nl());
        }
        usleep(10000);
    }
    if (rank == 0) printf("\n");


    for (int i = 0; i < nprocs; i++) {
        if (i == rank) {
            printf("rank = %i, l2g row:", rank);
            for (auto i : bcd.l2g_row_) {
                std::cout << " " << i;
            }
            std::cout << std::endl;
        }
        usleep(10000);
    }

    for (int i = 0; i < nprocs; i++) {
        if (i == rank) {
            printf("rank = %i, l2g col:", rank);
            for (auto i : bcd.l2g_col_) {
                std::cout << " " << i;
            }
            std::cout << std::endl;
        }
        usleep(10000);
    }

    int m = 7, n = 11;
    int mb = 4, nb = 3;

    std::vector<double> A_glb;
    std::vector<double> A_loc;

    BlockCyclicHandle bcd_loc(m, n, mb, nb);
    BlockCyclicHandle bcd_glb(m, n, m, n);

    if (rank == 0)
    {
        A_glb.resize(m*n);
        for (int i = 0; i < A_glb.size(); ++i) 
        {
            A_glb[i] = i;
        }

        printf("A (global): \n");
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                printf("%6.2f ", A_glb[i+j*m]);
            }
            printf("\n");
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    A_loc.resize(bcd_loc.ml() * bcd_loc.nl());
    Cpdgemr2d(m, n, A_glb.data(), 1, 1, bcd_glb.desc(), A_loc.data(), 1, 1, bcd_loc.desc(), bcd_glb.ctxt());

    for (int i = 0; i < nprocs; ++i)
    {
        if (i == rank)
        {
            // print A_loc
            printf("rank = %i    A (local): \n", rank);
            for (int i = 0; i < bcd_loc.ml(); ++i)
            {
                for (int j = 0; j < bcd_loc.nl(); ++j)
                {
                    printf("%6.2f ", A_loc[i+j*bcd_loc.ml()]);
                }
                printf("\n");
            }
        }
        usleep(10000);
    }

    std::vector<double> B;
    if (rank == 0)
    {
        B.resize(m*n, 0);
    }

    Cpdgemr2d(m, n, A_loc.data(), 1, 1, bcd_loc.desc(), B.data(), 1, 1, bcd_glb.desc(), bcd_glb.ctxt());

    if (rank == 0)
    {
        printf("B (global): \n");
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                printf("%6.2f ", B[i+j*m]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
}

