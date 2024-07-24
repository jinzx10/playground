#include "block_cyclic_handle.h"

#include <cassert>
#include <unistd.h>

void print_loc(double* A, int m, int n, const std::string& msg, MPI_Comm comm = MPI_COMM_WORLD)
{
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    for (int i = 0; i < nprocs; ++i)
    {
        if (i == rank)
        {
            printf("rank = %i    %s: \n", rank, msg.c_str());
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    printf("%6.2f ", A[i+j*m]);
                }
                printf("\n");
            }
        }
        usleep(10000);
    }
}

void print_glb(double* A, int m, int n, const std::string& msg, MPI_Comm comm = MPI_COMM_WORLD, int iprint = 0)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == iprint)
    {
        printf("%s: \n", msg.c_str());
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                printf("%6.2f ", A[i+j*m]);
            }
            printf("\n");
        }
    }
}


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
    int mb = 1, nb = 1;

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
    }

    print_glb(A_glb.data(), m, n, "original (global)");

    A_loc.resize(bcd_loc.ml() * bcd_loc.nl());
    Cpdgemr2d(m, n, A_glb.data(), 1, 1, bcd_glb.desc(), A_loc.data(), 1, 1, bcd_loc.desc(), bcd_glb.ctxt());

    print_loc(A_loc.data(), bcd_loc.ml(), bcd_loc.nl(), "scattered (local)");

    std::vector<double> B;
    if (rank == 0)
    {
        B.resize(m*n, 0);
    }

    Cpdgemr2d(m, n, A_loc.data(), 1, 1, bcd_loc.desc(), B.data(), 1, 1, bcd_glb.desc(), bcd_glb.ctxt());

    print_glb(B.data(), m, n, "gathered (global)");

    MPI_Finalize();
}

