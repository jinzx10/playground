#include "parallel_2d.h"

#include <cassert>
#include <string>
#include <unistd.h>

extern "C"
{
    void Cpdgemr2d(int m, int n, double* A, int IA, int JA, int *descA, double* B, int IB, int JB, int *descB, int gcontext);
}

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

    int n_glb = 7;
    int nb = 2;

    std::vector<double> A_glb; // source
    std::vector<double> A_loc; // scattered
    std::vector<double> B_glb; // re-gathered

    Parallel_2D para_glb, para_loc;
    para_loc.init(n_glb, n_glb, nb, MPI_COMM_WORLD);
    para_glb.init(n_glb, n_glb, 2*n_glb, MPI_COMM_WORLD);

    // fill up the global matrix
    if (rank == 0)
    {
        A_glb.resize(n_glb * n_glb);
        for (int i = 0; i < A_glb.size(); ++i) 
        {
            A_glb[i] = i;
        }
    }
    print_glb(A_glb.data(), n_glb, n_glb, "original (global)");


    // allocate space for local matrix
    A_loc.resize(para_loc.nloc);

    // scatter
    Cpdgemr2d(n_glb, n_glb, A_glb.data(), 1, 1, para_glb.desc, A_loc.data(), 1, 1, para_loc.desc, para_glb.blacs_ctxt);

    print_loc(A_loc.data(), para_loc.nrow, para_loc.ncol, "scattered (local)");


    // allocate space for re-gathered global matrix
    if (rank == 0)
    {
        B_glb.resize(n_glb * n_glb);
    }

    Cpdgemr2d(n_glb, n_glb, A_loc.data(), 1, 1, para_loc.desc, B_glb.data(), 1, 1, para_glb.desc, para_loc.blacs_ctxt);

    print_glb(B_glb.data(), n_glb, n_glb, "gathered (global)");

    MPI_Finalize();
}

