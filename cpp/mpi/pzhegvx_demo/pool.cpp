#include "aux.h"

#include <complex>
#include <vector>
#include <cmath>
#include <cstdio>
#include <unistd.h>
#include <cassert>

const double pi{std::acos(-1.0)};
const std::complex<double> imag_i{0.0, 1.0};

void pzhegvx_once(
    int sz,
    std::complex<double>* A_loc,
    std::complex<double>* B_loc,
    int nrow,
    int ncol,
    int* desc,
    int nv,
    double* w,
    std::complex<double>* V_loc,
    int* desc_v,
    int nprocs,
    int ictxt)
{
    const char jobz = 'V', range = 'I', uplo = 'U';
    const int itype = 1, il = 1, iu = nv, one = 1;

    // sizes of work space (-1 to trigger work space query)
    int lwork = -1, lrwork = -1, liwork = -1;

    // number of eigenvalues & eigenvectors found
    int m = 0, nz = 0;

    int info = 0;

    double abstol = pdlamch_(&ictxt, "S");
    double orfac = 0.01;

    std::vector<std::complex<double>> work(1);
    std::vector<double> rwork(3);
    std::vector<int> iwork(1);

    std::vector<int> ifail(sz);
    std::vector<int> iclustr(2 * nprocs);
    std::vector<double> gap(nprocs);

    const double vl = 0.0, vu = 0.0;

    std::vector<std::complex<double>> A_copy(A_loc, A_loc + nrow * ncol);
    std::vector<std::complex<double>> B_copy(B_loc, B_loc + nrow * ncol);

    // work space query
    pzhegvx_(&itype, &jobz, &range, &uplo,
            &sz,
            A_copy.data(), &one, &one, desc, B_copy.data(), &one, &one, desc,
            &vl, &vu, &il, &iu, &abstol,
            &m, &nz, w,
            &orfac,
            V_loc, &one, &one, desc_v,
            work.data(), &lwork,
            rwork.data(), &lrwork,
            iwork.data(), &liwork,
            ifail.data(), iclustr.data(), gap.data(), &info);

    lwork = static_cast<int>(work[0].real());
    lrwork = static_cast<int>(rwork[0]);
    liwork = iwork[0];

    work.resize(lwork);
    rwork.resize(lrwork);
    iwork.resize(liwork);

    // diagonalization
    pzhegvx_(&itype, &jobz, &range, &uplo,
            &sz,
            A_copy.data(), &one, &one, desc, B_copy.data(), &one, &one, desc,
            &vl, &vu, &il, &iu, &abstol,
            &m, &nz, w,
            &orfac,
            V_loc, &one, &one, desc_v,
            work.data(), &lwork,
            rwork.data(), &lrwork,
            iwork.data(), &liwork,
            ifail.data(), iclustr.data(), gap.data(), &info);

    assert(m == nv);
    assert(nz == nv);
    //assert(info == 0);
}


/* 
 * This program demonstrates how to use the ScaLAPACK to compute the
 * eigendecomposition of a complex Hermitian matrix.
 *
 */
int main() {

    MPI_Init(nullptr, nullptr);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // size of the global matrix
    int sz = 0;

    // block size of 2d block-cyclic distribution
    int nb = 0;

    // number of eigenvectors to look for (nbands)
    int nv = 0;

    // number of diagonalizations run in parallel (kpar)
    int npool = 0;

    // total number of matrices to diagonalize (nk)
    int nm = 0;

    // number of matrices in each pool
    int nm_pool = 0;

    // size of the BLACS grid for each pool
    int nprocs_pool = 0, nprow_pool = 0, npcol_pool = 0;


    /************************************************************
     *              read parameters from stdin
     ************************************************************/
    if (rank == 0) {

        do {
            printf("Enter the size of the global matrix (must be positive): ");
            fflush(stdout);
        } while (scanf("%i", &sz) == 0 || sz <= 0);

        do {
            printf("Enter the distribution block size (must be positive): ");
            fflush(stdout);
        } while (scanf("%i", &nb) == 0 || nb <= 0);

        do {
            printf("Enter the number of eigenvectors to look for (must be positive): ");
            fflush(stdout);
        } while (scanf("%i", &nv) == 0 || nv <= 0);

        do {
            printf("Enter the number of diagonalizations run in parallel (must be a divisor of %i): ", nprocs);
            fflush(stdout);
        } while (scanf("%i", &npool) == 0 || npool <= 0 || nprocs % npool != 0);
        nprocs_pool = nprocs / npool;

        do {
            printf("Enter the total number of matrices to diagonalize (must a multiple of %i): ", npool);
            fflush(stdout);
        } while (scanf("%i", &nm) == 0 || nm <= 0 || nm % npool != 0);
        nm_pool = nm / npool;

        do {
            printf("Enter the number of rows of the process grid (must be a divisor of %i): ", nprocs_pool);
            fflush(stdout);
        } while (scanf("%i", &nprow_pool) == 0 || nprow_pool <= 0 || nprocs_pool % nprow_pool != 0);
        npcol_pool = nprocs_pool / nprow_pool;

        printf("size of the global matrix = %i\n", sz);
        printf("block size = %i\n", nb);
        printf("number of eigenvectors = %i\n", nv);
        printf("total number of matrices = %i\n", nm);
        printf("number of matrices in each pool = %i\n", nm_pool);
        printf("number of diagonalizations run in parallel = %i\n", npool);
        printf("size of the BLACS grid = (%i, %i)\n", nprow_pool, npcol_pool);
    }

    MPI_Bcast(&sz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nb, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nv, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nm, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&npool, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nm_pool, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nprow_pool, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&npcol_pool, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nprocs_pool, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // communicator for diagonalization
    MPI_Comm comm_d;
    MPI_Comm_split(MPI_COMM_WORLD, rank % npool, rank, &comm_d);

    int rank_d = 0;
    MPI_Comm_rank(comm_d, &rank_d);

    // A X = B X diag(w)
    std::vector<double> w;
    std::vector<std::complex<double>> X;
    std::vector<std::complex<double>> A;
    std::vector<std::complex<double>> B;

    // eigenvectors and eigenvalues
    std::vector<std::complex<double>> evec;
    std::vector<double> eval;

    if (rank_d == 0) {

        w.resize(sz);
        X.resize(sz * sz);
        A.resize(sz * sz);
        B.resize(sz * sz);

        const double fac = 1.0 / std::sqrt(sz);

        // test with a simplified case: B = I
        for (int j = 0; j < sz; ++j) {
            w[j] = static_cast<double>(j) / sz;
            B[j * sz + j] = 1.0; // identity
            for (int k = 0; k < sz; ++k) {
                X[j + k*sz] = fac * std::exp(2.0 * pi * j * k / sz * imag_i);
            }
        }

        // tmp = X * diag(w)
        std::vector<std::complex<double>> tmp{X};
        for (int k = 0; k < sz; ++k)
            for (int j = 0; j < sz; ++j)
                tmp[j + k*sz] *= w[k];

        std::complex<double> alpha{1.0, 0.0};
        std::complex<double> beta{0.0, 0.0};

        // A = tmp * X^H
        zgemm_("N", "C", &sz, &sz, &sz,
               &alpha, tmp.data(), &sz, X.data(), &sz,
               &beta, A.data(), &sz);
    }

    int ictxt = Csys2blacs_handle(comm_d);
    char order = 'R';
    Cblacs_gridinit(&ictxt, &order, nprow_pool, npcol_pool);

    // coordinates of the process
    int iprow = 0, ipcol = 0;
    Cblacs_gridinfo(ictxt, &nprow_pool, &npcol_pool, &iprow, &ipcol);

    const int zero = 0;

    // number of local rows and columns
    int nrow = numroc_(&sz, &nb, &iprow, &zero, &nprow_pool);
    int ncol = numroc_(&sz, &nb, &ipcol, &zero, &npcol_pool);

    // local matrices
    std::vector<std::complex<double>> A_loc(nrow * ncol);
    std::vector<std::complex<double>> B_loc(nrow * ncol);


    int info = 0;

    // descriptor for the distributed matrix
    int desc_loc[9];
    descinit_(desc_loc, &sz, &sz, &nb, &nb,
              &zero, &zero, &ictxt, &nrow, &info);

    // descriptor for the global matrix
    int desc_glb[9];
    descinit_(desc_glb, &sz, &sz, &sz, &sz,
              &zero, &zero, &ictxt, &sz, &info);

    // NOTE: eigenvectors should have the same global size as the
    // matrix A/B even if we are only interested in a subset of them.
    // See the ScaLAPACK documentation of pzhegvx on argument "W", "Z"
    // and the "Alignment requirements" section for details.
    eval.resize(sz);
    evec.resize(nrow * ncol);

    // scatter A and B to local matrices
    Cpzgemr2d(sz, sz,
              A.data(), 1, 1, desc_glb,
              A_loc.data(), 1, 1, desc_loc,
              ictxt);

    Cpzgemr2d(sz, sz,
              B.data(), 1, 1, desc_glb,
              B_loc.data(), 1, 1, desc_loc,
              ictxt);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int i = 0; i < nm_pool; ++i) {
        pzhegvx_once(sz, A_loc.data(), B_loc.data(), nrow, ncol, desc_loc,
                     nv, eval.data(), evec.data(), desc_loc, nprocs_pool, ictxt);
        if (rank_d == 0) {
            printf("group %i: %i/%i done\n", rank % npool, i+1, nm_pool);
        }
    }

    double time_elapsed = MPI_Wtime() - start;
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(10000);

    if (rank == 0) {
        printf("total time elapsed = %8.5f\n", time_elapsed);

        double err_eval_max = 0.0;
        for (int i = 0; i < nv; ++i) {
            err_eval_max = std::max(err_eval_max, std::abs(w[i] - eval[i]));
        }
        printf("max error in eigenvalues = %8.5e\n", err_eval_max);
    }

    MPI_Finalize();

    return 0;
}
