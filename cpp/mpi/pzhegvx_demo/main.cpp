#include "aux.h"

#include <complex>
#include <vector>
#include <cmath>
#include <cstdio>
#include <unistd.h>
#include <cassert>

const double pi = std::acos(-1.0);
const std::complex<double> imag_i{0.0, 1.0};


void print(double d) { printf("%8.5f  ", d); }
void print(const std::complex<double>& d) { printf("(%8.5f, %8.5f)  ", d.real(), d.imag()); }


template<typename T>
void print_glb(T* A, int m, int n, const std::string& msg = "") {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        if (!msg.empty()) { printf("%s\n", msg.c_str()); }

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                print(A[i+j*m]);
            }
            printf("\n");
        }
    }
}


template <typename T>
void print_loc(T* A, int m, int n, const std::string& msg = "") {
    int rank, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int i = 0; i < nprocs; ++i) {
        if (i == rank) {
            printf("%s    rank = %i: \n", msg.c_str(), rank);
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    print(A[i+j*m]);
                }
                printf("\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(10000);
    }
}


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
    int ictxt)
{
    int nprow = 0, npcol = 0, iprow = 0, ipcol = 0;
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &iprow, &ipcol);

    const char jobz = 'V', range = 'I', uplo = 'U';
    const int itype = 1, il = 1, iu = nv, one = 1;

    // sizes of work space (-1 to trigger work space query)
    int lwork = -1, lrwork = -1, liwork = -1;

    // number of eigenvalues & eigenvectors found
    int m = 0, nz = 0;

    int info = 0;

    double abstol = pdlamch_(&ictxt, "S");
    double orfac = -1;

    std::vector<std::complex<double>> work(1);
    std::vector<double> rwork(3);
    std::vector<int> iwork(1);

    std::vector<int> ifail(sz);
    std::vector<int> iclustr(2 * nprow * npcol);
    std::vector<double> gap(nprow * npcol);

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

    //printf("(%i, %i): lwork = %i   lrwork = %i   liwork = %i\n",
    //        iprow, ipcol, lwork, lrwork, liwork);

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

    // size of the process grid
    int nprow = 0, npcol = 0;

    // number of trials
    int nt = 0;

    // number of eigenvectors to look for
    int nv = 0;

    /************************************************************
     *                          setup
     ************************************************************/
    if (rank == 0) {

        do {
            printf("Enter the number of rows of the process grid (must be a divisor of %i): ", nprocs);
            fflush(stdout);
            scanf("%i", &nprow);
        } while (nprocs % nprow != 0);
        npcol = nprocs / nprow;

        do {
            printf("Enter the size of the global matrix (must be positive): ");
            fflush(stdout);
            scanf("%i", &sz);
        } while (sz <= 0);

        do {
            printf("Enter the distribution block size (must be positive): ");
            fflush(stdout);
            scanf("%i", &nb);
        } while (nb <= 0);

        do {
            printf("Enter the number of eigenvectors to look for (must be positive): ");
            fflush(stdout);
            scanf("%i", &nv);
        } while (nv <= 0);

        do {
            printf("Enter the number of trials (must be positive): ");
            fflush(stdout);
            scanf("%i", &nt);
        } while (nt <= 0);
    }

    MPI_Bcast(&nprow, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&npcol, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nb, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nv, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // global variables used to construct the problem
    // A X = B X diag(w)
    std::vector<double> w;
    std::vector<std::complex<double>> X;
    std::vector<std::complex<double>> A;
    std::vector<std::complex<double>> B;

    // eigenvectors and eigenvalues
    std::vector<std::complex<double>> evec;
    std::vector<double> eval;

    if (rank == 0) {

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

        //int itype = 1;
        //char jobz = 'V';
        //char range = 'A';
        //char uplo = 'U';
        //double vl = 0.0, vu = 0.0;
        //int il = 0, iu = 0;
        //double abstol = 2.0 * dlamch_("S");
        //int m = 0;

        //int lwork = -1;
        //std::vector<std::complex<double>> work(1);
        //std::vector<double> rwork(7 * sz);
        //std::vector<int> iwork(5 * sz);
        //std::vector<int> ifail(sz);
        //int info = 0;

        //eval.resize(sz);
        //evec.resize(sz * sz);

        //// work space query
        //zhegvx_(&itype, &jobz, &range, &uplo, &sz,
        //        A.data(), &sz, B.data(), &sz,
        //        &vl, &vu, &il, &iu, &abstol,
        //        &m, eval.data(), evec.data(), &sz,
        //        work.data(), &lwork, rwork.data(), iwork.data(),
        //        ifail.data(), &info);
        //lwork = static_cast<size_t>(work[0].real());
        //work.resize(lwork);

        //// diagonalization
        //zhegvx_(&itype, &jobz, &range, &uplo, &sz,
        //        A.data(), &sz, B.data(), &sz,
        //        &vl, &vu, &il, &iu, &abstol,
        //        &m, eval.data(), evec.data(), &sz,
        //        work.data(), &lwork, rwork.data(), iwork.data(),
        //        ifail.data(), &info);

        //print_glb(w.data(), sz, 1, "referece eigenvalues");
        //print_glb(eval.data(), sz, 1, "computed eigenvalues");
        //print_glb(X.data(), sz, sz, "reference eigenvectors");
        //print_glb(evec.data(), sz, sz, "computed eigenvectors");
    }

    int ictxt = Csys2blacs_handle(MPI_COMM_WORLD);
    char order = 'R';
    Cblacs_gridinit(&ictxt, &order, nprow, npcol);

    // coordinates of the process
    int iprow = 0, ipcol = 0;
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &iprow, &ipcol);

    const int zero = 0;

    // number of local rows and columns
    int nrow = numroc_(&sz, &nb, &iprow, &zero, &nprow);
    int ncol = numroc_(&sz, &nb, &ipcol, &zero, &npcol);

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

    //print_glb(A.data(), sz, sz);
    //print_loc(A_loc.data(), nrow, ncol, "A_loc");

    //print_glb(B.data(), sz, sz);
    //print_loc(B_loc.data(), nrow, ncol, "B_loc");
    double start = MPI_Wtime();

    for (int i = 0; i < nt; ++i) {
        pzhegvx_once(sz, A_loc.data(), B_loc.data(), nrow, ncol, desc_loc,
                     nv, eval.data(), evec.data(), desc_loc, ictxt);
    }
    double time_elapsed = MPI_Wtime() - start;

    //print_loc(evec.data(), nrow, ncol_v, "evec");
    //print_glb(eval.data(), nv, 1);

    if (rank == 0) {
        printf("average time elapsed = %8.5f\n", time_elapsed / nt);

        double err_eval_max = 0.0;
        for (int i = 0; i < nv; ++i) {
            err_eval_max = std::max(err_eval_max, std::abs(w[i] - eval[i]));
        }
        printf("max error in eigenvalues = %8.5e\n", err_eval_max);
    }

    MPI_Finalize();

    return 0;
}
