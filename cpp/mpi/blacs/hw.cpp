#include <cstdio>
#include <mpi.h>
#include <unistd.h>
#include <array>
#include <cmath>
#include <cassert>
#include <vector>

extern "C"
{
    int Csys2blacs_handle(MPI_Comm SysCtxt);
	void Cblacs_pinfo(int *myid, int *nprocs);
	void Cblacs_get(int icontxt, int what, int *val);
	void Cblacs_gridmap(int* icontxt, int *usermap, int ldumap, int nprow, int npcol);
	void Cblacs_gridinfo(int icontxt, int* nprow, int *npcol, int *myprow, int *mypcol);
    void Cblacs_gridinit(int* icontxt, char* layout, int nprow, int npcol);
    int Cblacs_pnum(int icontxt, int prow, int pcol);
    void Cblacs_pcoord(int icontxt, int pnum, int *prow, int *pcol);
	void Cblacs_exit(int icontxt);

	void blacs_pinfo_(int *myid, int *nprocs);
	void blacs_get_(int* icontxt, int* what, int *val);
	void blacs_gridinit_( int *ictxt, const char *order, const int *nprow, const int *npcol );
	void blacs_gridinfo_( const int *ictxt, int *nprow, int *npcol, int *myprow, int *mypcol );

	int numroc_(const int *n, const int *nb, const int *iproc, const int *srcproc, const int *nprocs);
	void descinit_(int *desc, const int *m, const int *n, const int *mb, const int *nb, const int *irsrc, const int *icsrc, const int *ictxt, const int *lld, int *info);

}


class BlockCyclicHandle
{
public:

    enum class ProcGrid { Square, Row, Column };

    /**
     * @brief
     *
     *
     */
    BlockCyclicHandle(
        int mg,
        int ng,
        int mb,
        int nb,
        MPI_Comm comm = MPI_COMM_WORLD,
        ProcGrid layout = ProcGrid::Square
    );

    int blacs_ctxt() const { return desc_[1]; }

    int mg() const { return desc_[2]; }
    int ng() const { return desc_[3]; }

    int mb() const { return desc_[4]; }
    int nb() const { return desc_[5]; }

    int ml() const { return ml_; }
    int nl() const { return nl_; }

    int mp() const { return mp_; }
    int np() const { return np_; }

    int ip() const { return ip_; }
    int jp() const { return jp_; }

    bool in_this_process(int i, int j) const;

//private:

    /// size of the local matrix
    int ml_;
    int nl_;

    /// size of the process grid
    int mp_;
    int np_;

    /// process coordinates
    int ip_;
    int jp_;

    /// ScaLAPACK descriptor
    std::array<int, 9> desc_;

    /// global-to-local index mapping
    std::vector<int> g2l_row_;
    std::vector<int> g2l_col_;

    /// local-to-global index mapping
    std::vector<int> l2g_row_;
    std::vector<int> l2g_col_;

    /// greatest common divisor
    int _gcd(int a, int b) { return b == 0 ? a : _gcd(b, a % b); }

    /**
     * @brief Factorizes w = p * q such that p and q have the largest possible
     * greatest common divisor.
     *
     */
    void _fact(int w, int& p, int& q);
};


void BlockCyclicHandle::_fact(int w, int& p, int& q)
{
    int p_max = static_cast<int>(std::sqrt(w + 0.5));
    int gcd_max = -1;
    for (int i = 1; i <= p_max; ++i)
    {
        int j = w / i;
        if (i * j != w) { continue; }

        int gcd = _gcd(i, j);
        if (gcd >= gcd_max)
        {
            p = i;
            q = j;
            gcd_max = gcd;
        }
    }
}


bool BlockCyclicHandle::in_this_process(int i, int j) const
{
    return g2l_row_[i] != -1 && g2l_col_[j] != -1;
}


BlockCyclicHandle::BlockCyclicHandle(
    int mg,
    int ng,
    int mb,
    int nb,
    MPI_Comm comm,
    ProcGrid layout
):
    g2l_row_(mg, -1),
    g2l_col_(ng, -1)
{
    int num_proc = 0;
    MPI_Comm_size(comm, &num_proc);

    // determine the number of rows and columns of the process grid
    mp_ = 1, np_ = 1;
    switch (layout)
    {
        case ProcGrid::Row   : np_ = num_proc; break;
        case ProcGrid::Column: mp_ = num_proc; break;
        case ProcGrid::Square: _fact(num_proc, mp_, np_); break;
    }

    // initialize the BLACS grid and get the process coordinates
    int ctxt = Csys2blacs_handle(comm);
    char order = 'C'; // always column-major
    Cblacs_gridinit(&ctxt, &order, mp_, np_);
    Cblacs_gridinfo(ctxt, &mp_, &np_, &ip_, &jp_);

    // local number of rows and columns
    int zero = 0;
    ml_ = numroc_(&mg, &mb, &ip_, &zero, &mp_);
    nl_ = numroc_(&ng, &nb, &jp_, &zero, &np_);

    // initialize ScaLAPACK descriptor
    descinit_(desc_.data(), &mg, &ng, &mb, &nb, &zero, &zero, &ctxt, &ml_, &zero);

    // generate the global-to-local and local-to-global index mappings
    l2g_row_.resize(ml_);
    for (int i = 0; i < ml_; ++i)
    {
        l2g_row_[i] = i / mb * mp_ * mb + ip_ * mb + i % mb;
        g2l_row_[l2g_row_[i]] = i;
    }

    l2g_col_.resize(nl_);
    for (int j = 0; j < nl_; ++j)
    {
        l2g_col_[j] = j / nb * np_ * nb + jp_ * nb + j % nb;
        g2l_col_[l2g_col_[j]] = j;
    }
}


int main() {
    MPI_Init(nullptr, nullptr);

    int nprocs, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    BlockCyclicHandle bcd(7, 11, 4, 6);

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

    MPI_Finalize();
}
