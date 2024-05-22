#include "block_cyclic_handle.h"

#include <cmath>
#include <unistd.h>

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
    ProcGrid layout,
    MPI_Comm comm
):
    ml_(0), nl_(0),
    desc_{0},
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
    int lld = std::max(ml_, 1); // suppress warning when ml_ = 0
    descinit_(desc_.data(), &mg, &ng, &mb, &nb, &zero, &zero, &ctxt, &lld, &zero);

    // generate the global-to-local and local-to-global index mappings
    l2g_row_.resize(ml_);
    for (int i = 0; i < ml_; ++i)
    {
        l2g_row_[i] = (i / mb * mp_ + ip_) * mb + i % mb;
        g2l_row_[l2g_row_[i]] = i;
    }

    l2g_col_.resize(nl_);
    for (int j = 0; j < nl_; ++j)
    {
        l2g_col_[j] = (j / nb * np_ + jp_) * nb + j % nb;
        g2l_col_[l2g_col_[j]] = j;
    }
}

