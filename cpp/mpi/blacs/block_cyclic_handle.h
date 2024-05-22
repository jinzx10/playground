#ifndef BLOCK_CYCLIC_HANDLE_H
#define BLOCK_CYCLIC_HANDLE_H

#include "scalapack.h"

#include <vector>
#include <array>

class BlockCyclicHandle
{
public:

    enum class ProcGrid { Square, Row, Column };

    /**
     * @brief Initializes a block-cyclic distribution handle.
     *
     * @param   mg      number of rows in the global matrix
     * @param   ng      number of columns in the global matrix
     * @param   mb      number of rows in a block
     * @param   nb      number of columns in a block
     * @param   layout  process grid layout, could be Row, Column, or Square
     * @param   comm    MPI communicator over which BLACS grid is created
     *
     * @note The BLACS grid always exhausts the number of available processes in
     * the given communicator. The size of the grid is determined by "layout". If
     * "layout" is Row, the grid has only one row and as many columns as there are
     * processes. If "layout" is Column, the grid has only one column; if "layout"
     * is Square, the grid is made to have the largest possible greatest common
     * divisor between the number of rows and columns.
     *
     * @note If comm is a disjoint communicator created by MPI_Comm_split, the
     * resulting BLACS grid will also be disjoint.
     *
     */
    BlockCyclicHandle(
        int mg,
        int ng,
        int mb,
        int nb,
        ProcGrid layout = ProcGrid::Square,
        MPI_Comm comm = MPI_COMM_WORLD
    );


    int ctxt() const { return desc_[1]; }

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

    int* desc() { return desc_.data(); }
    // scalapack functions take int* instead of const int*
    // so we cannot make this function const

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
    static int _gcd(int a, int b) { return b == 0 ? a : _gcd(b, a % b); }

    /**
     * @brief Factorizes w = p * q such that p and q have the largest possible
     * greatest common divisor.
     *
     */
    static void _fact(int w, int& p, int& q);
};


#endif
