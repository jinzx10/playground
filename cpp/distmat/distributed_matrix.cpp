#include <type_traits>
#include <complex>
#include <array>
#include <vector>

extern "C" {
    // BLACS
    void Cblacs_pinfo(int*, int*);
    void Cblacs_get(int, int, int*);
    void Cblacs_barrier(int, char*);
    void Cblacs_gridinit(int* , char*, int, int);
    void Cblacs_gridinfo(int, int*, int*, int*, int*);
    void Cblacs_gridexit(int);
    void Cdgesd2d(int, int, int, double*, int, int, int);
    void Cdgerv2d(int, int, int, double*, int, int, int);

    // PBLAS
    void pdgemm_(char*, char*, int*, int*, int*, double*, double*, int*, int*, int*, double*, int*, int*, int*, double*, double*, int*, int*, int*);

    // ScaLAPACK utilities
    int numroc_(int const*, int const*, int const*, int const*, int const*);
    void descinit_(int*, int const*, int const*, int*, int*, int*, int*, int*, int*, int*);
}

template <typename T,
         typename = typename std::enable_if<
         std::is_same<T, float>::value ||
         std::is_same<T, double>::value ||
         std::is_same<T, std::complex<float>>::value ||
         std::is_same<T, std::complex<double>>::value>::type
         >
class DistributedMatrix {

public:
    DistributedMatrix(): desc_{}, data_{} {}
    ~DistributedMatrix() {}

    inline void clear() { data_.clear(); }
    inline bool empty() const { return data_.empty(); }


    //****** Processor grid is handled internally ******

    // Resizes the matrix for a distributed m x n matrix with block size mb x nb.
    void resize(const int m, const int n, const int mb, const int nb);

    // Scatters data (interpreted as a m x n column-major matrix) from root process into a block-cyclic
    // distribution with block size mb x nb.
    void scatter_from(const int root, std::vector<T>& data, const int m, const int n, const int mb, const int nb);

    // Gathers the block-cyclicly distributed matrix into a column-major matrix on root process.
    // data on other processes are untouched.
    void gather_to(const int root, std::vector<T>& data, int& m, int& n) const;


private:
    std::array<int, 9> desc_;
    std::vector<T> data_;
};


int main() {

    DistributedMatrix<double> A, B, C;



    return 0;
}
