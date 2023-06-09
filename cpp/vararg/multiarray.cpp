#include <iostream>
#include <cassert>


template <typename T, char Format='C'>
class MultiArray {

public:

    template <typename ...Ts>
    MultiArray(Ts... args) {
        _build<Ts...>(args...);
    }

    T& operator()(int i) {
        return data_[i];
    }

    template <typename ...Ts>
    T& operator()(Ts ...args) {
        assert(sizeof...(args) == dim_);
        assert(_index(args...) < nelem_);
        return data_[_index(args...)];
    }


//private:

    template <typename ...Ts>
    void _build(Ts ...args) {
        delete[] size_;
        delete[] stride_;
        delete[] data_;

        dim_ = sizeof...(args);

        size_ = new int[dim_];
        _set_size(args...);

        nelem_ = 1;
        for (int i = 0; i != dim_; ++i) {
            nelem_ *= size_[i];
        }

        stride_ = new int[dim_];

        format_ = Format;
        if (Format == 'C') {
            stride_[dim_-1] = 1;
            for (int i = dim_-2; i >= 0; --i) {
                stride_[i] = stride_[i+1]*size_[i+1];
            }
        } else if (Format == 'F') {
            stride_[0] = 1;
            for (int i = 1; i <= dim_-1; ++i) {
                stride_[i] = stride_[i-1]*size_[i-1];
            }
        } else {
            std::cerr << "Error: Unknown format" << std::endl;
            exit(1);
        }

        data_ = new T[nelem_];
    }

    void _set_size(int n) {
        size_[dim_-1] = n;
    }

    template <typename ...Ts>
    inline void _set_size(int n, Ts...args) {
        size_[dim_-1-sizeof...(args)] = n;
        _set_size(args...);
    }

    inline int _index(int i) {
        return i*stride_[dim_-1];
    }

    template <typename ...Ts>
    inline int _index(int i, Ts...args) {
        return i*stride_[dim_-1-sizeof...(args)] + _index(args...);
    }

    char format_ = 'C';
    int dim_ = 0;
    int nelem_ = 0;
    int* size_ = nullptr;
    int* stride_ = nullptr;
    T* data_ = nullptr;
};


int main() {

    MultiArray<float> ccube(4,5,6,7, 8);
    std::cout << ccube._index(1,2,3,4, 4) << std::endl;

    for (int i = 0; i != 4; ++i) {
        std::cout << ccube.stride_[i] << std::endl;
    }
    std::cout << std::endl;

    MultiArray<double,'F'> fcube(4,5,6,7);
    std::cout << fcube._index(1,2,3,4) << std::endl;

    for (int i = 0; i != 4; ++i) {
        std::cout << fcube.stride_[i] << std::endl;
    }
    std::cout << std::endl;


    return 0;
}