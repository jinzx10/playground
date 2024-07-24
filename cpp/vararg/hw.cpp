#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <tuple>
#include <utility>

class Test {
public:
    template <typename ...Ts>
    Test(Ts... args) : shape_{args...} {}

    template <typename ...Ts>
    Test(std::vector<size_t> const& s) : shape_{s} {}

    template<typename... Ts>
    size_t index(Ts... args)
    {
        std::array<size_t, sizeof...(args)> indices = {args...};
        assert(sizeof...(args) == shape_.size() &&
               std::lexicographical_compare(indices.begin(), indices.end(), shape_.begin(), shape_.end()));

        size_t idx = 0;
        for (size_t i = 0; i < sizeof...(args); ++i)
        {
            idx = idx * shape_[i] + indices[i];
        }
        return idx;
    }

//private:

    std::vector<size_t> shape_;


    template<typename... Ts>
    size_t _index_c(Ts... args)
    {
        assert(sizeof...(args) == shape_.size() && std::vector<size_t>{args...} < shape_);
        return _index_c_impl(std::forward_as_tuple(args...), std::make_index_sequence<sizeof...(Ts)-1>{});
    }

    template <typename T>
    size_t _index_c_impl(std::tuple<T> const& tup, std::index_sequence<>)
    {
        return std::get<0>(tup);
    }

    template <typename ...Ts, size_t ...Ints>
    size_t _index_c_impl(std::tuple<Ts...> const& tup, std::index_sequence<Ints...>)
    {
        return _index_c_impl(std::forward_as_tuple(std::get<Ints>(tup)...), std::make_index_sequence<sizeof...(Ints)-1>{})
                * shape_[sizeof...(Ints)] + std::get<sizeof...(Ints)>(tup);
    }

    //size_t _index_f(size_t i) {
    //    assert (i < shape_.back());
    //    return i; 
    //}

    //template<typename... Ts>
    //size_t _index_f(size_t i, Ts... args)
    //{
    //    assert(i < shape_[shape_.size() - sizeof...(args) - 1]);
    //    return i + shape_[shape_.size() - sizeof...(args) - 1] * _index_f(args...);
    //}

};

std::vector<size_t> generate_shape(size_t n) {
    return std::vector<size_t>(n, 2);
}

int main() {

    std::cout << "enter dim: " << std::endl;
    size_t d = 0;

    std::cin >> d;

    Test t(generate_shape(d));

    size_t i0 = 0;
    size_t i1 = 0;
    size_t i2 = 0;
    size_t i3 = 0;
    size_t i4 = 0;
    size_t i5 = 0;
    size_t i6 = 0;
    size_t i7 = 0;
    size_t i8 = 0;
    size_t i9 = 0;

    std::cout << "enter ints as indices: " << std::endl;
    std::cin >> i0 >> i1 >> i2 >> i3 >> i4 >> i5 >> i6 >> i7 >> i8 >> i9;

    size_t index = 0;

    //index = ((i0 * n1 + i1) * n2 + i2) * n3 + i3;
    //index = t._index_c(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9);
    index = t.index(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9);


    std::cout << "index = " << index << std::endl;


    //std::cout << "c index: " << ((i0 * n1 + i1) * n2 + i2) * n3 + i3 << std::endl;
    //std::cout << "index(): " << t.index(i0, i1, i2, i3) << std::endl;
    //std::cout << "_index(): " << t._index_c(i0, i1, i2, i3) << std::endl;

    //std::cout << "f index: " << i0 + n0 * (i1 + n1 * (i2 + n2 * i3)) << std::endl;
    //std::cout << "_index_f(): " << t._index_f(i0, i1, i2, i3) << std::endl;

    //Test t2(n0);
    //std::cout << "index(): " << t2.index(i0) << std::endl;
    //std::cout << "_index(): " << t2._index_c(i0) << std::endl;

    return 0;

}
