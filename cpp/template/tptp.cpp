#include <vector>
#include <array>
#include <iostream>
#include <complex>
#include <armadillo>
#include <type_traits>

template <template<typename T, typename ...> class Container, bool is_cplx = false>
struct Test {
	using Ret = typename std::conditional<is_cplx, std::complex<double>, double>::type;
	void print() { std::cout << typeid(Container<Ret>).name() << std::endl; }
};

template<typename, typename>
struct meta {};

template<typename A, template<typename ...> typename C, typename B>
struct meta<A, C<B>> {
    using type = C<A>;
};

//template<typename A, template<typename ...> typename C, typename B, typename N >
//struct meta<A, C<B,N>> {
//    using type = C<A, N>;
//};

//template<typename A, template<typename ...T> typename C, typename B>
//struct meta<A, C<B>> {
//    using type = C<A>;
//};

int main() {

	Test<arma::Col, true> vec_arma;
	vec_arma.print();

	Test<std::vector, true> vec_vec;
	vec_vec.print();

	std::cout << typeid(meta<std::complex<double>, arma::vec>::type).name() << std::endl;
	std::cout << typeid(meta<std::complex<double>, std::vector<double>>::type).name() << std::endl;

	//Test<std::array, true> vec_arr;

	return 0;
}
