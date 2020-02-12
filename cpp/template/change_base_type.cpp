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

template <typename ...>
using void_t = void;

template<typename, typename>
struct replace_base_type {};

template<typename A, template<typename ...> typename C, typename B>
struct replace_base_type<A, C<B>> {
    using type = C<A>;
};

int main() {

	Test<arma::Col, true> vec_arma;
	vec_arma.print();

	Test<std::vector, true> vec_vec;
	vec_vec.print();

	std::cout << typeid(replace_base_type<std::complex<double>, arma::vec>::type).name() << std::endl;
	std::cout << typeid(replace_base_type<std::complex<double>, std::vector<double>>::type).name() << std::endl;


	//Test<std::array, true> vec_arr;

	return 0;
}
