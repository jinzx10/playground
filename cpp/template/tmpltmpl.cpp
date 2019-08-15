#include <vector>
#include <array>
#include <iostream>
#include <complex>
#include <armadillo>

template <template<typename T, typename ...> class Container, bool is_cplx = false>
class Test
{
	public:

		using Ret = std::conditional_t<is_cplx, std::complex<double>, double>;

		void print() {
			std::cout << typeid(Container<Ret>).name() << std::endl;
		}
};

int main() {

	Test<arma::Col, true> vec_arma;
	vec_arma.print();

	Test<std::vector, true> vec_vec;
	vec_vec.print();

	//Test<std::array, true> vec_arr;

	return 0;
}
