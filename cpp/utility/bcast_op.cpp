#include <armadillo>
#include "bcast_op.h"

using namespace arma;

template <typename T1, typename T2>
struct power
{
	//template <typename U1, typename U2>
	//auto operator()(U1 v, U2 i) { 
	//	return pow(v, i); 
	//}

	auto operator()(Col<T1> v, T2 i) -> Col<T1> { return arma::pow(v, i); }
	T1 operator()(T1 v, T2 i) { return std::pow(v, i); }
};


struct test
{
	template <typename T1, typename T2>
	auto operator()(T1 m, T2 n) {
		return pow(m,n);
	}
};

int main() {
	uword m = 3;

	vec a = randu(m);
	
	a.print();
	bcast_op(a, a.t(), std::plus<>()).print();
	bcast_op(a.t(), a, std::minus<>()).print();

	uvec u = {0,1,2};
	uvec v = {1,2,3};

	//bcast_op(u, v.t(), power<uword, uword>()).print();
	//bcast_op(u, v.t(), test()).print();



	return 0;
}
