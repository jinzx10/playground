#include <iostream>
#include <vector>
#include <array>
#include <complex>
#include <type_traits>

template <size_t sz = 1, bool is_cplx = false>
struct Base
{
	using Val = typename std::conditional<is_cplx, std::complex<double>, double>::type;
	using Vec = std::array<Val, sz>;
};

template <size_t sz, bool is_cplx>
struct Derived : Base<sz,is_cplx>
{
	typename Base<sz, is_cplx>::Val val;
	typename Base<sz, is_cplx>::Vec vec;
};

template <size_t sz, bool is_cplx>
struct Derived2 : Base<sz,is_cplx>
{
	using typename Base<sz, is_cplx>::Val;
	using typename Base<sz, is_cplx>::Vec;

	Val val;
	Vec vec;
};

int main() {
	Derived<4,true> d;
	std::cout << typeid(d.val).name() << std::endl;
	std::cout << typeid(d.vec).name() << std::endl;

	Derived2<3,false> d2;
	std::cout << typeid(d2.val).name() << std::endl;
	std::cout << typeid(d2.vec).name() << std::endl;

	return 0;
}
