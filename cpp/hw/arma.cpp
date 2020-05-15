#include <iostream>
#include <armadillo>
#include <chrono>
#include "../utility/widgets.h"
#include "../utility/arma_helper.h"

#define GET_VARIABLE_NAME(Variable) (#Variable)

using namespace arma;

template <typename T1, typename T2>
auto jr(T1 const& m1, T2 const& m2) {
	return join_rows(m1, m2);
}

template <typename T, typename ...Ts>
auto jr(T const& m, Ts const& ...args) {
	return jr(m, jr(args...)).eval();
}

int main() {
	mat a = randu(3,2);
	vec b = zeros(3);
	mat c = ones(3,4);
	mat d;

	Stopwatch sw;

	//mat j0 = join_r(a,mat{b},c);
	//j0.print();

	//auto f = [&](mat const& i, mat const& j, mat const& k, mat const& l) {return jr(i,j,k,l);};
	//sw.timeit(10, f, a, b, c, a);

	d = join_d(a, b, c);
	d.print();


	//j.print();

	return 0;
}
