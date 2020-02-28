#include <iostream>
#include <armadillo>
#include <chrono>

#define GET_VARIABLE_NAME(Variable) (#Variable)

using namespace arma;
using iclock = std::chrono::high_resolution_clock;


int main() {

	int sz = 5;	
	vec a = randu(sz);
	vec b = randu(sz);
	sp_mat s;
	s = diagmat(join_cols(vec{0}, a));

	s(span(1,sz), 0) = b;
	s(0, span(1,sz)) = b.t();

	s.print();

	conv_to<mat>::from(s).print();


	return 0;
}
