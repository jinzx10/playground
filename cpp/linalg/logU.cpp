#include <armadillo>
#include "../utility/math_helper.h"

using namespace arma;

int main() {

	uword sz = 5;
	mat a = randu(sz, sz);
	std::cout << "a = " << std::endl;
	a.print();

	a = orth_lowdin(a);

	a.print();
	
	mat err = a*a.t() - eye(sz,sz);
	std::cout << "deviation from orthogonal: " << accu(abs(err%err)) << std::endl;

	cx_mat u, s, ls;

	schur(u,s,conv_to<cx_mat>::from(a));
	std::cout << "u = " << std::endl;
	u.print();
	std::cout << "s = " << std::endl;
	s.print();
	cx_vec d = s.diag() / abs(s.diag());
	ls = u * diagmat(log(d)) * u.t();
	std::cout << "logU: " << std::endl;
	ls.print();

	cx_mat la = logmat(a);
	la.print();
	mat l1 = real(logmat(a));
	std::cout << "logmat: " << std::endl;
	l1.print();


	return 0;
}
