#include <iostream>
#include <armadillo>

using namespace arma;

template <typename T>
struct Solve
{
	void solve() {
		//T& m = static_cast<T&>(*this);
		//arma::eig_sym( m.eigval, m.eigvec, m.H );
		T* m = static_cast<T*>(this);
		arma::eig_sym( m->eigval, m->eigvec, m->H );
	}
};

struct tls : Solve<tls>
{
	tls(mat H_): H(H_) {}
	mat H;
	vec eigval;
	mat eigvec;
};

int main() {
	mat H = {{2, 1}, {1, 0}};
	tls model(H);
	model.solve();
	std::cout << "H = " << std::endl;
	H.print();
	std::cout << "val = " << std::endl;
	model.eigval.print();
	std::cout << "vec = " << std::endl;
	model.eigvec.print();

	return 0;
}
