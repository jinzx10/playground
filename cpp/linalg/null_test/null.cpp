#include <armadillo>

using namespace arma;
using namespace std;

int main() {

	mat Zt;
	Zt.load("Zt.dat");

	mat identity = zeros(size(Zt));
	identity.diag(2) = ones(Zt.n_rows);
	cout << "size: " << size(Zt) << std::endl;
	std::cout << "dev from I = " << norm(identity-Zt) << std::endl;
	mat ns;
	bool null_status = null(ns, Zt);
	cout << boolalpha << "null: " << null_status << endl;


	mat U, V;
	vec S;
	bool svd_status = svd(U, S, V, Zt);
	cout << boolalpha << "svd: " << svd_status << endl;


	return 0;
}
