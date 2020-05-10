#include <armadillo>

using namespace arma;
using namespace std;

int main() {

	mat Zt;
	Zt.load("Zt.dat");

	mat ns;
	bool null_status = null(ns, Zt);
	cout << boolalpha << "null: " << null_status << endl;


	/*
	vec sv;
	bool svd_status = svd(sv, Zt);
	cout << boolalpha << "svd: " << svd_status << endl;
	*/


	return 0;
}
