#include <armadillo>

using namespace arma;
using namespace std;

int main() {

	mat Zt, ovl;
	Zt.load("Zt.dat");
	ovl.load("ovl2.dat");

	mat ns;
	bool status = null(ns, Zt);

	cout << boolalpha << status << endl;


	return 0;
}
