#include <iostream>
#include <tr1/cmath>
#include <iomanip>

using namespace std;

int main() {

	cout << std::setprecision(15) << std::tr1::conf_hyperg(0.5,1.5,3.14) << endl;
	return 0;
}
