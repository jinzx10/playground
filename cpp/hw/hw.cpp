#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <string>
#include <armadillo>

using namespace std;

void help(char info[]) {
	std::cout << info << std::endl;
}

int main() {

	arma::mat a = arma::ones(5,30);

	double* ptr = a.memptr();

	std::cout << ptr[10] << std::endl;

    return 0;
}
