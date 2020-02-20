#include <iostream>
#include <string>
#include <sstream>
#include <armadillo>
#include "../utility/widgets.h"

using namespace arma;

int main(int, char** argv) {

	vec a = ones(10);
	int i,j;

	readargs(argv, i, j);

	a.print();
	std::cout << std::endl;

	a.insert_rows(i,1);
	a.print();
	std::cout << std::endl;
	
	a.insert_rows(j,1);
	a.print();
	std::cout << std::endl;
	


	return 0;
}

