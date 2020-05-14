#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <string>
#include <type_traits>
#include "../utility/widgets.h"
#include "../utility/arma_helper.h"

using namespace arma;
using namespace std;


int main(int, char**argv) {
	
	arma::Col<int> col1 = arma::Col<int>{3};
	arma::Row<char> row1 = {'a','c','x'};

	arma::mat mat1(3,5), mat2, mat3;
	arma::cube cube1(2,3,4);

	dim(col1).print();
	dim(mat1).print();
	dim(row1).print();
	dim(cube1).print();

	set_size({2,8}, mat1);
	mat1.print();

	set_size({3,3}, mat1, mat2, mat3);

	cout << endl;
	mat1.print();
	cout << endl;
	cout << endl;
	mat2.print();
	cout << endl;
	cout << endl;
	mat3.print();
	cout << endl;



    return 0;
}
