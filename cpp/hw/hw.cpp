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

template <typename eT>
int dim(arma::Col<eT> const&) {
	return 1;
}

template <typename eT>
int dim(arma::Row<eT> const&) {
	return 1;
}

template <typename eT>
int dim(arma::Mat<eT> const& ) {
	return 2;
}

template <typename eT>
int dim(arma::Cube<eT> const& ) {
	return 3;
}






int sum(int i, int j, int k) {
	return i+j+k;
}

struct Test
{
	Test(): x(0) {}
	Test(int i): x(i) {}

	int x;
};

int main(int, char**argv) {
	Test t(;
	int a, b, c;

	sum(a,b,c);
	












	arma::Col<int> a = arma::Col<int>{3};
	a.print();

	arma::mat b(3,5);
	arma::cube c(2,3,4);

	cout << dim(a) << endl;
	cout << dim(c) << endl;
	cout << dim(b) << endl;

    return 0;
}
