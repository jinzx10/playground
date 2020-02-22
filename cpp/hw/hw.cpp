#include <iostream>
#include <string>
#include <sstream>
#include <vector>

template <typename T>
void tq(T& val) {
	std::cout << "tq" << std::endl;
}

template <typename T, typename ...Ts, typename R>
void tq(T&, Ts&...args, R& r) {
	std::cout << "tq" << std::endl;
	tq(args..., r);
}

/*
template <typename T, int N = 0>
void pour2(T& val) {
	std::cout << N << std::endl;
}

template <typename T, typename ...Ts, int N = 0>
void pour2(T& val, Ts& ...args) {
	std::cout << N << std::endl;
    pour2<Ts...>(args...);
}
*/


int main() {

	int i,j,k,l;
	tq(i,j,k,l);


    return 0;
}
/*
#include <iostream>
#include <string>
#include <sstream>
#include <armadillo>
#include "../utility/widgets.h"

using namespace arma;

int main(int, char** argv) {

	std::string a;
	int b;

	readargs(argv, a, b);

	std::cout << a << std::endl;
	std::cout << b << std::endl;




	return 0;
}

*/
