#include <type_traits>
#include <sstream>
#include <iostream>
#include <string>
//#include "../utility/widgets.h"

using namespace std;


// read arguments from the command line
template <int N = 1>
void readargs(char** args, std::string& var) {
	var = args[N];
}

template <int N = 1, typename T>
void readargs(char** args, T& var) {
	std::stringstream ss(args[N]);
	ss >> var;
}

template <int N = 1, typename T, typename ...Ts>
void readargs(char** args, T& var, Ts& ...rest) {
	readargs<N>(args, var);
	readargs<N+1>(args, rest...);
}

int main(int argc, char* argv[]) {
	
	double x;
	int y;
	string z, w;
	int a[5];
	int* pa[5];

	cout << sizeof(a)/sizeof(int) << endl;
	cout << sizeof(pa)/sizeof(int*) << endl;
	cout << sizeof(argv) << endl;
	cout << sizeof(*argv) << endl;
	cout << sizeof(argv[0]) << endl;

	/*
	readargs(argv, x, y, z, w);

	std::cout << x << std::endl;
	std::cout << y << std::endl;
	std::cout << z << std::endl;
	std::cout << w << std::endl;
	*/


	return 0;
}
