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

int main(int argc, char** argv) {
	
	double x;
	string str1, str2;
	int y;

	readargs(argv, str1, x, str2, y);

	std::cout << x << std::endl;
	std::cout << y << std::endl;
	std::cout << str1 << std::endl;
	std::cout << str2 << std::endl;


	return 0;
}
