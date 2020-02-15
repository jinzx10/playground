#include <iostream>
#include <string>
#include <sstream>


template <int N, typename T>
void readargs(char** args, T& var) {
	std::stringstream ss;
	ss << args[N];
	ss >> var;
}


template <int N = 1, typename T, typename ...Ts>
void readargs(char** args, T& var, Ts& ...vars) {
	std::stringstream ss;
	ss << args[N];
	ss >> var;
	readargs<N+1, Ts...>(args, vars...);
}

int main(int, char** argv) {

	int u;
	char v;
	double w;
	readargs(argv, u, v, w);

	std::cout << "u = " << u << std::endl;
	std::cout << "v = " << v << std::endl;
	std::cout << "w = " << w << std::endl;

	return 0;
}

