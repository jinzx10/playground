#ifndef __READ_ARGUMENTS_H__
#define __READ_ARGUMENTS_H__

#include <sstream>

template <int N = 1, typename T>
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
	readargs<N+1>(args, vars...);
}

#endif
