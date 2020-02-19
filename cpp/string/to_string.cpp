#include <string>
#include <iostream>
#include "../utility/widgets.h"

int main(int, char** argv) {
	double g = 0.0;
	unsigned long long sz = 0;

	readargs(argv, g, sz);

	std::cout << std::to_string(g) << std::endl;
	std::cout << std::to_string(sz) << std::endl;
	return 0;
}
